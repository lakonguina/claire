"""Nightly job: refit noise models and hyperparameters by EM, then rebuild derived state by replay."""

import asyncio
import logging
import math
from collections import defaultdict

from app import engine, params as params_mod
from app.db import AsyncConnection, execute, fetch, migrate, pool
from app.model.family import FAMILIES, reported_family
from app.model.length import self_report_variance
from app.model.reaction import Item, dawid_skene
from app.model.reading import fit_mixture
from app.model.stats import grid_mean_var, mean_var, normalize
from app.params import Params, load_params

log = logging.getLogger(__name__)
DERIVED = ["turn_inference", "conversation_state", "theta_posteriors", "alpha_posteriors", "user_reading", "beta_counts"]
MIN_WEIGHT = 20.0


async def finalized_turns(conn: AsyncConnection) -> list[dict]:
    """Every closed turn, with its first reply, verdict and click, and its inferred state."""
    return await fetch(conn, """
        SELECT t.*, c.host, c.assistant_model AS conv_model, i.length_mean, i.reading_probs, s.family_probs,
               s.kappa_mean, s.speed_mean,
               (SELECT extract(epoch FROM r.created_at - t.created_at) FROM replies r WHERE r.turn_id = t.id
                ORDER BY r.created_at LIMIT 1) AS duration,
               (SELECT r.reply_words_exact FROM replies r WHERE r.turn_id = t.id ORDER BY r.created_at LIMIT 1) AS exact,
               (SELECT observed FROM outcomes o WHERE o.turn_id = t.id AND o.source = 'assistant' LIMIT 1) AS verdict,
               (SELECT last_reply_words FROM outcomes o WHERE o.turn_id = t.id AND o.source = 'assistant' LIMIT 1) AS self_report,
               (SELECT row_to_json(k) FROM clicks k WHERE k.turn_id = t.id ORDER BY k.created_at LIMIT 1) AS click,
               EXISTS (SELECT 1 FROM buttons_shown b WHERE b.turn_id = t.id) AS any_shown,
               (SELECT outcome FROM gold_labels g WHERE g.turn_id = t.id ORDER BY g.created_at DESC LIMIT 1) AS gold
        FROM turns t JOIN conversations c ON c.id = t.conversation_id
        JOIN turn_inference i ON i.turn_id = t.id
        JOIN conversation_state s ON s.conversation_id = t.conversation_id""")


def refit_reactions(turns: list[dict], p: Params) -> None:
    items = []
    for t in turns:
        rp = t["reading_probs"]
        obs = engine.observations(t["verdict"], t["family_probs"], t["click"], t["any_shown"],
                                  [rp["read"], rp["skim"], rp["away"]])
        items.append(Item(obs, t["gold"]))
    if len(items) >= 50:
        p.confusions, _, _ = dawid_skene(items, p.confusions)


def refit_sensors(turns: list[dict], p: Params) -> None:
    writing, requested, fallback = [], [], ([], [])
    for t in turns:
        w = (t["duration"] or 0) - p.overhead
        s2 = math.log(w) + t["speed_mean"] if w > 0 else None
        s3 = math.log(t["requested_words"]) + t["kappa_mean"] if t["requested_words"] else None
        if t["exact"]:
            truth = math.log(t["exact"])
            if s2 is not None:
                writing.append(s2 - truth)
            if s3 is not None:
                requested.append(s3 - truth)
        elif t["self_report"] and s3 is not None:
            fallback[0].append(math.log(t["self_report"]))
            fallback[1].append(s3)
    if len(writing) >= MIN_WEIGHT:
        p.noise["writing_time"] = math.sqrt(max(mean_var(writing)[1], 0.01))
    if len(requested) >= MIN_WEIGHT:
        p.noise["requested"] = math.sqrt(max(mean_var(requested)[1], 0.01))
    if len(fallback[0]) >= MIN_WEIGHT:
        p.noise["self_report"] = math.sqrt(self_report_variance(*fallback, p.noise["requested"] ** 2))
    widget = [float(t["duration"]) - math.exp(math.log(t["exact"]) - t["speed_mean"])
              for t in turns if t["exact"] and t["duration"]]
    if len(widget) >= MIN_WEIGHT:
        widget.sort()
        p.overhead = max(widget[len(widget) // 2], 0.5)


def refit_families(turns: list[dict], conversations: list[dict], p: Params, theta_means: dict) -> None:
    for f in FAMILIES:
        speeds, speed_w, biases, bias_w = [], [], [], []
        for t in turns:
            weight = t["family_probs"][f]
            w = (t["duration"] or 0) - p.overhead
            if w > 0:
                speeds.append(t["length_mean"] - math.log(w))
                speed_w.append(weight)
            key = (t["user_id"], t["request_type"])
            if key in theta_means:
                biases.append(t["length_mean"] - math.log(t["needed_words"]) - theta_means[key])
                bias_w.append(weight)
        fam = p.family[f]
        if sum(speed_w) >= MIN_WEIGHT:
            mu, var = mean_var(speeds, speed_w)
            fam.update(speed_mu=mu, speed_sd=math.sqrt(max(var, 0.01)))
        if sum(bias_w) >= MIN_WEIGHT:
            b, var = mean_var(biases, bias_w)
            fam.update(b=b, w=max(var, 0.01))
        kw = [c["family_probs"][f] for c in conversations]
        if sum(kw) >= MIN_WEIGHT:
            fam["kappa"] = mean_var([c["kappa_mean"] for c in conversations], kw)[0]
    center = sum(p.family[f]["b"] for f in FAMILIES) / len(FAMILIES)  # mu0 absorbs the population level
    for f in FAMILIES:
        p.family[f]["b"] -= center

    by_host, by_report = defaultdict(list), defaultdict(list)
    for c in conversations:
        by_host[c["host"]].append(c["family_probs"])
        if (r := reported_family(c["assistant_model"])) is not None:
            by_report[r].append(c["family_probs"])
    average = lambda probs: dict(zip(FAMILIES, normalize([sum(x[f] for x in probs) + 0.5 for f in FAMILIES])))
    p.host_prior = {h: average(v) for h, v in by_host.items() if len(v) >= MIN_WEIGHT}
    for r, v in by_report.items():
        if len(v) >= MIN_WEIGHT:
            p.reliability[r] = average(v)


def refit_reading(turns: list[dict], rho: dict[int, float], p: Params) -> None:
    data = [(math.log(t["reading_probs"]["seconds"]), t["length_mean"] - rho.get(t["user_id"], 0.0)) for t in turns]
    if len(data) >= MIN_WEIGHT:
        p.reading, _ = fit_mixture(data, p.reading)


def refit_hyper(alphas: dict[int, list], thetas: dict[tuple, list], p: Params) -> None:
    if len(alphas) < 5:
        return
    a_stats = {u: grid_mean_var(g) for u, g in alphas.items()}
    means = [m for m, _ in a_stats.values()]
    p.hyper.mu0, spread = mean_var(means)
    p.hyper.sigma_alpha = math.sqrt(spread + sum(v for _, v in a_stats.values()) / len(a_stats))
    residuals = defaultdict(list)
    for (u, t), g in thetas.items():
        if u in a_stats:
            residuals[t].append(grid_mean_var(g)[0] - a_stats[u][0])
    p.hyper.gamma = {t: mean_var(r)[0] for t, r in residuals.items() if len(r) >= 3}
    if len(p.hyper.gamma) >= 2:
        p.hyper.sigma_gamma = math.sqrt(max(mean_var(list(p.hyper.gamma.values()))[1], 0.01))
    eta = [r - p.hyper.gamma.get(t, 0.0) for t, rs in residuals.items() for r in rs]
    if len(eta) >= 5:
        p.hyper.sigma_eta = math.sqrt(max(mean_var(eta)[1], 0.01))


async def save_params(conn: AsyncConnection, p: Params) -> None:
    for table in ("family_params", "sensor_noise", "annotator_confusion", "self_report_reliability",
                  "host_family_prior", "hyperparams"):
        await execute(conn, f"DELETE FROM {table}")
    for f, v in p.family.items():
        await execute(conn, "INSERT INTO family_params VALUES (%s, %s, %s, %s, %s, %s)",
                      f, v["b"], v["w"], v["kappa"], v["speed_mu"], v["speed_sd"])
    for sensor, sigma in p.noise.items():
        await execute(conn, "INSERT INTO sensor_noise VALUES (%s, 'all', %s)", sensor, sigma)
    rows = [(a, s, y, o, prob) for a, strata in p.confusions.items() for s, ys in strata.items()
            for y, obs in ys.items() for o, prob in obs.items()]
    async with conn.cursor() as cur:
        await cur.executemany("INSERT INTO annotator_confusion VALUES (%s, %s, %s, %s, %s)", rows)
    for r, row in p.reliability.items():
        for f, prob in row.items():
            await execute(conn, "INSERT INTO self_report_reliability VALUES (%s, %s, %s)", r, f, prob)
    for h, row in p.host_prior.items():
        for f, prob in row.items():
            await execute(conn, "INSERT INTO host_family_prior VALUES (%s, %s, %s)", h, f, prob)
    h = p.hyper
    values = {"mu0": h.mu0, "sigma_alpha": h.sigma_alpha, "sigma_gamma": h.sigma_gamma, "sigma_eta": h.sigma_eta,
              "tau": h.tau, "overhead": p.overhead, "reading_s_r": p.reading.s_r,
              "reading_mu_away": p.reading.mu_away, "reading_s_away": p.reading.s_away,
              **{f"gamma:{t}": g for t, g in h.gamma.items()}}
    for name, value in values.items():
        await execute(conn, "INSERT INTO hyperparams (name, value) VALUES (%s, %s)", name, value)


async def replay(conn: AsyncConnection, p: Params) -> int:
    """Rebuild every derived table from the events, in the order turns were closed online."""
    await execute(conn, f"TRUNCATE {', '.join(DERIVED)}")
    closes = await fetch(conn, """
        SELECT t.id, n.created_at AS closed_at FROM turns t
        JOIN LATERAL (SELECT created_at FROM turns n WHERE n.conversation_id = t.conversation_id
                      AND n.turn_index = t.turn_index + 1) n ON true
        ORDER BY n.created_at, t.id""")
    for row in closes:
        await engine.finalize_turn(conn, row["id"], row["closed_at"], p)
    return len(closes)


async def run(conn: AsyncConnection) -> int:
    p = await load_params(conn)
    turns = await finalized_turns(conn)
    conversations = await fetch(conn, """SELECT c.host, c.assistant_model, s.family_probs, s.kappa_mean
                                         FROM conversations c JOIN conversation_state s ON s.conversation_id = c.id""")
    thetas = {(r["user_id"], r["request_type"]): r["grid"] for r in await fetch(conn, "SELECT * FROM theta_posteriors")}
    alphas = {r["user_id"]: r["grid"] for r in await fetch(conn, "SELECT * FROM alpha_posteriors")}
    rho = {r["user_id"]: r["rho_mean"] for r in await fetch(conn, "SELECT user_id, rho_mean FROM user_reading")}
    refit_reactions(turns, p)
    refit_sensors(turns, p)
    refit_families(turns, conversations, p, {k: grid_mean_var(g)[0] for k, g in thetas.items()})
    refit_reading(turns, rho, p)
    refit_hyper(alphas, thetas, p)
    await save_params(conn, p)
    params_mod.invalidate()
    return await replay(conn, p)


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    await pool.open()
    await migrate()
    async with pool.connection() as conn, conn.transaction():
        await execute(conn, f"LOCK TABLE {', '.join(DERIVED)} IN EXCLUSIVE MODE")
        log.info("nightly: replayed %d turns", await run(conn))
    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
