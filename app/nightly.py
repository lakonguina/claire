"""Nightly job (SPEC 6): refit every noise model, hyperparameter and topic
cluster, then rebuild the derived state by replaying the events."""

import asyncio
import collections
import logging
import math

from app import db
from app import engine
from app import params as params_lib
from app import topics
from app import vocab
from app.model import family
from app.model import length
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta

log = logging.getLogger(__name__)
DERIVED = [
    "turn_inference",
    "conversation_state",
    "theta_posteriors",
    "alpha_posteriors",
    "user_reading",
    "pref_counts",
]
MIN_WEIGHT = 20.0  # Evidence needed before a parameter leaves its prior.
MIN_ITEMS = 50  # Turns needed before Dawid-Skene runs.
MIN_USERS = 5  # Users needed before the hyperparameters are refit.
MIN_CELLS = 3  # Cells of a request type needed to estimate its gamma.
MIN_VAR = 0.01
HOST_SMOOTHING = 0.5


async def finalized_turns(conn: db.Connection) -> list[dict]:
    """Returns every closed turn with its first events and inferred state."""
    return await db.fetch(
        conn,
        """SELECT t.*, c.host, i.length_mean, i.reading_probs, s.family_probs,
               s.kappa_mean, s.speed_mean,
               (SELECT extract(epoch FROM r.created_at - t.created_at)::float8
                FROM replies r WHERE r.turn_id = t.id
                ORDER BY r.created_at LIMIT 1) AS duration,
               (SELECT r.reply_words_exact FROM replies r
                WHERE r.turn_id = t.id ORDER BY r.created_at LIMIT 1) AS exact,
               o.observed AS verdict, o.last_reply_words AS self_report,
               (SELECT row_to_json(k) FROM clicks k WHERE k.turn_id = t.id
                ORDER BY k.created_at LIMIT 1) AS click,
               EXISTS (SELECT 1 FROM buttons_shown b WHERE b.turn_id = t.id)
                   AS any_shown,
               (SELECT outcome FROM gold_labels g WHERE g.turn_id = t.id
                ORDER BY g.created_at DESC LIMIT 1) AS gold
           FROM turns t JOIN conversations c ON c.id = t.conversation_id
           JOIN turn_inference i ON i.turn_id = t.id
           JOIN conversation_state s ON s.conversation_id = t.conversation_id
           LEFT JOIN LATERAL (SELECT * FROM outcomes WHERE turn_id = t.id
               AND source = %s ORDER BY created_at LIMIT 1) o ON true""",
        vocab.Source.ASSISTANT,
    )


def refit_reactions(turns: list[dict], p: params_lib.Params) -> None:
    """Refits every annotator's confusion matrix by Dawid-Skene."""
    items = [
        reaction.Item(
            engine.observations(
                t["verdict"],
                t["family_probs"],
                t["click"],
                t["any_shown"],
                [t["reading_probs"][c] for c in reading.COMPONENTS],
            ),
            t["gold"],
        )
        for t in turns
    ]
    if len(items) >= MIN_ITEMS:
        p.confusions, _, _ = reaction.dawid_skene(items, p.confusions)


def _sd(values: list[float]) -> float:
    return math.sqrt(max(stats.mean_var(values)[1], MIN_VAR))


def refit_sensors(turns: list[dict], p: params_lib.Params) -> None:
    """Refits the length sensors' noise and the timing overhead (SPEC 6.2)."""
    writing, requested, self_reports, fallback_requested, overheads = (
        [],
        [],
        [],
        [],
        [],
    )
    for t in turns:
        w = (t["duration"] or 0) - p.overhead
        s2 = math.log(w) + t["speed_mean"] if w > 0 else None
        s3 = (
            math.log(t["requested_words"]) + t["kappa_mean"]
            if t["requested_words"]
            else None
        )
        if t["exact"]:
            truth = math.log(t["exact"])
            if s2 is not None:
                writing.append(s2 - truth)
            if s3 is not None:
                requested.append(s3 - truth)
            if t["duration"]:
                overheads.append(
                    t["duration"] - math.exp(truth - t["speed_mean"])
                )
        elif t["self_report"] and s3 is not None:
            self_reports.append(math.log(t["self_report"]))
            fallback_requested.append(s3)
    if len(writing) >= MIN_WEIGHT:
        p.noise["writing_time"] = _sd(writing)
    if len(requested) >= MIN_WEIGHT:
        p.noise["requested"] = _sd(requested)
    if len(self_reports) >= MIN_WEIGHT:
        p.noise["self_report"] = math.sqrt(
            length.self_report_variance(
                self_reports, fallback_requested, p.noise["requested"] ** 2
            )
        )
    if len(overheads) >= MIN_WEIGHT:
        p.overhead = max(sorted(overheads)[len(overheads) // 2], 0.5)


def _cell(turn: dict) -> tuple[int, str]:
    return turn["user_id"], engine.style_of(turn)["cell"]


def refit_families(
    turns: list[dict],
    conversations: list[dict],
    p: params_lib.Params,
    theta_means: dict[tuple[int, str], float],
) -> None:
    """Refits each family's speed, bias and compliance, and the priors."""
    for f in family.FAMILIES:
        speeds, speed_w, biases, bias_w = [], [], [], []
        for t in turns:
            weight = t["family_probs"][f]
            w = (t["duration"] or 0) - p.overhead
            if w > 0:
                speeds.append(t["length_mean"] - math.log(w))
                speed_w.append(weight)
            if _cell(t) in theta_means:
                biases.append(
                    t["length_mean"]
                    - math.log(t["needed_words"])
                    - theta_means[_cell(t)]
                )
                bias_w.append(weight)
        fam = p.family[f]
        if sum(speed_w) >= MIN_WEIGHT:
            mu, var = stats.mean_var(speeds, speed_w)
            fam.update(speed_mu=mu, speed_sd=math.sqrt(max(var, MIN_VAR)))
        if sum(bias_w) >= MIN_WEIGHT:
            b, var = stats.mean_var(biases, bias_w)
            fam.update(b=b, w=max(var, MIN_VAR))
        weights = [c["family_probs"][f] for c in conversations]
        if sum(weights) >= MIN_WEIGHT:
            fam["kappa"] = stats.mean_var(
                [c["kappa_mean"] for c in conversations], weights
            )[0]
    # mu0 holds the population level, so the family biases are centred.
    center = sum(p.family[f]["b"] for f in family.FAMILIES) / len(
        family.FAMILIES
    )
    for f in family.FAMILIES:
        p.family[f]["b"] -= center

    by_host, by_report = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    for c in conversations:
        by_host[c["host"]].append(c["family_probs"])
        reported = family.reported_family(c["assistant_model"])
        if reported is not None:
            by_report[reported].append(c["family_probs"])

    def average(probs: list[dict]) -> dict[str, float]:
        sums = [
            sum(x[f] for x in probs) + HOST_SMOOTHING for f in family.FAMILIES
        ]
        return dict(zip(family.FAMILIES, stats.normalize(sums)))

    p.host_prior = {
        h: average(v) for h, v in by_host.items() if len(v) >= MIN_WEIGHT
    }
    for r, v in by_report.items():
        if len(v) >= MIN_WEIGHT:
            p.reliability[r] = average(v)


def refit_reading(
    turns: list[dict], rho: dict[int, float], p: params_lib.Params
) -> None:
    """Refits the population reading mixture (SPEC 6.6)."""
    data = [
        (
            math.log(t["reading_probs"]["seconds"]),
            t["length_mean"] - rho.get(t["user_id"], 0.0),
        )
        for t in turns
    ]
    if len(data) >= MIN_WEIGHT:
        p.reading, _ = reading.fit_mixture(data, p.reading)


def refit_hyper(
    alphas: dict[int, list[float]],
    thetas: dict[tuple[int, str], list[float]],
    p: params_lib.Params,
) -> None:
    """Refits mu0, sigma_alpha, gamma, sigma_gamma and sigma_eta (SPEC 6.5)."""
    if len(alphas) < MIN_USERS:
        return
    a_stats = {u: stats.grid_mean_var(g) for u, g in alphas.items()}
    hyper = p.hyper
    hyper.mu0, spread = stats.mean_var([m for m, _ in a_stats.values()])
    mean_var = sum(v for _, v in a_stats.values()) / len(a_stats)
    hyper.sigma_alpha = math.sqrt(spread + mean_var)
    residuals = collections.defaultdict(list)
    for (u, cell), grid in thetas.items():
        if u in a_stats:
            residuals[theta.request_type_of(cell)].append(
                stats.grid_mean_var(grid)[0] - a_stats[u][0]
            )
    hyper.gamma = {
        t: stats.mean_var(r)[0]
        for t, r in residuals.items()
        if len(r) >= MIN_CELLS
    }
    if len(hyper.gamma) >= 2:
        hyper.sigma_gamma = _sd(list(hyper.gamma.values()))
    eta = [
        r - hyper.gamma.get(t, 0.0) for t, rs in residuals.items() for r in rs
    ]
    if len(eta) >= MIN_USERS:
        hyper.sigma_eta = _sd(eta)


async def save_params(conn: db.Connection, p: params_lib.Params) -> None:
    """Replaces every learned parameter table."""
    rows = {
        "family_params": [
            (f, v["b"], v["w"], v["kappa"], v["speed_mu"], v["speed_sd"])
            for f, v in p.family.items()
        ],
        "sensor_noise": [
            (s, params_lib.ALL_FAMILIES, sigma) for s, sigma in p.noise.items()
        ],
        "annotator_confusion": [
            (a, s, y, o, prob)
            for a, strata in p.confusions.items()
            for s, ys in strata.items()
            for y, obs in ys.items()
            for o, prob in obs.items()
        ],
        "self_report_reliability": [
            (r, f, prob)
            for r, row in p.reliability.items()
            for f, prob in row.items()
        ],
        "host_family_prior": [
            (h, f, prob)
            for h, row in p.host_prior.items()
            for f, prob in row.items()
        ],
    }
    hyper = {name: getattr(p.hyper, name) for name in params_lib.HYPER_NAMES}
    hyper |= {
        n: getattr(p.reading, f) for n, f in params_lib.READING_NAMES.items()
    }
    hyper |= {
        f"{params_lib.GAMMA_PREFIX}{t}": g for t, g in p.hyper.gamma.items()
    }
    hyper["overhead"] = p.overhead
    rows["hyperparams"] = list(hyper.items())
    async with conn.cursor() as cursor:
        for table, values in rows.items():
            await cursor.execute(f"DELETE FROM {table}")
            if values:
                # Trailing columns (updated_at) take their defaults.
                columns = ", ".join(["%s"] * len(values[0]))
                await cursor.executemany(
                    f"INSERT INTO {table} VALUES ({columns})", values
                )


async def replay(conn: db.Connection, p: params_lib.Params) -> int:
    """Rebuilds every derived table from the events, in their live order.

    Args:
        conn: The connection, inside the nightly transaction.
        p: The parameters to replay with.

    Returns:
        The number of turns closed.
    """
    await db.execute(conn, f"TRUNCATE {", ".join(DERIVED)}")
    closes = await db.fetch(
        conn,
        """SELECT t.id, n.created_at AS at FROM turns t
           JOIN LATERAL (SELECT created_at FROM turns n
               WHERE n.conversation_id = t.conversation_id
               AND n.turn_index = t.turn_index + 1) n ON true""",
    )
    edits = await db.fetch(
        conn, "SELECT *, created_at AS at FROM profile_edits"
    )
    events = sorted(
        [(c["at"], 0, c) for c in closes] + [(e["at"], 1, e) for e in edits],
        key=lambda e: (e[0], e[1], e[2].get("id", 0)),
    )
    for at, is_edit, event in events:
        if is_edit:
            await engine.apply_edit(conn, event, p)
        else:
            await engine.finalize_turn(conn, event["id"], at, p)
    return len(closes)


async def run(conn: db.Connection) -> int:
    """Refits everything, then replays; returns the number of turns closed."""
    p = await params_lib.load_params(conn)
    turns = await finalized_turns(conn)
    conversations = await db.fetch(
        conn,
        "SELECT c.host, c.assistant_model, s.family_probs, s.kappa_mean "
        "FROM conversations c "
        "JOIN conversation_state s ON s.conversation_id = c.id",
    )
    thetas = {
        (r["user_id"], r["request_type"]): r["grid"]
        for r in await db.fetch(conn, "SELECT * FROM theta_posteriors")
    }
    alphas = {
        r["user_id"]: r["grid"]
        for r in await db.fetch(conn, "SELECT * FROM alpha_posteriors")
    }
    rho = {
        r["user_id"]: r["rho_mean"]
        for r in await db.fetch(
            conn, "SELECT user_id, rho_mean FROM user_reading"
        )
    }
    refit_reactions(turns, p)
    refit_sensors(turns, p)
    refit_families(
        turns,
        conversations,
        p,
        {k: stats.grid_mean_var(g)[0] for k, g in thetas.items()},
    )
    refit_reading(turns, rho, p)
    refit_hyper(alphas, thetas, p)
    await save_params(conn, p)
    await topics.recluster(conn)
    params_lib.cache.invalidate()
    return await replay(conn, p)


async def main() -> None:
    """Runs the nightly job in one transaction, locking the derived state."""
    logging.basicConfig(level=logging.INFO)
    await db.migrate()
    await db.pool.open()
    async with db.pool.connection() as conn, conn.transaction():
        await db.execute(
            conn, f"LOCK TABLE {", ".join(DERIVED)} IN EXCLUSIVE MODE"
        )
        log.info("nightly: replayed %d turns", await run(conn))
    await db.pool.close()


if __name__ == "__main__":
    asyncio.run(main())
