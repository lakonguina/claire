"""GET /metrics (section 11). Treatment vs control comparisons are 'measured'; everything else is 'estimated'."""

import math
import random
import statistics
from collections import defaultdict

from app.config import settings
from app.db import AsyncConnection, fetch
from app.model.stats import bootstrap_ci
from app.params import load_params


def _wpsf(turns: list[dict]) -> float | None:
    moved = sum(t["p_moved"] for t in turns)
    return sum(t["words"] for t in turns) / moved if moved > 0 else None


def _rate(key: str):
    return lambda turns: sum(t[key] for t in turns) / len(turns) if turns else None


STATS = {"words_per_step_forward": _wpsf, "repair_rate": _rate("p_repair"), "missing_info_rate": _rate("p_missing")}


def _summary(turns: list[dict], cluster_key: str) -> dict:
    clusters = defaultdict(list)
    for t in turns:
        clusters[t[cluster_key]].append(t)
    groups = list(clusters.values())
    flat = lambda gs: [t for g in gs for t in g]
    out = {"turns": len(turns), "label": "estimated"}
    for name, fn in STATS.items():
        out[name] = fn(turns)
        out[name + "_ci95"] = bootstrap_ci(groups, lambda gs, fn=fn: fn(flat(gs)))
    return out


def _measured_saving(treatment: list[dict], control: list[dict], cluster_key: str) -> dict | None:
    """Relative reduction of words per step forward, control vs treatment, bootstrapped per group."""
    def clusters(turns):
        by = defaultdict(list)
        for t in turns:
            by[t[cluster_key]].append(t)
        return list(by.values())

    tc, cc = clusters(treatment), clusters(control)
    if len(tc) < 2 or len(cc) < 2:
        return None
    saving = lambda t, c: (1 - _wpsf(t) / _wpsf(c)) if _wpsf(t) and _wpsf(c) else None
    rng = random.Random(0)
    draws = sorted(v for v in (
        saving([x for g in rng.choices(tc, k=len(tc)) for x in g], [x for g in rng.choices(cc, k=len(cc)) for x in g])
        for _ in range(1000)) if v is not None)
    ci = (draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]) if draws else None
    return {"label": "measured", "words_per_step_saving": saving(treatment, control), "ci95": ci}


async def compute(conn: AsyncConnection) -> dict:
    rows = await fetch(conn, """
        SELECT t.id, t.user_id, t.conversation_id, c.grp, c.host, t.requested_words, t.reply_mode,
               to_char(date_trunc('week', t.created_at), 'YYYY-MM-DD') AS week,
               i.length_mean, i.reaction_probs, cs.family_probs
        FROM turns t JOIN conversations c ON c.id = t.conversation_id
        JOIN turn_inference i ON i.turn_id = t.id
        LEFT JOIN conversation_state cs ON cs.conversation_id = t.conversation_id""")
    turns = [{
        **r, "words": math.exp(r["length_mean"]), "p_moved": r["reaction_probs"]["moved_on"],
        "p_repair": 1 - r["reaction_probs"]["moved_on"], "p_missing": r["reaction_probs"]["missing_info"],
        "family": max(r["family_probs"], key=r["family_probs"].get) if r["family_probs"] else "other",
    } for r in rows]
    cluster_key = "user_id" if settings.control_unit == "user" else "conversation_id"
    by = lambda *keys: _group(turns, keys)

    out = {
        "control_unit": settings.control_unit,
        "by_group": {g: _summary(ts, cluster_key) for (g,), ts in by("grp").items()},
        "measured": _measured_saving([t for t in turns if t["grp"] == "treatment"],
                                     [t for t in turns if t["grp"] == "control"], cluster_key),
        "weekly": [{"week": w, "group": g, **_summary(ts, cluster_key)} for (w, g), ts in sorted(by("week", "grp").items())],
        "per_user": [{"user_id": u, "group": g, **_summary(ts, cluster_key)} for (u, g), ts in sorted(by("user_id", "grp").items())],
        "compliance": [
            {"family": f, "host": h, "median_length_over_requested": statistics.median(
                t["words"] / t["requested_words"] for t in ts)}
            for (f, h), ts in sorted(_group([t for t in turns if t["requested_words"]], ("family", "host")).items())
        ],
    }
    if settings.control_unit == "conversation":
        out["control_over_time"] = [
            {"user_id": u, "week": w, **_summary(ts, cluster_key)}
            for (u, w), ts in sorted(_group([t for t in turns if t["grp"] == "control"], ("user_id", "week")).items())
        ]
    out["clicks"] = await fetch(conn, """
        SELECT s.button, s.position, count(*) AS shown, count(k.turn_id) AS clicked,
               count(k.turn_id)::float / count(*) AS click_rate
        FROM buttons_shown s LEFT JOIN clicks k ON k.turn_id = s.turn_id AND k.button = s.button
        GROUP BY s.button, s.position ORDER BY s.button, s.position""")
    params = await load_params(conn)
    out["noise"] = {
        "sensor_sigma": params.noise,
        "confusion_diagonal": {
            f"{a}/{s}": {y: row.get(y) for y, row in rows.items() if y in row}
            for a, strata in params.confusions.items() if a == "assistant" for s, rows in strata.items()
        },
    }
    coverage = (await fetch(conn, """
        SELECT count(*) AS turns,
               avg((EXISTS (SELECT 1 FROM replies r WHERE r.turn_id = t.id))::int)
                   FILTER (WHERE t.reply_mode = 'widget') AS widget_turns_with_show_reply,
               avg((t.reply_mode = 'widget')::int) AS widget_share
        FROM turns t"""))[0]
    reuse = (await fetch(conn, """
        SELECT avg((n > 1)::int) AS conversation_id_reused FROM (
            SELECT count(*) AS n FROM turns GROUP BY conversation_id) c"""))[0]
    out["tool_coverage"] = {**coverage, **reuse}
    return out


def _group(turns: list[dict], keys: tuple) -> dict:
    out = defaultdict(list)
    for t in turns:
        out[tuple(t[k] for k in keys)].append(t)
    return out
