"""GET /metrics (SPEC 11).

Treatment-versus-arm comparisons are labelled "measured"; every other figure
is "estimated".
"""

import collections
import math
import random
import statistics

from app import config
from app import db
from app import params as params_lib
from app import vocab
from app.model import stats

CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 0.75
RESAMPLES = 1000
ESTIMATED, MEASURED = "estimated", "measured"


def _words_per_step(turns: list[dict]) -> float | None:
    moved = sum(t["p_moved"] for t in turns)
    words = sum(t["words"] + t["tool_words"] for t in turns)
    return words / moved if moved > 0 else None


def _mean_of(key: str):
    def mean(turns: list[dict]) -> float | None:
        return sum(t[key] for t in turns) / len(turns) if turns else None

    return mean


STATISTICS = {
    "words_per_step_forward": _words_per_step,
    "repair_rate": _mean_of("p_repair"),
    "missing_info_rate": _mean_of("p_missing"),
    "tool_words_per_turn": _mean_of("tool_words"),
}


def _group(turns: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    out = collections.defaultdict(list)
    for t in turns:
        out[tuple(t[k] for k in keys)].append(t)
    return out


def _flat(clusters: list[list[dict]]) -> list[dict]:
    return [t for cluster in clusters for t in cluster]


def _summary(turns: list[dict], unit: str) -> dict:
    clusters = list(_group(turns, unit).values())
    out = {"turns": len(turns), "label": ESTIMATED}
    for name, statistic in STATISTICS.items():
        out[name] = statistic(turns)
        out[f"{name}_ci95"] = stats.bootstrap_ci(
            clusters, lambda cs, f=statistic: f(_flat(cs))
        )
    return out


def _saving(treatment: list[dict], arm: list[dict]) -> float | None:
    t, a = _words_per_step(treatment), _words_per_step(arm)
    return 1 - t / a if t and a else None


def _measured(turns: list[dict], arm: str, unit: str) -> dict | None:
    """Returns the words-per-step saving of treatment over an arm."""
    by_arm = {
        g: list(_group(ts, unit).values())
        for (g,), ts in _group(turns, "grp").items()
    }
    treated, other = by_arm.get(vocab.Arm.TREATMENT, []), by_arm.get(arm, [])
    if len(treated) < 2 or len(other) < 2:
        return None
    rng = random.Random(0)
    draws = sorted(
        v
        for v in (
            _saving(
                _flat(rng.choices(treated, k=len(treated))),
                _flat(rng.choices(other, k=len(other))),
            )
            for _ in range(RESAMPLES)
        )
        if v is not None
    )
    last = len(draws) - 1
    return {
        "label": MEASURED,
        "words_per_step_saving": _saving(_flat(treated), _flat(other)),
        "ci95": (draws[int(0.025 * last)], draws[int(0.975 * last)])
        if draws
        else None,
    }


async def compute(conn: db.Connection) -> dict:
    """Returns every metric of SPEC 11."""
    settings = config.settings
    rows = await db.fetch(
        conn,
        "SELECT t.id, t.user_id, t.conversation_id, c.grp, c.host, "
        "t.requested_words, length(t.raw_payload::text) AS payload_chars, "
        "to_char(date_trunc('week', t.created_at), 'YYYY-MM-DD') AS week, "
        "i.length_mean, i.reaction_probs, cs.family_probs "
        "FROM turns t JOIN conversations c ON c.id = t.conversation_id "
        "JOIN turn_inference i ON i.turn_id = t.id "
        "LEFT JOIN conversation_state cs "
        "ON cs.conversation_id = t.conversation_id",
    )
    turns = []
    for r in rows:
        probs, families = r["reaction_probs"], r["family_probs"] or {}
        turns.append(
            {
                **r,
                "words": math.exp(r["length_mean"]),
                "tool_words": r["payload_chars"]
                / CHARS_PER_TOKEN
                * WORDS_PER_TOKEN,
                "p_moved": probs[vocab.Outcome.MOVED_ON],
                "p_repair": 1 - probs[vocab.Outcome.MOVED_ON],
                "p_missing": probs[vocab.Outcome.MISSING_INFO],
                "family": max(families, key=families.get, default=None),
            }
        )
    unit = (
        "user_id"
        if settings.control_unit == vocab.RandomizationUnit.USER
        else "conversation_id"
    )
    out = {
        "control_unit": settings.control_unit,
        "by_group": {
            g: _summary(ts, unit) for (g,), ts in _group(turns, "grp").items()
        },
        "measured": {
            arm: _measured(turns, arm, unit)
            for arm in vocab.Arm
            if arm != vocab.Arm.TREATMENT
        },
        "weekly": [
            {"week": w, "group": g, **_summary(ts, unit)}
            for (w, g), ts in sorted(_group(turns, "week", "grp").items())
        ],
        "per_user": [
            {"user_id": u, "group": g, **_summary(ts, unit)}
            for (u, g), ts in sorted(_group(turns, "user_id", "grp").items())
        ],
        "compliance": [
            {
                "family": f,
                "host": h,
                "median_length_over_requested": statistics.median(
                    t["words"] / t["requested_words"] for t in ts
                ),
            }
            for (f, h), ts in sorted(
                _group(
                    [t for t in turns if t["requested_words"]], "family", "host"
                ).items()
            )
        ],
    }
    if settings.control_unit == vocab.RandomizationUnit.CONVERSATION:
        held_back = [t for t in turns if t["grp"] != vocab.Arm.TREATMENT]
        out["arms_over_time"] = [
            {"user_id": u, "week": w, **_summary(ts, unit)}
            for (u, w), ts in sorted(
                _group(held_back, "user_id", "week").items()
            )
        ]
    out["clicks"] = await db.fetch(
        conn,
        "SELECT s.button, s.position, count(*) AS shown, "
        "count(k.turn_id) AS clicked, "
        "count(k.turn_id)::float / count(*) AS click_rate "
        "FROM buttons_shown s LEFT JOIN clicks k "
        "ON k.turn_id = s.turn_id AND k.button = s.button "
        "GROUP BY s.button, s.position ORDER BY s.button, s.position",
    )
    p = await params_lib.load_params(conn)
    out["noise"] = {
        "sensor_sigma": p.noise,
        "confusion_diagonal": {
            f"{vocab.Source.ASSISTANT}/{s}": {
                y: row[y] for y, row in rows_.items()
            }
            for s, rows_ in p.confusions[vocab.Source.ASSISTANT].items()
        },
    }
    out["tool_coverage"] = await db.fetchrow(
        conn,
        "SELECT count(*) AS turns, "
        "avg((EXISTS (SELECT 1 FROM replies r WHERE r.turn_id = t.id))::int) "
        "FILTER (WHERE t.reply_mode = %s) AS widget_turns_with_show_reply, "
        "avg((t.reply_mode = %s)::int) AS widget_share, "
        "(SELECT avg((n > 1)::int) FROM (SELECT count(*) AS n FROM turns "
        "GROUP BY conversation_id) c) AS conversation_id_reused "
        "FROM turns t",
        vocab.ReplyMode.WIDGET,
        vocab.ReplyMode.WIDGET,
    )
    return out
