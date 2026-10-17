"""GET /metrics (SPEC 11).

Treatment-versus-arm comparisons are labelled "measured"; every other figure
is "estimated".
"""

import collections
import datetime
import json
import math
import random
import statistics

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import models
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


def _week(at: datetime.datetime) -> str:
    return (at - datetime.timedelta(days=at.weekday())).date().isoformat()


async def _turns(session: sa_asyncio.AsyncSession) -> list[dict]:
    """Returns every closed turn with what the metrics need."""
    rows = (
        await session.execute(
            sa.select(
                models.Turn,
                models.Conversation,
                models.TurnInference,
                models.ConversationState,
            )
            .join(
                models.Conversation,
                models.Conversation.id == models.Turn.conversation_id,
            )
            .join(
                models.TurnInference,
                models.TurnInference.turn_id == models.Turn.id,
            )
            .outerjoin(
                models.ConversationState,
                models.ConversationState.conversation_id
                == models.Turn.conversation_id,
            )
        )
    ).all()
    out = []
    for turn, conv, inference, state in rows:
        probs = inference.reaction_probs
        families = state.family_probs if state else {}
        payload_chars = len(json.dumps(turn.raw_payload, ensure_ascii=False))
        out.append(
            {
                "user_id": turn.user_id,
                "conversation_id": turn.conversation_id,
                "grp": conv.grp,
                "host": conv.host,
                "week": _week(turn.created_at),
                "requested_words": turn.requested_words,
                "words": math.exp(inference.length_mean),
                "tool_words": payload_chars / CHARS_PER_TOKEN * WORDS_PER_TOKEN,
                "p_moved": probs[vocab.Outcome.MOVED_ON],
                "p_repair": 1 - probs[vocab.Outcome.MOVED_ON],
                "p_missing": probs[vocab.Outcome.MISSING_INFO],
                "family": max(families, key=families.get, default=None),
            }
        )
    return out


async def _clicks(session: sa_asyncio.AsyncSession) -> list[dict]:
    shown, click = models.ButtonShown, models.Click
    rows = await session.execute(
        sa.select(
            shown.button,
            shown.position,
            sa.func.count().label("shown"),
            sa.func.count(click.id).label("clicked"),
        )
        .outerjoin(
            click,
            sa.and_(
                click.turn_id == shown.turn_id, click.button == shown.button
            ),
        )
        .group_by(shown.button, shown.position)
        .order_by(shown.button, shown.position)
    )
    return [
        {**r, "click_rate": r["clicked"] / r["shown"]} for r in rows.mappings()
    ]


async def _coverage(session: sa_asyncio.AsyncSession) -> dict:
    t = models.Turn
    modes = dict(
        (
            await session.execute(
                sa.select(t.reply_mode, sa.func.count()).group_by(t.reply_mode)
            )
        ).all()
    )
    widget_ids = sa.select(t.id).where(t.reply_mode == vocab.ReplyMode.WIDGET)
    with_reply = await session.scalar(
        sa.select(sa.func.count(sa.distinct(models.Reply.turn_id))).where(
            models.Reply.turn_id.in_(widget_ids)
        )
    )
    per_conversation = list(
        await session.scalars(
            sa.select(sa.func.count())
            .select_from(t)
            .group_by(t.conversation_id)
        )
    )
    turns = sum(modes.values())
    widget = modes.get(vocab.ReplyMode.WIDGET, 0)
    return {
        "turns": turns,
        "widget_share": widget / turns if turns else None,
        "widget_turns_with_show_reply": with_reply / widget if widget else None,
        "conversation_id_reused": (
            sum(n > 1 for n in per_conversation) / len(per_conversation)
            if per_conversation
            else None
        ),
    }


async def compute(session: sa_asyncio.AsyncSession) -> dict:
    """Returns every metric of SPEC 11."""
    settings = config.settings
    turns = await _turns(session)
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
                ).items(),
                key=lambda item: tuple(map(str, item[0])),
            )
        ],
        "clicks": await _clicks(session),
        "tool_coverage": await _coverage(session),
    }
    if settings.control_unit == vocab.RandomizationUnit.CONVERSATION:
        held_back = [t for t in turns if t["grp"] != vocab.Arm.TREATMENT]
        out["arms_over_time"] = [
            {"user_id": u, "week": w, **_summary(ts, unit)}
            for (u, w), ts in sorted(
                _group(held_back, "user_id", "week").items()
            )
        ]
    p = await params_lib.load_params(session)
    out["noise"] = {
        "sensor_sigma": p.noise,
        "confusion_diagonal": {
            f"{vocab.Source.ASSISTANT}/{s}": {
                y: row[y] for y, row in ys.items()
            }
            for s, ys in p.confusions[vocab.Source.ASSISTANT].items()
        },
    }
    return out
