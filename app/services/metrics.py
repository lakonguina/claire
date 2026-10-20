"""GET /metrics (SPEC 11).

Treatment-versus-arm comparisons are labelled "measured"; every other figure
is "estimated".
"""

import datetime
import json
import math

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config
from app.core import vocab
from app.model import stats
from app import models
from app.services import params as params_lib

CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 0.75
RESAMPLES = 1000
ESTIMATED, MEASURED = "estimated", "measured"
# Every figure is a ratio of sums: name -> (numerator, denominator) columns.
RATIOS = {
    "words_per_step_forward": ("spent", "p_moved"),
    "repair_rate": ("p_repair", "one"),
    "missing_info_rate": ("p_missing", "one"),
    "tool_words_per_turn": ("tool_words", "one"),
}
SUMMED = sorted({c for pair in RATIOS.values() for c in pair})


def _summary(frame: pd.DataFrame, unit: str) -> dict:
    """Returns every ratio with its cluster-bootstrap 95 % interval."""
    sums = frame.groupby(unit)[SUMMED].sum()
    out = {"turns": len(frame), "label": ESTIMATED}
    for name, (num, den) in RATIOS.items():
        total = sums[den].sum()
        out[name] = float(sums[num].sum() / total) if total else None
        out[f"{name}_ci95"] = stats.cluster_bootstrap(
            sums[num].to_numpy(), sums[den].to_numpy()
        )
    return out


def _measured(frame: pd.DataFrame, arm: str, unit: str) -> dict | None:
    """Returns the words-per-step saving of treatment over an arm."""
    groups = [
        frame[frame.grp == g].groupby(unit)[["spent", "p_moved"]].sum()
        for g in (vocab.Arm.TREATMENT, arm)
    ]
    if min(len(g) for g in groups) < 2:
        return None
    rng = np.random.default_rng(0)

    def per_step(g: pd.DataFrame, picks: np.ndarray) -> np.ndarray:
        return g.spent.to_numpy()[picks].sum(-1) / g.p_moved.to_numpy()[
            picks
        ].sum(-1)

    draws = [
        per_step(g, rng.integers(len(g), size=(RESAMPLES, len(g))))
        for g in groups
    ]
    everything = [per_step(g, np.arange(len(g))) for g in groups]
    low, high = np.percentile(1 - draws[0] / draws[1], [2.5, 97.5])
    return {
        "label": MEASURED,
        "words_per_step_saving": float(1 - everything[0] / everything[1]),
        "ci95": (float(low), float(high)),
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
                "one": 1.0,
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


def _records(frame: pd.DataFrame, keys: list[str], unit: str) -> list[dict]:
    return [
        {**dict(zip(keys, key)), **_summary(group, unit)}
        for key, group in frame.groupby(keys)
    ]


async def compute(session: sa_asyncio.AsyncSession) -> dict:
    """Returns every metric of SPEC 11."""
    settings = config.settings
    frame = pd.DataFrame(await _turns(session))
    unit = (
        "user_id"
        if settings.control_unit == vocab.RandomizationUnit.USER
        else "conversation_id"
    )
    out = {"control_unit": settings.control_unit}
    if not frame.empty:
        frame["spent"] = frame.words + frame.tool_words
        held_back = frame[frame.grp != vocab.Arm.TREATMENT]
        compliance = (
            frame[frame.requested_words.notna()]
            .assign(ratio=lambda f: f.words / f.requested_words)
            .groupby(["family", "host"])
            .ratio.median()
        )
        out |= {
            "by_group": {
                g: _summary(group, unit) for g, group in frame.groupby("grp")
            },
            "measured": {
                arm: _measured(frame, arm, unit)
                for arm in vocab.Arm
                if arm != vocab.Arm.TREATMENT
            },
            "weekly": _records(frame, ["week", "grp"], unit),
            "per_user": _records(frame, ["user_id", "grp"], unit),
            "compliance": [
                {"family": f, "host": h, "median_length_over_requested": r}
                for (f, h), r in compliance.items()
            ],
        }
        if settings.control_unit == vocab.RandomizationUnit.CONVERSATION:
            out["arms_over_time"] = _records(
                held_back, ["user_id", "week"], unit
            )
    p = await params_lib.load_params(session)
    out |= {
        "clicks": await _clicks(session),
        "tool_coverage": await _coverage(session),
        "noise": {
            "sensor_sigma": p.noise,
            "confusion_diagonal": {
                f"{vocab.Source.ASSISTANT}/{s}": {
                    y: row[y] for y, row in ys.items()
                }
                for s, ys in p.confusions[vocab.Source.ASSISTANT].items()
            },
        },
    }
    return out
