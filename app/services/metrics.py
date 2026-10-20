"""GET /metrics: the three KPIs, their diagnostics, and the funnel.

Comparisons of randomised groups (holdout, ablated field, candidate catalog)
are labelled "measured"; every other figure is "estimated".
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
from app.services import replies

CHARS_PER_TOKEN = 4.0
WORDS_PER_TOKEN = 0.75
RESAMPLES = 1000
ESTIMATED, MEASURED = "estimated", "measured"
CALIBRATION_BINS = 5
CHURN = datetime.timedelta(days=30)
REFRESH_LIFE = datetime.timedelta(days=90)
# Every figure is a ratio of sums: name -> (numerator, denominator) columns.
RATIOS = {
    "words_per_goal_reached": ("spent", "goals"),
    "repair_rate": ("p_repair", "one"),
    "questions_per_goal": ("asked", "goals"),
    "words_per_step_forward": ("spent", "p_moved"),
    "missing_info_rate": ("p_missing", "one"),
    "wrong_fact_rate": ("p_wrong_fact", "one"),
    "rephrase_rate": ("rephrased", "known_distance"),
    "tool_words_per_turn": ("tool_words", "one"),
    "turns_over_token_budget": ("over_budget", "one"),
}
SUMMED = sorted({c for pair in RATIOS.values() for c in pair})
# Whether the assistant sent what it should, per turn: field -> (sent, due).
FIELDS = {
    "goal_on_first_turn": ("has_goal", "first"),
    "user_message_when_asked": ("has_message", "message_asked"),
    "choices_when_asked": ("has_choices", "choices_asked"),
    "last_turn_id": ("has_last_turn", "later"),
    "messages_since_last_call": ("has_count", "later"),
}
Session = sa_asyncio.AsyncSession


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


def _measured(
    treated: pd.DataFrame, other: pd.DataFrame, unit: str
) -> dict | None:
    """Returns the words-per-goal saving of one random group over another."""
    groups = [
        g.groupby(unit)[["spent", "goals"]].sum() for g in (treated, other)
    ]
    if (
        min(len(g) for g in groups) < 2
        or min(g.goals.sum() for g in groups) == 0
    ):
        return None
    rng = np.random.default_rng(0)

    def per_goal(g: pd.DataFrame, picks: np.ndarray) -> np.ndarray:
        goals = g.goals.to_numpy()[picks].sum(-1)
        spent = g.spent.to_numpy()[picks].sum(-1)
        return spent / np.where(goals, goals, np.nan)

    draws = [
        per_goal(g, rng.integers(len(g), size=(RESAMPLES, len(g))))
        for g in groups
    ]
    everything = [per_goal(g, np.arange(len(g))) for g in groups]
    low, high = np.nanpercentile(1 - draws[0] / draws[1], [2.5, 97.5])
    return {
        "label": MEASURED,
        "words_per_goal_saving": float(1 - everything[0] / everything[1]),
        "ci95": (float(low), float(high)),
    }


def _week(at: datetime.datetime) -> str:
    return (at - datetime.timedelta(days=at.weekday())).date().isoformat()


async def _ids(session: Session, statement) -> set[int]:
    return set(await session.scalars(statement))


def _asked(style: dict, what: vocab.Requested) -> bool:
    return bool(style.get("asks", {}).get(what))


async def turns(session: Session, p: params_lib.Params) -> list[dict]:
    """Returns every turn with what the metrics need; inferred when closed."""
    reached = await _ids(
        session,
        sa.select(models.Outcome.turn_id).where(
            models.Outcome.source == vocab.Source.ASSISTANT,
            models.Outcome.goal_reached.is_(True),
        ),
    )
    goals = await _ids(session, sa.select(models.Goal.turn_id))
    messages = await _ids(session, sa.select(models.Message.turn_id))
    with_choices = await _ids(
        session,
        sa.select(models.Reply.turn_id).where(models.Reply.choices > 0),
    )
    # What an instruction asks is due in the call that follows it.
    following = dict(
        (
            await session.execute(
                sa.select(models.Turn.follows, sa.func.min(models.Turn.id))
                .where(models.Turn.follows.is_not(None))
                .group_by(models.Turn.follows)
            )
        ).all()
    )
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
            .outerjoin(
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
    budget = config.settings.token_budget
    out = []
    for turn, conv, inference, state in rows:
        families = state.family_probs if state else {}
        style = turn.style or {}
        nxt = following.get(turn.id)
        later = turn.turn_index > 1
        tokens = (
            len(json.dumps(turn.raw_payload, ensure_ascii=False))
            / CHARS_PER_TOKEN
        )
        message = _asked(style, vocab.Requested.USER_MESSAGE) and nxt
        choices = _asked(style, vocab.Requested.CHOICES)
        row = {
            "closed": inference is not None,
            "user_id": turn.user_id,
            "conversation_id": turn.conversation_id,
            "grp": conv.grp,
            "catalog": conv.catalog,
            "ablated": conv.ablated_field,
            "host": conv.host,
            "week": _week(turn.created_at),
            "requested_words": turn.requested_words,
            "tool_words": tokens * WORDS_PER_TOKEN,
            "over_budget": float(tokens > budget),
            # Questions asked: clarifying ones and the ground-truth one.
            "asked": float(style.get("clarify") == vocab.Clarify.ASK)
            + float(bool(style.get("survey"))),
            "format": (style.get("form") or {}).get(vocab.Dimension.FORMAT),
            "mobile": bool(style.get("mobile")),
            "expertise": style.get("expertise"),
            "target_words": turn.target_words,
            "last_call": turn.created_at,
            "ambiguity": turn.ambiguity,
            "ambiguity_calibrated": style.get("ambiguity_calibrated"),
            "goals": float(turn.id in reached),
            "family": max(families, key=families.get, default=None),
            "one": 1.0,
            "first": float(not later),
            "later": float(later),
            "has_goal": float(not later and turn.id in goals),
            "has_last_turn": float(
                later and "last_turn_id" in turn.raw_payload
            ),
            "has_count": float(
                later and turn.messages_since_last_call is not None
            ),
            "message_asked": float(bool(message)),
            "has_message": float(bool(message) and nxt in messages),
            "choices_asked": float(choices),
            "has_choices": float(choices and turn.id in with_choices),
        }
        if inference is not None:
            probs = inference.reaction_probs
            distance = inference.next_topic_distance
            row |= {
                "words": math.exp(inference.length_mean),
                "p_moved": probs[vocab.Outcome.MOVED_ON],
                "p_repair": 1 - probs[vocab.Outcome.MOVED_ON],
                "p_missing": probs[vocab.Outcome.MISSING_INFO],
                "p_wrong_fact": probs[vocab.Outcome.WRONG_FACT],
                "p_misunderstood": probs[vocab.Outcome.MISUNDERSTOOD],
                "p_wrong_format": probs[vocab.Outcome.WRONG_FORMAT],
                "known_distance": float(distance is not None),
                "rephrased": float(
                    distance is not None and distance < p.rephrase_distance
                ),
            }
        out.append(row)
    return out


async def _count(session: Session, statement) -> int:
    return await session.scalar(
        sa.select(sa.func.count()).select_from(statement.subquery())
    )


async def _coverage(session: Session) -> dict:
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
    unanswered = list(
        await session.scalars(
            sa.select(t.messages_since_last_call).where(
                t.messages_since_last_call.is_not(None)
            )
        )
    )
    total = sum(modes.values())
    widget = modes.get(vocab.ReplyMode.WIDGET, 0)
    return {
        "turns": total,
        "widget_share": widget / total if total else None,
        "widget_turns_with_show_reply": with_reply / widget if widget else None,
        "conversation_id_reused": (
            sum(n > 1 for n in per_conversation) / len(per_conversation)
            if per_conversation
            else None
        ),
        # Replies the assistant gave without calling plan_reply first.
        "replies_without_call": (
            sum(unanswered) / (sum(unanswered) + len(unanswered))
            if unanswered
            else None
        ),
    }


def _share(part: float, whole: float) -> float | None:
    return float(part / whole) if whole else None


def field_rates(frame: pd.DataFrame) -> dict[str, float | None]:
    """Returns how often the assistant sends each field when it should."""
    return {
        name: _share(frame[sent].sum(), frame[due].sum())
        for name, (sent, due) in FIELDS.items()
    }


def _calibration(closed: pd.DataFrame) -> dict:
    """Returns the ambiguity calibration curve, and asking vs answering."""
    sent = closed[closed.ambiguity.notna()]
    bins = np.linspace(0, 1, CALIBRATION_BINS + 1)
    curve = []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = sent[(sent.ambiguity >= low) & (sent.ambiguity <= high)]
        if len(inside):
            curve.append(
                {
                    "ambiguity": (float(low), float(high)),
                    "turns": len(inside),
                    "calibrated": float(inside.ambiguity_calibrated.mean()),
                    "misunderstood_or_rephrased": float(
                        np.maximum(
                            inside.p_misunderstood, inside.rephrased
                        ).mean()
                    ),
                }
            )
    ambiguous = closed[closed.ambiguity_calibrated.notna()]
    asking = ambiguous.asked == 1
    return {
        "curve": curve,
        "repair_rate_after_asking": _share(
            ambiguous[asking].p_repair.sum(), asking.sum()
        ),
        "repair_rate_after_answering": _share(
            ambiguous[~asking].p_repair.sum(), (~asking).sum()
        ),
    }


async def _widget(session: Session) -> dict:
    """Returns what the widget saw: copies, time on screen, answers."""
    e = models.WidgetEvent
    by_kind = dict(
        (
            await session.execute(
                sa.select(
                    e.kind, sa.func.count(sa.distinct(e.turn_id))
                ).group_by(e.kind)
            )
        ).all()
    )
    shown = await _count(session, sa.select(models.Reply.turn_id).distinct())
    # The widget reports a running total: a turn's time is its largest.
    visible = list(
        await session.scalars(
            sa.select(sa.func.max(e.value["value"].as_float()))
            .where(e.kind == vocab.WidgetEvent.VISIBLE)
            .group_by(e.turn_id)
        )
    )
    answer = e.value["value"].as_string().label("answer")
    answers = dict(
        (
            await session.execute(
                sa.select(answer, sa.func.count())
                .where(e.kind == vocab.WidgetEvent.SATISFACTION)
                .group_by(answer)
            )
        ).all()
    )
    return {
        "shown_replies": shown,
        "copy_rate": _share(by_kind.get(vocab.WidgetEvent.COPY, 0), shown),
        "median_visible_seconds": (
            float(np.median(visible)) if visible else None
        ),
        "choices_answered": by_kind.get(vocab.WidgetEvent.CHOICE, 0),
        "satisfaction": answers,
        "privacy_link_opened": by_kind.get(vocab.WidgetEvent.PRIVACY_LINK, 0),
        "render_errors": by_kind.get(vocab.WidgetEvent.RENDER_ERROR, 0),
    }


async def _funnel(session: Session, now: datetime.datetime) -> dict:
    """Returns the acquisition funnel, from page views to active people."""
    v, t, c = models.Visit, models.Turn, models.OAuthCredential
    days = sa.func.count(sa.distinct(sa.func.date(t.created_at)))
    visits = (
        await session.execute(
            sa.select(
                v.page,
                v.utm["utm_source"].as_string().label("source"),
                sa.func.count().label("visits"),
            )
            .group_by("page", "source")
            .order_by("page", "source")
        )
    ).mappings()
    # A connector whose app stopped refreshing its token was removed.
    last_refresh = (
        sa.select(c.grant_id, sa.func.max(c.expires_at).label("until"))
        .where(c.kind == vocab.Credential.REFRESH)
        .group_by(c.grant_id)
        .subquery()
    )
    churned = sa.select(last_refresh.c.grant_id).where(
        last_refresh.c.until - REFRESH_LIFE < now - CHURN
    )
    # The account instruction is pasted when every call says none was missed.
    following = (
        sa.select(t.user_id)
        .group_by(t.user_id)
        .having(
            sa.func.count() >= 3,
            sa.func.max(sa.func.coalesce(t.messages_since_last_call, 1)) == 0,
        )
    )
    return {
        "visits": [dict(r) for r in visits],
        "apps_registered": await _count(
            session, sa.select(models.OAuthClient.client_id)
        ),
        "consents": await _count(session, sa.select(models.OAuthGrant.id)),
        "people_with_a_call": await _count(
            session, sa.select(t.user_id).distinct()
        ),
        "people_following_the_instruction": await _count(session, following),
        "people_active_two_days": await _count(
            session,
            sa.select(t.user_id).group_by(t.user_id).having(days >= 2),
        ),
        "connectors_churned_30d": await _count(session, churned),
    }


def _records(frame: pd.DataFrame, keys: list[str], unit: str) -> list[dict]:
    return [
        {**dict(zip(keys, key)), **_summary(group, unit)}
        for key, group in frame.groupby(keys)
    ]


def _axes(treated: pd.DataFrame) -> dict:
    """Returns the diagnostics of length, form and expertise."""
    sized = treated[treated.target_words.notna()]
    length = [
        {
            "family": family,
            "median_target": float(group.target_words.median()),
            "median_written": float(group.words.median()),
            "median_written_over_target": float(
                (group.words / group.target_words).median()
            ),
        }
        for family, group in sized.groupby("family")
    ]
    form = [
        {
            "format": value,
            "mobile": bool(mobile),
            "turns": len(group),
            "wrong_format_rate": float(group.p_wrong_format.mean()),
        }
        for (value, mobile), group in treated[treated.format.notna()].groupby(
            ["format", "mobile"]
        )
    ]
    expertise = {
        arm: {
            "turns": len(group),
            # A raised point is taken up when the person moves on with it.
            "moved_on_rate": float(group.p_moved.mean()),
            "goals_per_turn": float(group.goals.mean()),
        }
        for arm, group in treated[treated.expertise.notna()].groupby(
            "expertise"
        )
    }
    return {"length": length, "form": form, "expertise": expertise}


def _launches(treated: pd.DataFrame, unit: str) -> dict:
    """Returns what each ablated field and the candidate catalog are worth.

    A field asked or not, a catalog or the other, are random draws: their
    comparisons are measured.
    """
    asked = treated[treated.ablated.isna()]
    candidate = treated.catalog == vocab.Catalog.CANDIDATE
    return {
        "field_ablation": {
            field: {
                "without_it": _summary(treated[treated.ablated == field], unit),
                "saving_of_asking": _measured(
                    asked, treated[treated.ablated == field], unit
                ),
            }
            for field in config.settings.ablation_fields
        },
        "catalogs": {
            "by_catalog": {
                c: _summary(group, unit)
                for c, group in treated.groupby("catalog")
            },
            "candidate_saving": _measured(
                treated[candidate], treated[~candidate], unit
            ),
        },
    }


async def compute(session: Session) -> dict:
    """Returns every metric."""
    now = await session.scalar(sa.select(sa.func.now()))
    per_user = await replies.by_user(session, now)
    unit = "user_id" if per_user else "conversation_id"
    p = await params_lib.load_params(session)
    every = pd.DataFrame(await turns(session, p))
    out = {"randomization_unit": unit}
    if not every.empty:
        out["fields"] = {
            "all": field_rates(every),
            "by_class": {
                family: field_rates(group)
                for family, group in every.groupby("family")
            },
        }
    frame = every[every.closed].copy() if not every.empty else every
    if not frame.empty:
        frame["spent"] = frame.words + frame.tool_words
        treated = frame[frame.grp == vocab.Arm.TREATMENT]
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
                arm: _measured(treated, frame[frame.grp == arm], unit)
                for arm in vocab.Arm
                if arm != vocab.Arm.TREATMENT
            },
            **_launches(treated, unit),
            **_axes(treated),
            # A chat's goals are settled once it ended: T days without a call.
            "ended_chats": {
                g: _summary(group, unit)
                for g, group in frame[
                    frame.conversation_id.map(
                        frame.groupby("conversation_id").last_call.max()
                    )
                    < now - datetime.timedelta(days=p.chat_end_days)
                ].groupby("grp")
            },
            "ambiguity": _calibration(frame),
            "weekly": _records(frame, ["week", "grp"], unit),
            "per_user": _records(frame, ["user_id", "grp"], unit),
            "compliance": [
                {"family": f, "host": h, "median_length_over_requested": r}
                for (f, h), r in compliance.items()
            ],
        }
        if not per_user:
            # Per chat, learning spills over into the arms: make it visible.
            out["arms_over_time"] = _records(
                held_back, ["user_id", "week"], unit
            )
    out |= {
        "chat_end_days": p.chat_end_days,
        "rephrase_distance": p.rephrase_distance,
        "tool_coverage": await _coverage(session),
        "widget": await _widget(session),
        "funnel": await _funnel(session, now),
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
