"""Lean KPIs for the admin home — few numbers, tied to Claire's thesis.

One Metric That Matters (early stage): do adapted replies use fewer words
*without* more person corrections than holdout?

Computable today from the event log (no goal/survey yet):
* median reply words — adapted vs holdout
* correction rate — share of closed turns with form/length forced
* loop rate — show after plan
* widget view rate — person actually saw the reply
* active people (7d) — is anyone using it
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app import models, store
from app.core import vocab

_WORD = re.compile(r"\w+(?:['’-]\w+)*", re.UNICODE)


@dataclass(frozen=True)
class ArmKpis:
    """One arm: adapted or holdout."""

    turns_shown: int
    turns_closed: int
    median_words: float | None
    correction_rate: float | None


@dataclass(frozen=True)
class Snapshot:
    """Admin home KPI strip."""

    people_total: int
    active_people_7d: int
    plans: int
    replies: int
    loop_rate: float | None
    view_rate: float | None
    adapted: ArmKpis
    holdout: ArmKpis


def _words(text: str) -> int:
    return len(_WORD.findall(text or ""))


def _reply_words(payload: dict[str, Any] | None) -> int | None:
    if not payload:
        return None
    total = 0
    found = False
    for part in payload.get("parts") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            found = True
            total += _words(text)
    return total if found else None


def _rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def _arm(
    *,
    words: list[int],
    closed: int,
    corrected: int,
) -> ArmKpis:
    return ArmKpis(
        turns_shown=len(words),
        turns_closed=closed,
        median_words=float(median(words)) if words else None,
        correction_rate=_rate(corrected, closed),
    )


async def compute(session: AsyncSession) -> Snapshot:
    """Scan the event log once; return the admin KPI snapshot."""
    people_total = int(
        await session.scalar(sa.select(sa.func.count(models.Person.id))) or 0
    )
    since = datetime.now(timezone.utc) - timedelta(days=7)
    active_people_7d = int(
        await session.scalar(
            sa.select(sa.func.count(sa.distinct(models.Event.person_id))).where(
                models.Event.created_at >= since
            )
        )
        or 0
    )

    events = list(
        await session.scalars(
            sa.select(models.Event).order_by(models.Event.id.asc())
        )
    )

    holdout_by_turn: dict[str, bool] = {}
    reply_words: dict[str, int] = {}
    viewed: set[str] = set()
    closed_force: dict[str, bool] = {}
    plans = replies = 0

    for event in events:
        meta = event.meta or {}
        turn_id = event.turn_id
        if event.type == vocab.EventType.DECISION and turn_id:
            holdout_by_turn[turn_id] = bool(
                meta.get("holdout") or meta.get("mode") == vocab.Mode.NEUTRAL
            )
        elif event.type == vocab.EventType.PLAN_CALL:
            plans += 1
        elif event.type == vocab.EventType.REPLY and turn_id:
            replies += 1
            try:
                payload = store.read_payload(event)
            except Exception:  # pragma: no cover
                payload = None
            words = _reply_words(payload)
            if words is not None:
                reply_words[turn_id] = words
        elif event.type == vocab.EventType.WIDGET and turn_id:
            if meta.get("kind") == "reply_view":
                viewed.add(turn_id)
        elif event.type == vocab.EventType.TURN_CLOSED and turn_id:
            closed_force[turn_id] = bool(
                meta.get("form_forced") or meta.get("length_forced")
            )

    adapted_words: list[int] = []
    holdout_words: list[int] = []
    adapted_closed = adapted_corrected = 0
    holdout_closed = holdout_corrected = 0

    for turn_id, words in reply_words.items():
        if holdout_by_turn.get(turn_id, False):
            holdout_words.append(words)
        else:
            adapted_words.append(words)

    for turn_id, forced in closed_force.items():
        if holdout_by_turn.get(turn_id, False):
            holdout_closed += 1
            if forced:
                holdout_corrected += 1
        else:
            adapted_closed += 1
            if forced:
                adapted_corrected += 1

    return Snapshot(
        people_total=people_total,
        active_people_7d=active_people_7d,
        plans=plans,
        replies=replies,
        loop_rate=_rate(replies, plans),
        view_rate=_rate(len(viewed), replies),
        adapted=_arm(
            words=adapted_words,
            closed=adapted_closed,
            corrected=adapted_corrected,
        ),
        holdout=_arm(
            words=holdout_words,
            closed=holdout_closed,
            corrected=holdout_corrected,
        ),
    )


def as_template(snapshot: Snapshot) -> dict[str, Any]:
    """Plain dict for Jinja (formatted strings ready to show)."""

    def pct(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{100 * value:.0f}%"

    def words(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.0f}"

    return {
        "people_total": snapshot.people_total,
        "active_people_7d": snapshot.active_people_7d,
        "loop_rate": pct(snapshot.loop_rate),
        "view_rate": pct(snapshot.view_rate),
        "plans": snapshot.plans,
        "replies": snapshot.replies,
        "adapted_words": words(snapshot.adapted.median_words),
        "holdout_words": words(snapshot.holdout.median_words),
        "adapted_correction": pct(snapshot.adapted.correction_rate),
        "holdout_correction": pct(snapshot.holdout.correction_rate),
        "adapted_shown": snapshot.adapted.turns_shown,
        "holdout_shown": snapshot.holdout.turns_shown,
        "adapted_closed": snapshot.adapted.turns_closed,
        "holdout_closed": snapshot.holdout.turns_closed,
    }
