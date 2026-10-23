"""MCP turn loop: auth caller, plan / show, close previous turn, log events.

Conversations and turns are handles (``c_…`` / ``t_…``) in the event log.
Holdout is a stable hash of ``conversation_id``.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

import sqlalchemy as sa
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.mcpserver.context import Context
from sqlalchemy.ext.asyncio import AsyncSession

from app import models
from app import store
from app.core import config, vocab
from app.mcp.forms import detect_forms
from app.mcp.request_info import RequestInfo, collect_request_info
from app.mcp.schemas import (
    PlanReplyIn,
    PlanReplyOut,
    ShowReplyIn,
    ShowReplyOut,
)
from app.model import decision
from app.model import rewards


def person_id_from_ctx(ctx: Context) -> int:
    """Return ``Person.id`` from the bearer access-token subject."""
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError) as exc:
        raise PermissionError("unauthenticated") from exc
    user = getattr(request, "user", None)
    if not isinstance(user, AuthenticatedUser):
        raise PermissionError("unauthenticated")
    subject = user.access_token.subject
    if not subject:
        raise PermissionError("unauthenticated")
    return int(subject)


async def bind_caller(
    session: AsyncSession, ctx: Context
) -> tuple[int, RequestInfo]:
    """Resolve person, collect request info, refresh connection cache."""
    person_id = person_id_from_ctx(ctx)
    info = collect_request_info(ctx)
    person = await session.get(models.Person, person_id)
    if person is None:
        raise LookupError(f"person {person_id} missing")
    store.touch_person(person, info)
    return person_id, info


def new_id(prefix: str) -> str:
    """Mint a short public handle (``c_…`` / ``t_…``)."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def is_holdout(conversation_id: str, share: float | None = None) -> bool:
    """Stable holdout draw from ``conversation_id`` (hash → [0, 1))."""
    if share is None:
        share = config.settings.holdout_share
    digest = hashlib.sha256(conversation_id.encode()).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    return unit < share


def plan_payload(plan: PlanReplyIn) -> dict[str, Any]:
    """JSON payload for plan_call; drop overlong ``user_message``."""
    data = plan.model_dump(mode="json", exclude_none=True)
    message = plan.user_message
    if message is None:
        return data
    if len(message.split()) <= config.settings.user_message_words:
        return data
    data.pop("user_message", None)
    data.setdefault("user_message_chars", len(message))
    return data


async def _decision_meta(
    session: AsyncSession, turn_id: str
) -> dict[str, Any] | None:
    row = await session.scalar(
        sa.select(models.Event.meta)
        .where(
            models.Event.turn_id == turn_id,
            models.Event.type == vocab.EventType.DECISION,
        )
        .order_by(models.Event.id.desc())
        .limit(1)
    )
    return dict(row) if row else None


async def _conversation_id_for_turn(
    session: AsyncSession, turn_id: str
) -> str | None:
    """Look up the chat handle already logged for this turn."""
    return await session.scalar(
        sa.select(models.Event.conversation_id)
        .where(
            models.Event.turn_id == turn_id,
            models.Event.conversation_id.is_not(None),
        )
        .order_by(models.Event.id.asc())
        .limit(1)
    )


async def close_turn(
    session: AsyncSession,
    *,
    person_id: int,
    conversation_id: str | None,
    turn_id: str,
    request: RequestInfo,
    meta: dict[str, Any] | None = None,
    form_forced: bool = False,
    length_forced: bool = False,
) -> None:
    """Log ``turn_closed`` and train bandits from forced flags."""
    event_meta = dict(meta or {})
    event_meta["form_forced"] = form_forced
    event_meta["length_forced"] = length_forced
    decision_meta = await _decision_meta(session, turn_id)
    if decision_meta is not None:
        person = await session.get(models.Person, person_id)
        if person is not None:
            rewards.apply_turn_outcome(
                person,
                decision_meta,
                form_forced=form_forced,
                length_forced=length_forced,
            )
    await store.append_event(
        session,
        person_id=person_id,
        event_type=vocab.EventType.TURN_CLOSED,
        conversation_id=conversation_id,
        turn_id=turn_id,
        request=request,
        meta=event_meta,
    )


async def run_plan(
    session: AsyncSession,
    *,
    person_id: int,
    plan: PlanReplyIn,
    request: RequestInfo,
) -> PlanReplyOut:
    """Close last turn if any, mint ids, log plan_call + decision."""
    conversation_id = plan.conversation_id or new_id("c")
    turn_id = new_id("t")
    holdout = is_holdout(conversation_id)
    mode = vocab.Mode.NEUTRAL if holdout else vocab.Mode.ADAPTED

    if plan.last_turn_id:
        signal = rewards.forced_signal(plan.parts)
        await close_turn(
            session,
            person_id=person_id,
            conversation_id=plan.conversation_id,
            turn_id=plan.last_turn_id,
            request=request,
            meta={
                "via": "plan_reply",
                "words_forced": signal["words_forced"],
            },
            form_forced=signal["form_forced"],
            length_forced=signal["length_forced"],
        )

    await store.append_event(
        session,
        person_id=person_id,
        event_type=vocab.EventType.PLAN_CALL,
        conversation_id=conversation_id,
        turn_id=turn_id,
        request=request,
        meta={
            "mode": mode,
            "holdout": holdout,
            "model": plan.model,
            "messages_since_last_call": plan.messages_since_last_call,
            "attachments": plan.attachments,
            "policy_version": config.settings.policy_version,
        },
        payload=plan_payload(plan),
    )

    person = await session.get(models.Person, person_id)
    parts_meta: list[dict] = []
    if holdout:
        parts = []
    else:
        parts, parts_meta = decision.decide_parts(
            plan.parts,
            form_counts=person.form_counts if person else {},
            words_counts=person.words_counts if person else {},
        )
    await store.append_event(
        session,
        person_id=person_id,
        event_type=vocab.EventType.DECISION,
        conversation_id=conversation_id,
        turn_id=turn_id,
        request=request,
        meta={
            "mode": mode,
            "holdout": holdout,
            "policy_version": config.settings.policy_version,
            "parts": parts_meta,
        },
        payload={
            "parts": [part.model_dump(mode="json") for part in parts],
        },
    )
    return PlanReplyOut(
        conversation_id=conversation_id,
        turn_id=turn_id,
        mode=mode,
        parts=parts,
    )


async def run_show(
    session: AsyncSession,
    *,
    person_id: int,
    show: ShowReplyIn,
    request: RequestInfo,
    mode: str = "fallback",
) -> ShowReplyOut:
    """Log written reply parts for a turn."""
    conversation_id = await _conversation_id_for_turn(session, show.turn_id)
    parts_meta = [
        {
            "name": part.name,
            "forms": [form.value for form in detect_forms(part.text)],
            "chars": len(part.text),
        }
        for part in show.parts
    ]
    await store.append_event(
        session,
        person_id=person_id,
        event_type=vocab.EventType.REPLY,
        conversation_id=conversation_id,
        turn_id=show.turn_id,
        request=request,
        meta={"mode": mode, "parts": parts_meta},
        payload={
            "parts": [part.model_dump(mode="json") for part in show.parts],
        },
    )
    return ShowReplyOut(
        ok=True,
        turn_id=show.turn_id,
        mode=mode,
        parts=list(show.parts),
    )


async def record_widget(
    session: AsyncSession,
    *,
    person_id: int,
    turn_id: str,
    kind: str,
    value: str | int | float | None,
    request: RequestInfo,
) -> None:
    """Append a widget event (``reply_view``, ``render_error``, …)."""
    conversation_id = await _conversation_id_for_turn(session, turn_id)
    await store.append_event(
        session,
        person_id=person_id,
        event_type=vocab.EventType.WIDGET,
        conversation_id=conversation_id,
        turn_id=turn_id,
        request=request,
        meta={"kind": kind, "value": value},
    )
