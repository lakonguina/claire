"""MCP turn loop: auth caller, plan / show, close previous turn, log events.

Conversations and turns are handles (``c_…`` / ``t_…``) in the event log.
Holdout is a stable hash of ``conversation_id``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
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
    PlannedPartOut,
    PlanReplyOut,
    ShowReplyIn,
    ShowReplyOut,
)
from app.model import decision
from app.model import rewards

WIDGET_INSTRUCTION = (
    "display=widget: call claire_show_reply once with your whole answer. "
    "Do not write that answer (or a summary) in the chat — the widget is "
    "the reply. End your turn after show_reply. "
    "Do not call claire_plan_reply again for this reply."
)
FALLBACK_INSTRUCTION = (
    "display=fallback: this host has no Claire widget. Write your whole "
    "answer once in the chat as your normal message. "
    "Do not call claire_show_reply. "
    "Do not call claire_plan_reply again for this reply."
)
ADAPTED_PREFIX = "Answer only what the user asked; do not digress. "


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


def message_key(plan: PlanReplyIn) -> str:
    """Keyed hash of the user's message (compared, never stored in clear)."""
    return hmac.new(
        config.settings.data_key.encode(),
        json.dumps([plan.user_message, plan.user_request]).encode(),
        hashlib.sha256,
    ).hexdigest()


def _instruction(*, holdout: bool, display: vocab.Display) -> str:
    base = (
        FALLBACK_INSTRUCTION
        if display == vocab.Display.FALLBACK
        else WIDGET_INSTRUCTION
    )
    return base if holdout else ADAPTED_PREFIX + base


def show_note(display: vocab.Display) -> str:
    """Unambiguous closing line for ``claire_show_reply``."""
    if display == vocab.Display.WIDGET:
        return (
            "The reply is displayed to the person in the widget. "
            "Your turn is over: write nothing."
        )
    return (
        "No widget on this host. You should already have written the answer "
        "in the chat. Write nothing more."
    )


async def _given_plan(
    session: AsyncSession,
    plan_call: models.Event,
    *,
    display: vocab.Display,
) -> PlanReplyOut | None:
    """Rebuild the result already returned for ``plan_call``'s turn."""
    decided = await session.scalar(
        sa.select(models.Event).where(
            models.Event.turn_id == plan_call.turn_id,
            models.Event.type == vocab.EventType.DECISION,
        )
    )
    if decided is None:
        return None
    payload = store.read_payload(decided) or {}
    holdout = bool(decided.meta.get("holdout"))
    return PlanReplyOut(
        conversation_id=plan_call.conversation_id,
        turn_id=plan_call.turn_id,
        mode=decided.meta["mode"],
        display=display,
        instruction=_instruction(holdout=holdout, display=display),
        parts=[PlannedPartOut(**part) for part in payload.get("parts", [])],
    )


async def _is_shown(session: AsyncSession, turn_id: str) -> bool:
    return (
        await session.scalar(
            sa.select(models.Event.id)
            .where(
                models.Event.turn_id == turn_id,
                models.Event.type == vocab.EventType.REPLY,
            )
            .limit(1)
        )
    ) is not None


def _plan_calls_q(*, person_id: int, conversation_id: str | None):
    """Base query for this person's plan_call events (optionally one chat)."""
    statement = sa.select(models.Event).where(
        models.Event.person_id == person_id,
        models.Event.type == vocab.EventType.PLAN_CALL,
    )
    if conversation_id:
        statement = statement.where(
            models.Event.conversation_id == conversation_id
        )
    return statement


async def _abandoned_attempts(
    session: AsyncSession,
    *,
    person_id: int,
    conversation_id: str | None,
    last_turn_id: str | None,
    key: str,
) -> list[models.Event]:
    """Prior plan_calls after ``last_turn_id`` for a *different* user message.

    Those are abandoned when the person edits and retries (chat replay).
    """
    if not conversation_id or not last_turn_id:
        return []
    rows = (
        await session.scalars(
            _plan_calls_q(
                person_id=person_id, conversation_id=conversation_id
            )
            .where(
                models.Event.meta["after_turn_id"].astext == last_turn_id
            )
            .order_by(models.Event.id.asc())
        )
    ).all()
    return [row for row in rows if row.meta.get("message_key") != key]


async def _repeated_plan(
    session: AsyncSession,
    *,
    person_id: int,
    plan: PlanReplyIn,
    key: str,
    display: vocab.Display,
) -> PlanReplyOut | None:
    """Return the turn this call repeats, if any.

    A call repeats a turn when it would give that turn a second successor
    for the *same* message, or when that turn is still open (planned, not
    shown) for the same message. ``display`` / ``instruction`` follow the
    current host capabilities (not the first attempt's).
    """
    plan_calls = _plan_calls_q(
        person_id=person_id, conversation_id=plan.conversation_id
    )
    if plan.conversation_id and plan.last_turn_id:
        same = await session.scalar(
            plan_calls.where(
                models.Event.meta["after_turn_id"].astext
                == plan.last_turn_id,
                models.Event.meta["message_key"].astext == key,
            ).limit(1)
        )
        if same is not None:
            return await _given_plan(session, same, display=display)
    latest = await session.scalar(
        plan_calls.order_by(models.Event.id.desc()).limit(1)
    )
    if (
        latest is not None
        and latest.meta.get("message_key") == key
        and not await _is_shown(session, latest.turn_id)
    ):
        return await _given_plan(session, latest, display=display)
    return None


async def _is_closed(session: AsyncSession, turn_id: str) -> bool:
    return (
        await session.scalar(
            sa.select(models.Event.id)
            .where(
                models.Event.turn_id == turn_id,
                models.Event.type == vocab.EventType.TURN_CLOSED,
            )
            .limit(1)
        )
    ) is not None


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
    display: vocab.Display,
) -> PlanReplyOut:
    """Close last turn if any, mint ids, log plan_call + decision.

    ``display`` comes from the host's MCP Apps capabilities (known at
    connect). A repeat of the same call returns the first result and logs
    nothing (instruction / display refreshed for this host).
    """
    # Serialize this person's plan calls so a parallel repeat sees the first.
    await session.execute(sa.select(sa.func.pg_advisory_xact_lock(person_id)))
    key = message_key(plan)
    repeated = await _repeated_plan(
        session,
        person_id=person_id,
        plan=plan,
        key=key,
        display=display,
    )
    if repeated is not None:
        return repeated

    conversation_id = plan.conversation_id or new_id("c")
    turn_id = new_id("t")
    holdout = is_holdout(conversation_id)
    mode = vocab.Mode.NEUTRAL if holdout else vocab.Mode.ADAPTED

    abandoned = await _abandoned_attempts(
        session,
        person_id=person_id,
        conversation_id=plan.conversation_id,
        last_turn_id=plan.last_turn_id,
        key=key,
    )
    replay_meta: dict[str, Any] = {}
    if abandoned:
        replay_meta = {
            "replay": True,
            "replay_attempt": len(abandoned) + 1,
            "replay_of_turn_id": abandoned[-1].turn_id,
            "replay_of_turn_ids": [event.turn_id for event in abandoned],
        }

    if plan.last_turn_id and not await _is_closed(session, plan.last_turn_id):
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
            "display": display,
            "holdout": holdout,
            "model": plan.model,
            "messages_since_last_call": plan.messages_since_last_call,
            "after_turn_id": plan.last_turn_id,
            "message_key": key,
            "attachments": plan.attachments,
            "policy_version": config.settings.policy_version,
            **replay_meta,
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
            "display": display,
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
        display=display,
        instruction=_instruction(holdout=holdout, display=display),
        parts=parts,
    )


async def run_show(
    session: AsyncSession,
    *,
    person_id: int,
    show: ShowReplyIn,
    request: RequestInfo,
    display: vocab.Display,
) -> ShowReplyOut:
    """Log written reply parts for a turn; a repeat for the turn logs nothing."""
    note = show_note(display)
    if await _is_shown(session, show.turn_id):
        return ShowReplyOut(
            ok=True,
            turn_id=show.turn_id,
            display=display,
            note=note,
            parts=list(show.parts),
        )
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
        meta={"display": display, "parts": parts_meta},
        payload={
            "parts": [part.model_dump(mode="json") for part in show.parts],
        },
    )
    return ShowReplyOut(
        ok=True,
        turn_id=show.turn_id,
        display=display,
        note=note,
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
