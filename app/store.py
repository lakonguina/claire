"""High-level writes to the event log and person connection cache.

Callers pass plain Python dicts; this module encrypts sensitive payloads
and inserts rows. Reading history for learning goes through the same
table (decrypt when needed).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core import config, crypto, vocab
from app.mcp.request_info import RequestInfo
from app.models import Event, Person


async def create_person(session: AsyncSession) -> Person:
    """Insert a new person and return it (id assigned by the database).

    Args:
        session: Open async SQLAlchemy session (caller commits).

    Returns:
        The new :class:`~app.models.Person` with ``id`` populated.
    """
    person = Person()
    session.add(person)
    await session.flush()
    return person


def touch_person(person: Person, info: RequestInfo) -> None:
    """Update the person's last-seen connection cache from this request.

    Does not commit; the caller owns the session. History still belongs in
    ``token_refresh`` / per-call event ``meta["request"]``.

    Args:
        person: Row to update in place.
        info: Snapshot from :func:`~app.mcp.request_info.collect_request_info`.
    """
    person.last_seen_at = datetime.now(timezone.utc)
    if info.user_agent:
        person.last_user_agent = info.user_agent
    if info.ip:
        person.last_ip = info.ip
    if info.mcp_client_name:
        person.mcp_client_name = info.mcp_client_name
    if info.mcp_client_version:
        person.mcp_client_version = info.mcp_client_version
    if info.mcp_protocol_version:
        person.mcp_protocol_version = info.mcp_protocol_version


async def append_event(
    session: AsyncSession,
    *,
    person_id: int,
    event_type: vocab.EventType | str,
    meta: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
    request: RequestInfo | dict[str, Any] | None = None,
    conversation_id: str | None = None,
    turn_id: str | None = None,
    schema_version: int = config.SCHEMA_VERSION,
) -> Event:
    """Append one event; encrypt ``payload`` when present.

    Args:
        session: Open async session (caller commits).
        person_id: Owner of the event.
        event_type: Kind of event (see :class:`~app.core.vocab.EventType`).
        meta: Non-secret JSON kept searchable (defaults to ``{}``).
        payload: Optional sensitive JSON; stored only as ciphertext.
        request: Full MCP/HTTP dump; stored under ``meta["request"]``.
        conversation_id: Chat handle, if this event is about a chat.
        turn_id: Turn handle, if this event is about a turn.
        schema_version: Payload contract version (defaults to current).

    Returns:
        The inserted :class:`~app.models.Event` (flushed, not necessarily
        committed).
    """
    event_meta: dict[str, Any] = dict(meta or {})
    if request is not None:
        event_meta["request"] = (
            request.as_meta()
            if isinstance(request, RequestInfo)
            else request
        )
    event = Event(
        person_id=person_id,
        conversation_id=conversation_id,
        turn_id=turn_id,
        type=str(event_type),
        schema_version=schema_version,
        meta=event_meta,
        payload_enc=(
            crypto.encrypt(json.dumps(payload, ensure_ascii=False))
            if payload is not None
            else None
        ),
    )
    session.add(event)
    await session.flush()
    return event


def read_payload(event: Event) -> dict[str, Any] | None:
    """Decrypt and parse ``event.payload_enc``, if any.

    Args:
        event: A row that may hold ciphertext.

    Returns:
        The original payload dict, or ``None`` if nothing was stored.
    """
    if event.payload_enc is None:
        return None
    return json.loads(crypto.decrypt(event.payload_enc))
