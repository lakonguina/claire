"""Shape event rows into a chat timeline for the admin UI."""

from __future__ import annotations

import json
from typing import Any

from app import models, store
from app.core import vocab

_TOOL_LABELS = {
    vocab.EventType.PLAN_CALL: "claire_plan_reply",
    vocab.EventType.REPLY: "claire_show_reply",
    vocab.EventType.WIDGET: "claire_record_event",
}


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _payload(event: models.Event) -> dict[str, Any] | None:
    try:
        return store.read_payload(event)
    except Exception:  # pragma: no cover
        return None


def _user_text(payload: dict[str, Any] | None) -> str | None:
    if not payload:
        return None
    for key in ("user_message", "user_request"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _reply_parts(payload: dict[str, Any] | None) -> list[dict[str, str]]:
    if not payload:
        return []
    parts: list[dict[str, str]] = []
    for part in payload.get("parts") or []:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        parts.append(
            {
                "name": str(part.get("name") or "part"),
                "text": text.strip(),
            }
        )
    return parts


def _preview(text: str | None, limit: int = 120) -> str:
    if not text:
        return "Conversation"
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1].rstrip() + "…"


def _tool_body(event: models.Event) -> dict[str, Any]:
    """Compact JSON body for one tool call in the UI."""
    body: dict[str, Any] = {
        "at": event.created_at.isoformat() if event.created_at else None,
    }
    if event.turn_id:
        body["turn_id"] = event.turn_id
    meta = dict(event.meta or {})
    if meta:
        body["meta"] = meta
    payload = _payload(event)
    if payload is not None:
        body["payload"] = payload
    elif event.payload_enc is not None:
        body["payload"] = "<decrypt failed>"
    return body


def build_turns(events: list[models.Event]) -> list[dict[str, Any]]:
    """Group events into chat turns: user → tools → AI reply."""
    order: list[str] = []
    buckets: dict[str, list[models.Event]] = {}
    for event in events:
        key = event.turn_id or f"evt_{event.id}"
        if key not in buckets:
            order.append(key)
            buckets[key] = []
        buckets[key].append(event)

    turns: list[dict[str, Any]] = []
    for key in order:
        group = buckets[key]
        user_text = None
        reply_parts: list[dict[str, str]] = []
        model = None
        mode = None
        replay = False
        replay_attempt = None
        when = group[0].created_at
        decision_body: dict[str, Any] | None = None
        tool_events: list[models.Event] = []

        for event in group:
            payload = _payload(event)
            meta = event.meta or {}
            if event.type == vocab.EventType.PLAN_CALL:
                user_text = _user_text(payload) or user_text
                model = meta.get("model") or model
                mode = meta.get("mode") or mode
                if meta.get("replay"):
                    replay = True
                    replay_attempt = meta.get("replay_attempt")
                tool_events.append(event)
            elif event.type == vocab.EventType.REPLY:
                reply_parts = _reply_parts(payload) or reply_parts
                mode = meta.get("mode") or mode
                tool_events.append(event)
            elif event.type == vocab.EventType.WIDGET:
                tool_events.append(event)
            elif event.type == vocab.EventType.DECISION:
                mode = meta.get("mode") or mode
                decision_body = _tool_body(event)

        tools: list[dict[str, Any]] = []
        for event in tool_events:
            body = _tool_body(event)
            if (
                event.type == vocab.EventType.PLAN_CALL
                and decision_body is not None
            ):
                body["decision"] = decision_body
            tools.append(
                {
                    "name": _TOOL_LABELS[event.type],
                    "json": _dump(body),
                }
            )

        if not user_text and not reply_parts and not tools:
            continue

        turns.append(
            {
                "turn_id": key if not key.startswith("evt_") else None,
                "when": when,
                "model": model,
                "mode": mode,
                "replay": replay,
                "replay_attempt": replay_attempt,
                "user_text": user_text,
                "reply_parts": reply_parts,
                "reply_text": "\n\n".join(part["text"] for part in reply_parts),
                "tools": tools,
                "tool_count": len(tools),
            }
        )
    return turns


def conversation_preview(
    events: list[models.Event],
) -> tuple[str, int]:
    """Return (preview snippet, exchange count) for a conversation list row."""
    turns = build_turns(events)
    exchanges = sum(
        1 for turn in turns if turn["user_text"] or turn["reply_parts"]
    )
    for turn in turns:
        if turn["user_text"]:
            return _preview(turn["user_text"]), exchanges
    for turn in turns:
        if turn["reply_text"]:
            return _preview(turn["reply_text"]), exchanges
    return "Conversation", exchanges
