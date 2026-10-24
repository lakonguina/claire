"""Closed vocabularies: the only allowed labels in tools and events.

These strings are a contract with assistants and with old logged data.
We **add** values over time; we never rename or redefine existing ones
(so yesterday's events stay meaningful).

MVP deliberately omits Content / Device / PastedKind / ToolUsed. Device
is not a learning key (the same person uses phone and desktop); we may
still record User-Agent on connect (``token_refresh``).
"""

from __future__ import annotations

from enum import StrEnum


class Form(StrEnum):
    """How a slice of the reply should look on the page.

    Three list shapes (do not confuse them):

    * ``bullets`` — unordered ``-`` / ``*`` items
    * ``numbered_list`` — ordered ``1.`` ``2.`` steps
    * ``checklist`` — task list ``- [ ]`` / ``- [x]``
    """

    TEXT = "text"
    BULLETS = "bullets"
    NUMBERED_LIST = "numbered_list"
    CHECKLIST = "checklist"
    TABLE = "table"
    CODE = "code"


class ForcedLength(StrEnum):
    """Person imposed a length change (not Claire's choice).

    On the *next* ``claire_plan_reply``, these flags (and ``form_forced``)
    are the learning signal for the previous turn.
    """

    SHORTER = "shorter"
    LONGER = "longer"


class Mode(StrEnum):
    """Whether Claire shapes the reply or stays out of the way."""

    ADAPTED = "adapted"
    NEUTRAL = "neutral"  # Holdout: answer like a normal assistant.


class Display(StrEnum):
    """How the host must deliver the reply (from MCP Apps capabilities)."""

    WIDGET = "widget"  # MCP App UI is available.
    FALLBACK = "fallback"  # No widget — answer goes in the chat.


class EventType(StrEnum):
    """Kinds of rows in the append-only ``events`` table."""

    TOKEN_REFRESH = "token_refresh"  # Connect / refresh; store UA here.
    PLAN_CALL = "plan_call"
    DECISION = "decision"
    REPLY = "reply"
    WIDGET = "widget"
    TURN_CLOSED = "turn_closed"


class Credential(StrEnum):
    """OAuth credential kinds stored by hash (never cleartext)."""

    REQUEST = "request"  # Pending /authorize → consent.
    CODE = "code"
    ACCESS = "access"
    REFRESH = "refresh"


# Candidate shapes the bandit may pick for a reply part (MVP: flat catalog).
FORM_OPTIONS: tuple[tuple[Form, ...], ...] = (
    (Form.TEXT,),
    (Form.BULLETS,),
    (Form.NUMBERED_LIST,),
    (Form.CHECKLIST,),
    (Form.TABLE,),
    (Form.CODE,),
    (Form.TEXT, Form.BULLETS),
    (Form.TEXT, Form.CODE),
    (Form.TEXT, Form.NUMBERED_LIST),
)

# Allowed reply lengths (words). The model picks a bucket, not a free int.
WORD_BUCKETS: tuple[int, ...] = (30, 50, 80, 120, 180, 270)
