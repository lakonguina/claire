"""Pydantic schemas for Claire MCP tools — single source of truth.

Tool handlers must take these models (or fields derived only from them).
Do not re-declare the same arguments in ``server.py``.

Parts are keyed by ``name`` (chosen by the host LLM): the same string
flows plan → Claire result → show_reply. No parallel index.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from app.core import config, vocab

_USER_MESSAGE_WORDS = config.settings.user_message_words


class PlannedPartIn(BaseModel):
    """One slice of the reply — identified by ``name``.

    The host LLM picks a short ``name`` (e.g. ``intro``, ``steps``), plus
    ``form`` / ``words``. Claire may correct form and length on the way
    out but echoes the same ``name``.
    """

    name: str = Field(
        description=(
            "Short id unique within this plan (e.g. intro, steps); "
            "echoed in the result and required again in show_reply."
        ),
        min_length=1,
        max_length=64,
    )
    form: list[vocab.Form] | None = Field(
        default=None,
        description=(
            "Forms for this part, in order. List shapes: "
            "bullets (- items), numbered_list (1. 2.), checklist (- [ ]). "
            "Example: [text, bullets]."
        ),
    )
    form_forced: bool = Field(
        default=False,
        description=(
            "True when the person imposed the form. On a follow-up plan, "
            "this is the learning signal that the previous form was wrong."
        ),
    )
    words: int = Field(
        description=(
            "Word length for this part. Host estimate when unforced; "
            "absolute target when words_forced is true."
        ),
        ge=1,
    )
    words_forced: bool = Field(
        default=False,
        description=(
            "True when the person gave an explicit length target "
            "(e.g. “80 words”, “3 bullets”). Claire uses words as-is "
            "(snapped to a bucket). Do not combine with length_forced. "
            "On a follow-up plan, signals the previous length was wrong."
        ),
    )
    length_forced: vocab.ForcedLength | None = Field(
        default=None,
        description=(
            "shorter / longer when the person asked for a direction only "
            "(no number). Claire chooses how much. Omit when words is the "
            "assistant's own estimate. Do not combine with words_forced. "
            "On a follow-up plan, signals the previous length was wrong."
        ),
    )

    @model_validator(mode="after")
    def _length_force_mode(self) -> PlannedPartIn:
        """Absolute target and directional nudge are mutually exclusive."""
        if self.words_forced and self.length_forced is not None:
            raise ValueError(
                "Use words_forced with an absolute words target, or "
                "length_forced (shorter/longer), not both."
            )
        if self.form_forced and not self.form:
            raise ValueError("form_forced requires a non-empty form.")
        return self


class PlanReplyIn(BaseModel):
    """Full input of ``claire_plan_reply`` (no duplicate length/form fields)."""

    conversation_id: str | None = Field(
        default=None,
        description="Chat handle from a previous Claire result; omit on first turn.",
    )
    last_turn_id: str | None = Field(
        default=None,
        description="Previous turn handle; omit on first turn.",
    )
    messages_since_last_call: int | None = Field(
        default=None,
        description="Replies given without calling Claire since last call.",
        ge=0,
    )
    model: str | None = Field(
        default=None,
        description=(
            "Host model id for this turn; always pass it (may change mid-chat)."
        ),
    )
    user_message: str | None = Field(
        default=None,
        description=(
            f"Full user text when it has at most {_USER_MESSAGE_WORDS} words. "
            "Do not send with user_request / user_message_chars."
        ),
    )
    user_request: str | None = Field(
        default=None,
        description=(
            f"Short request only when the user message exceeds "
            f"{_USER_MESSAGE_WORDS} words (with user_message_chars)."
        ),
    )
    user_message_chars: int | None = Field(
        default=None,
        description=(
            f"Char count of a long message (>{_USER_MESSAGE_WORDS} words); "
            "only with user_request, never with user_message."
        ),
        ge=0,
    )
    attachments: bool | None = Field(
        default=None,
        description="True if an attachment or image is present.",
    )
    parts: list[PlannedPartIn] = Field(
        description=(
            "Reply slices: each needs unique name, form, words, and optional "
            "forced flags. Put all form/length here — no top-level length field."
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def _message_and_names(self) -> PlanReplyIn:
        """Validate message mode and unique part names."""
        if self.user_message and (
            self.user_request is not None or self.user_message_chars is not None
        ):
            raise ValueError(
                "Send user_message alone, or user_request + user_message_chars, not both."
            )
        names = [part.name for part in self.parts]
        if len(names) != len(set(names)):
            raise ValueError("Each part.name must be unique within the plan.")
        return self


class PlannedPartOut(BaseModel):
    """One part Claire wants the assistant to write — same ``name`` as input."""

    name: str = Field(
        description="Echo of plan.parts[].name; use this key in show_reply.",
        min_length=1,
        max_length=64,
    )
    form: list[vocab.Form] = Field(
        description="Ordered forms for this part.",
    )
    words: int = Field(description="Target length for this part, in words.", ge=1)


class PlanReplyOut(BaseModel):
    """Result of ``claire_plan_reply``."""

    conversation_id: str
    turn_id: str
    mode: vocab.Mode
    parts: list[PlannedPartOut] = Field(
        default_factory=list,
        description="Empty when mode is neutral (holdout). Match by name.",
    )


class ShownPart(BaseModel):
    """One written part sent back to Claire."""

    name: str = Field(
        description="Must match PlannedPartOut.name from claire_plan_reply.",
        min_length=1,
        max_length=64,
    )
    text: str = Field(description="Final text for that part.")


class ShowReplyIn(BaseModel):
    """Full input of ``claire_show_reply``."""

    turn_id: str = Field(description="Turn handle from claire_plan_reply.")
    parts: list[ShownPart] = Field(
        description="Written parts keyed by name from the plan result.",
        min_length=1,
    )


class ShowReplyOut(BaseModel):
    """Result of ``claire_show_reply``."""

    ok: bool = True
    turn_id: str | None = Field(
        default=None,
        description="Echo of the turn handle (for the MCP App widget).",
    )
    mode: str = Field(
        default="fallback",
        description="``app`` when the host renders the MCP App; else ``fallback``.",
    )
    parts: list[ShownPart] = Field(
        default_factory=list,
        description="Echo of written parts for the widget to display.",
    )


class RecordEventIn(BaseModel):
    """Widget → server event (view, render error, …)."""

    turn_id: str = Field(description="Turn handle from claire_show_reply.")
    kind: str = Field(
        description="Event kind (e.g. reply_view, render_error).",
        min_length=1,
        max_length=64,
    )
    value: str | int | float | None = Field(
        default=None,
        description="Optional payload (count, message, …).",
    )
