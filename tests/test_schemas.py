"""Plan / show input validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core import vocab
from app.mcp.schemas import PlanReplyIn, PlannedPartIn


def test_unique_part_names() -> None:
    with pytest.raises(ValidationError):
        PlanReplyIn(
            parts=[
                PlannedPartIn(name="a", words=30),
                PlannedPartIn(name="a", words=40),
            ]
        )


def test_message_modes_exclusive() -> None:
    with pytest.raises(ValidationError):
        PlanReplyIn(
            user_message="hi",
            user_request="hi",
            parts=[PlannedPartIn(name="a", words=30)],
        )


def test_words_forced_xor_length_forced() -> None:
    with pytest.raises(ValidationError):
        PlannedPartIn(
            name="a",
            words=40,
            words_forced=True,
            length_forced=vocab.ForcedLength.SHORTER,
        )


def test_form_forced_requires_form() -> None:
    with pytest.raises(ValidationError):
        PlannedPartIn(name="a", words=40, form_forced=True)
