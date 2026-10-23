"""Holdout, plan payload, forms detection, bandit replay."""

from __future__ import annotations

from types import SimpleNamespace

from app.core import vocab
from app.mcp import turn
from app.mcp.forms import detect_forms
from app.mcp.schemas import PlanReplyIn, PlannedPartIn
from app.model import replay


def test_holdout_stable_and_bounds() -> None:
    assert turn.is_holdout("c_abc", share=0.0) is False
    assert turn.is_holdout("c_abc", share=1.0) is True
    assert turn.is_holdout("c_abc", share=0.03) == turn.is_holdout(
        "c_abc", share=0.03
    )


def test_plan_payload_strips_long_message() -> None:
    long = ("word " * 200).strip()
    plan = PlanReplyIn(
        parts=[PlannedPartIn(name="x", words=30)],
        user_message=long,
    )
    payload = turn.plan_payload(plan)
    assert "user_message" not in payload
    assert payload["user_message_chars"] == len(long)


def test_detect_forms() -> None:
    assert vocab.Form.BULLETS in detect_forms("- a\n- b")
    assert vocab.Form.NUMBERED_LIST in detect_forms("1. a\n2. b")
    assert vocab.Form.CODE in detect_forms("hi\n\n```py\nx\n```")
    assert vocab.Form.CHECKLIST in detect_forms("- [ ] x\n- [ ] y")


def test_replay_rebuilds_counts() -> None:
    events = [
        SimpleNamespace(
            person_id=1,
            type=vocab.EventType.DECISION,
            turn_id="t1",
            meta={
                "mode": "adapted",
                "parts": [{"form_arm": "text", "words_arm": "80"}],
            },
        ),
        SimpleNamespace(
            person_id=1,
            type=vocab.EventType.TURN_CLOSED,
            turn_id="t1",
            meta={"form_forced": False, "length_forced": False},
        ),
        SimpleNamespace(
            person_id=1,
            type=vocab.EventType.DECISION,
            turn_id="t2",
            meta={
                "mode": "adapted",
                "parts": [{"form_arm": "bullets", "words_arm": "50"}],
            },
        ),
        SimpleNamespace(
            person_id=1,
            type=vocab.EventType.TURN_CLOSED,
            turn_id="t2",
            meta={"form_forced": True, "length_forced": False},
        ),
    ]
    states = replay.rebuild_bandits(events)
    assert states[1].form_counts.get("text") == 1.0
    assert "bullets" not in states[1].form_counts
    assert states[1].words_counts == {"80": 1.0, "50": 1.0}
