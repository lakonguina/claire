"""Turn loop against Postgres (skipped if DB is down)."""

from __future__ import annotations

import pytest

from app import store
from app.core import vocab
from app.mcp.request_info import RequestInfo
from app.mcp.schemas import PlanReplyIn, PlannedPartIn, ShowReplyIn, ShownPart
from app.mcp import turn


@pytest.mark.asyncio
async def test_run_plan_and_show(db_session) -> None:
    info = RequestInfo(ip="127.0.0.1", user_agent="pytest")
    person = await store.create_person(db_session)
    plan = PlanReplyIn(
        parts=[
            PlannedPartIn(
                name="main",
                words=100,
                form=[vocab.Form.TEXT],
            )
        ],
        user_message="hello world",
        model="test-model",
    )
    out = await turn.run_plan(
        db_session, person_id=person.id, plan=plan, request=info
    )
    assert out.conversation_id.startswith("c_")
    assert out.turn_id.startswith("t_")
    assert out.mode in (vocab.Mode.ADAPTED, vocab.Mode.NEUTRAL)

    shown = await turn.run_show(
        db_session,
        person_id=person.id,
        show=ShowReplyIn(
            turn_id=out.turn_id,
            parts=[ShownPart(name="main", text="hi")],
        ),
        request=info,
        mode="fallback",
    )
    assert shown.ok is True

    plan2 = PlanReplyIn(
        conversation_id=out.conversation_id,
        last_turn_id=out.turn_id,
        parts=[PlannedPartIn(name="main", words=40)],
        model="test-model",
    )
    out2 = await turn.run_plan(
        db_session, person_id=person.id, plan=plan2, request=info
    )
    assert out2.conversation_id == out.conversation_id
    assert out2.turn_id != out.turn_id
