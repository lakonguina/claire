"""Turn loop against Postgres (skipped if DB is down)."""

from __future__ import annotations

import pytest
import sqlalchemy as sa

from app import models, store
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
        db_session,
        person_id=person.id,
        plan=plan,
        request=info,
        display=vocab.Display.WIDGET,
    )
    assert out.conversation_id.startswith("c_")
    assert out.turn_id.startswith("t_")
    assert out.mode in (vocab.Mode.ADAPTED, vocab.Mode.NEUTRAL)
    assert out.display == vocab.Display.WIDGET
    assert "claire_show_reply" in out.instruction

    shown = await turn.run_show(
        db_session,
        person_id=person.id,
        show=ShowReplyIn(
            turn_id=out.turn_id,
            parts=[ShownPart(name="main", text="hi")],
        ),
        request=info,
        display=vocab.Display.WIDGET,
    )
    assert shown.ok is True
    assert "write nothing" in shown.note.lower()

    plan2 = PlanReplyIn(
        conversation_id=out.conversation_id,
        last_turn_id=out.turn_id,
        parts=[PlannedPartIn(name="main", words=40)],
        model="test-model",
    )
    out2 = await turn.run_plan(
        db_session,
        person_id=person.id,
        plan=plan2,
        request=info,
        display=vocab.Display.FALLBACK,
    )
    assert out2.display == vocab.Display.FALLBACK
    assert "Do not call claire_show_reply" in out2.instruction
    assert out2.conversation_id == out.conversation_id
    assert out2.turn_id != out.turn_id


async def _types(session, turn_id: str) -> list[str]:
    rows = await session.scalars(
        sa.select(models.Event.type).where(models.Event.turn_id == turn_id)
    )
    return sorted(rows.all())


@pytest.mark.asyncio
async def test_repeated_calls_reuse_the_turn(db_session) -> None:
    info = RequestInfo(ip="127.0.0.1", user_agent="pytest")
    person = await store.create_person(db_session)

    async def plan(message: str, after=None):
        return await turn.run_plan(
            db_session,
            person_id=person.id,
            plan=PlanReplyIn(
                conversation_id=after.conversation_id if after else None,
                last_turn_id=after.turn_id if after else None,
                parts=[PlannedPartIn(name="main", words=80)],
                user_message=message,
            ),
            request=info,
            display=vocab.Display.WIDGET,
        )

    async def show(planned) -> None:
        await turn.run_show(
            db_session,
            person_id=person.id,
            show=ShowReplyIn(
                turn_id=planned.turn_id,
                parts=[ShownPart(name="main", text="answer")],
            ),
            request=info,
            display=vocab.Display.WIDGET,
        )

    first = await plan("question one")
    # Same call again, or passing the still-open turn as the last one.
    assert await plan("question one") == first
    assert await plan("question one", after=first) == first
    await show(first)
    await show(first)
    assert await _types(db_session, first.turn_id) == sorted(
        [
            vocab.EventType.PLAN_CALL,
            vocab.EventType.DECISION,
            vocab.EventType.REPLY,
        ]
    )

    second = await plan("question two", after=first)
    assert second.turn_id != first.turn_id
    await show(second)
    # A turn has one successor, even after it was shown.
    assert await plan("question two", after=first) == second

    # A new message on an unshown turn is a new turn, not a repeat.
    third = await plan("question three", after=second)
    fourth = await plan("question four", after=third)
    assert fourth.turn_id != third.turn_id


@pytest.mark.asyncio
async def test_edited_retry_is_logged_as_replay(db_session) -> None:
    info = RequestInfo(ip="127.0.0.1", user_agent="pytest")
    person = await store.create_person(db_session)

    first = await turn.run_plan(
        db_session,
        person_id=person.id,
        plan=PlanReplyIn(
            parts=[PlannedPartIn(name="main", words=80)],
            user_message="original question",
        ),
        request=info,
        display=vocab.Display.FALLBACK,
    )
    assert first.display == vocab.Display.FALLBACK
    assert "Do not call claire_show_reply" in first.instruction

    await turn.run_show(
        db_session,
        person_id=person.id,
        show=ShowReplyIn(
            turn_id=first.turn_id,
            parts=[ShownPart(name="main", text="first answer")],
        ),
        request=info,
        display=vocab.Display.FALLBACK,
    )

    attempt_a = await turn.run_plan(
        db_session,
        person_id=person.id,
        plan=PlanReplyIn(
            conversation_id=first.conversation_id,
            last_turn_id=first.turn_id,
            parts=[PlannedPartIn(name="main", words=80)],
            user_message="draft one",
        ),
        request=info,
        display=vocab.Display.FALLBACK,
    )
    attempt_b = await turn.run_plan(
        db_session,
        person_id=person.id,
        plan=PlanReplyIn(
            conversation_id=first.conversation_id,
            last_turn_id=first.turn_id,
            parts=[PlannedPartIn(name="main", words=80)],
            user_message="draft two edited",
        ),
        request=info,
        display=vocab.Display.FALLBACK,
    )
    assert attempt_b.turn_id != attempt_a.turn_id

    plan_call = await db_session.scalar(
        sa.select(models.Event).where(
            models.Event.turn_id == attempt_b.turn_id,
            models.Event.type == vocab.EventType.PLAN_CALL,
        )
    )
    assert plan_call is not None
    assert plan_call.meta.get("replay") is True
    assert plan_call.meta.get("replay_of_turn_id") == attempt_a.turn_id
    assert plan_call.meta.get("replay_attempt") == 2
