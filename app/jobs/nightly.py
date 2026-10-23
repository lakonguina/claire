"""Nightly job: rebuild person bandit caches from the event log.

Run: ``python -m app.jobs.nightly``.
Not on the MCP request path — corrective batch only.
"""

from __future__ import annotations

import asyncio
import logging

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from app import models
from app.core import db
from app.core import vocab
from app.model import replay

log = logging.getLogger(__name__)


async def replay_bandits(session: AsyncSession) -> tuple[int, int, int]:
    """Replay events into person caches; rewrite rows that drifted.

    Returns:
        ``(matched, rewritten, closed_turns)``.
    """
    events = list(
        await session.scalars(
            sa.select(models.Event).order_by(models.Event.id.asc())
        )
    )
    closed_turns = sum(
        1 for event in events if event.type == vocab.EventType.TURN_CLOSED
    )
    rebuilt = replay.rebuild_bandits(events)
    people = (
        await session.scalars(sa.select(models.Person).order_by(models.Person.id))
    ).all()
    matched = rewritten = 0
    for person in people:
        state = rebuilt.get(person.id) or replay.BanditState()
        if replay.counts_match(person.form_counts, state.form_counts) and (
            replay.counts_match(person.words_counts, state.words_counts)
        ):
            matched += 1
            continue
        person.form_counts = state.form_counts
        person.words_counts = state.words_counts
        rewritten += 1
        log.warning("nightly replay corrected person_id=%s", person.id)
    return matched, rewritten, closed_turns


async def run(session: AsyncSession) -> dict[str, int]:
    """Replay bandits for every person; return counters."""
    people_count = int(
        await session.scalar(sa.select(sa.func.count(models.Person.id))) or 0
    )
    matched, rewritten, closed_turns = await replay_bandits(session)
    return {
        "people": people_count,
        "matched": matched,
        "rewritten": rewritten,
        "closed_turns": closed_turns,
    }


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    async with db.Session() as session, session.begin():
        report = await run(session)
    log.info(
        "nightly done: people=%d matched=%d rewritten=%d closed=%d",
        report["people"],
        report["matched"],
        report["rewritten"],
        report["closed_turns"],
    )


if __name__ == "__main__":
    asyncio.run(main())
