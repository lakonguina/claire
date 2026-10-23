"""Nightly replay job returns counters."""

from __future__ import annotations

import pytest

from app.jobs import nightly


@pytest.mark.asyncio
async def test_nightly_run_shape(db_session) -> None:
    report = await nightly.run(db_session)
    assert set(report) == {"people", "matched", "rewritten", "closed_turns"}
    assert report["matched"] + report["rewritten"] == report["people"]
    assert report["closed_turns"] >= 0
