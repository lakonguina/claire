"""Shared fixtures. DB tests skip when Postgres is unreachable."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text

# 32 zero bytes, base64 — enough for AES tests without reading .env.
os.environ.setdefault(
    "DATA_KEY",
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
)


@pytest_asyncio.fixture
async def db_session():
    """Yield a session; skip the test if Postgres is down."""
    from app.core import db

    try:
        async with db.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Postgres unavailable: {exc}")

    async with db.Session() as session:
        async with session.begin():
            yield session
            await session.rollback()
