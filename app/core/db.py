"""Database access: async engine, sessions, and schema migrations.

Postgres is Claire's long-term memory. SQLAlchemy talks to it; Alembic
applies schema changes so every environment (laptop, compose, prod) ends
up with the same tables.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from alembic import command
from alembic import config as alembic_config
from sqlalchemy import text
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config

ROOT = Path(__file__).resolve().parents[2]
# Async driver; ``DATABASE_URL`` stays the classic ``postgresql://…`` form.
DRIVER = "postgresql+psycopg"


def url() -> str:
    """Return the SQLAlchemy URL derived from ``DATABASE_URL``.

    Returns:
        A URL such as ``postgresql+psycopg://user:pass@host/db``.
    """
    database_url = config.settings.database_url
    return DRIVER + database_url[database_url.index(":") :]


engine = sa_asyncio.create_async_engine(url(), pool_pre_ping=True)
Session = sa_asyncio.async_sessionmaker(engine, expire_on_commit=False)


def _alembic() -> alembic_config.Config:
    """Build an Alembic config pointed at this repo's migrations."""
    cfg = alembic_config.Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url())
    cfg.attributes["configure_logger"] = False
    return cfg


async def migrate() -> None:
    """Apply pending Alembic revisions up to ``head``.

    Alembic's own loop runs in a worker thread so it does not nest with
    the FastAPI event loop.
    """
    await asyncio.to_thread(command.upgrade, _alembic(), "head")


async def ping() -> None:
    """Cheap liveness check used by ``/health``.

    Raises:
        Exception: Propagates driver errors if Postgres is unreachable.
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
