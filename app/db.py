"""Database access: the async SQLAlchemy engine, sessions and migrations."""

import asyncio
import pathlib

from alembic import command
from alembic import config as alembic_config
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config

ROOT = pathlib.Path(__file__).resolve().parent.parent
DRIVER = "postgresql+psycopg"


def url() -> str:
    """Returns the SQLAlchemy URL of the configured database."""
    database_url = config.settings.database_url
    return DRIVER + database_url[database_url.index(":") :]


engine = sa_asyncio.create_async_engine(url(), pool_pre_ping=True)
session = sa_asyncio.async_sessionmaker(engine, expire_on_commit=False)


def _alembic() -> alembic_config.Config:
    cfg = alembic_config.Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url())
    return cfg


async def migrate() -> None:
    """Upgrades the schema to the latest Alembic revision.

    Alembic's async environment runs its own event loop, hence the thread.
    """
    await asyncio.to_thread(command.upgrade, _alembic(), "head")
