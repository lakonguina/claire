"""Alembic environment: async migrations."""

from __future__ import annotations

import asyncio
from logging import config as logging_config

from alembic import context
from sqlalchemy.ext import asyncio as sa_asyncio

from app import models
from app.core import db

if context.config.config_file_name and context.config.attributes.get(
    "configure_logger", True
):
    logging_config.fileConfig(
        context.config.config_file_name, disable_existing_loggers=False
    )


def _configure(connection=None) -> None:
    context.configure(
        connection=connection,
        url=None if connection else db.url(),
        target_metadata=models.Base.metadata,
        compare_type=True,
    )


def _run(connection) -> None:
    _configure(connection)
    with context.begin_transaction():
        context.run_migrations()


async def _run_online() -> None:
    engine = sa_asyncio.create_async_engine(db.url())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
    await engine.dispose()


if context.is_offline_mode():
    _configure()
    with context.begin_transaction():
        context.run_migrations()
else:
    asyncio.run(_run_online())
