"""Alembic environment: async migrations of the app's models."""

import asyncio
from logging import config as logging_config

from alembic import context
import pgvector.sqlalchemy
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import db
from app import models

# The CLI configures logging; the app, which migrates at start, keeps its own.
if context.config.config_file_name and context.config.attributes.get(
    "configure_logger", True
):
    logging_config.fileConfig(
        context.config.config_file_name, disable_existing_loggers=False
    )


def _render_item(kind, item, autogen_context):
    """Renders pgvector and the app's own column types in revisions."""
    if kind == "type" and isinstance(item, pgvector.sqlalchemy.Vector):
        autogen_context.imports.add("import pgvector.sqlalchemy")
        return f"pgvector.sqlalchemy.Vector({item.dim})"
    if kind == "type" and isinstance(item, models.FloatArray):
        return "postgresql.ARRAY(sa.Float())"
    if kind == "type" and isinstance(
        item, (models.EncryptedText, models.EncryptedJson)
    ):
        # The ciphertext's column; the app encrypts and decrypts it.
        return "postgresql.BYTEA()"
    return False


def _configure(connection=None) -> None:
    context.configure(
        connection=connection,
        url=None if connection else db.url(),
        target_metadata=models.Base.metadata,
        render_item=_render_item,
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
