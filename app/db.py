"""PostgreSQL access: migrations, one async pool, small query helpers."""

import pathlib
from typing import Any

import pgvector.psycopg
import psycopg
from psycopg import rows
from psycopg.types import json as psycopg_json
import psycopg_pool

from app import config

MIGRATIONS = pathlib.Path(__file__).resolve().parent.parent / "migrations"
Connection = psycopg.AsyncConnection
Jsonb = psycopg_json.Jsonb


async def _configure(conn: Connection) -> None:
    await pgvector.psycopg.register_vector_async(conn)


pool = psycopg_pool.AsyncConnectionPool(
    config.settings.database_url,
    open=False,
    configure=_configure,
    kwargs={"row_factory": rows.dict_row, "autocommit": False},
)


async def migrate() -> None:
    """Applies the migrations not applied yet, on a dedicated connection.

    The pool's connections register the vector type, which exists only once
    the migrations have run.
    """
    async with await Connection.connect(config.settings.database_url) as conn:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations "
            "(name text PRIMARY KEY)"
        )
        cursor = await conn.execute("SELECT name FROM schema_migrations")
        done = {name for (name,) in await cursor.fetchall()}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name not in done:
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations VALUES (%s)", (path.name,)
                )


async def fetch(conn: Connection, sql: str, *args: Any) -> list[dict]:
    """Returns every row of a query."""
    return await (await conn.execute(sql, args)).fetchall()


async def fetchrow(conn: Connection, sql: str, *args: Any) -> dict | None:
    """Returns the first row of a query, or None."""
    return await (await conn.execute(sql, args)).fetchone()


async def execute(conn: Connection, sql: str, *args: Any) -> None:
    """Runs a statement."""
    await conn.execute(sql, args)
