"""PostgreSQL access: one async pool, migrations applied at start."""

from pathlib import Path

from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from app.config import settings

MIGRATIONS = Path(__file__).resolve().parent.parent / "migrations"
pool = AsyncConnectionPool(settings.database_url, open=False, kwargs={"row_factory": dict_row, "autocommit": False})

__all__ = ["AsyncConnection", "Jsonb", "pool", "migrate", "fetch", "fetchrow", "execute"]


async def migrate() -> None:
    async with pool.connection() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY)")
        done = {r["name"] for r in await (await conn.execute("SELECT name FROM schema_migrations")).fetchall()}
        for path in sorted(MIGRATIONS.glob("*.sql")):
            if path.name not in done:
                await conn.execute(path.read_text())
                await conn.execute("INSERT INTO schema_migrations VALUES (%s)", (path.name,))


async def fetch(conn: AsyncConnection, sql: str, *args) -> list[dict]:
    return await (await conn.execute(sql, args)).fetchall()


async def fetchrow(conn: AsyncConnection, sql: str, *args) -> dict | None:
    return await (await conn.execute(sql, args)).fetchone()


async def execute(conn: AsyncConnection, sql: str, *args) -> None:
    await conn.execute(sql, args)
