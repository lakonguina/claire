"""Creates a user and prints their connector URL.

Usage: python -m app.admin new-user
"""

import asyncio
import secrets
import sys

from app import config
from app import db
from app import service

COMMAND = "new-user"
TOKEN_BYTES = 24


async def new_user() -> str:
    """Creates a user; returns the connector URL holding their token."""
    token = secrets.token_urlsafe(TOKEN_BYTES)
    await db.migrate()
    await db.pool.open()
    async with db.pool.connection() as conn:
        await db.execute(
            conn,
            "INSERT INTO users (token_hash) VALUES (%s)",
            service.token_hash(token),
        )
    await db.pool.close()
    return f"{config.settings.public_url.rstrip("/")}/mcp/{token}"


if __name__ == "__main__":
    if sys.argv[1:] != [COMMAND]:
        sys.exit(__doc__)
    print(asyncio.run(new_user()))
