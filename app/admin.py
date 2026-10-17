"""Create a user and print their connector URL: python -m app.admin new-user"""

import asyncio
import secrets
import sys

from app.config import settings
from app.db import execute, migrate, pool
from app.service import token_hash


async def new_user() -> str:
    token = secrets.token_urlsafe(24)
    await pool.open()
    await migrate()
    async with pool.connection() as conn:
        await execute(conn, "INSERT INTO users (token_hash) VALUES (%s)", token_hash(token))
    await pool.close()
    return f"{settings.public_url.rstrip('/')}/mcp/{token}"


if __name__ == "__main__":
    if sys.argv[1:] != ["new-user"]:
        sys.exit(__doc__)
    print(asyncio.run(new_user()))
