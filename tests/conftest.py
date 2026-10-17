"""Fixtures: a fresh database per test and an MCP client for both eras."""

# Pytest injects fixtures by parameter name.
# pylint: disable=redefined-outer-name

import asyncio
import json
import os

os.environ.setdefault("DATABASE_URL", "postgresql://lf:lf@localhost:55432/lf")
os.environ.setdefault("ADMIN_TOKEN", "admin-secret")

# pylint: disable=wrong-import-position
import httpx
import pytest
import pytest_asyncio

from app import db
from app import main
from app import mcp_server
from app import params as params_lib
from app import service

# pylint: enable=wrong-import-position

UI_EXTENSION = "io.modelcontextprotocol/ui"
APPS_CAPS = {
    "extensions": {UI_EXTENSION: {"mimeTypes": ["text/html;profile=mcp-app"]}}
}
MODERN = "2026-07-28"
LEGACY = "2025-11-25"
META = "io.modelcontextprotocol/"
ACCEPT = "application/json, text/event-stream"


class McpClient:
    """A minimal JSON-RPC client over Streamable HTTP, for both eras."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        token: str,
        apps: bool,
        version: str = MODERN,
        host: str = "claude-ai",
    ):
        self.http, self.token, self.version = http, token, version
        self.url = f"/mcp/{token}"
        self.caps = APPS_CAPS if apps else {}
        self.info = {"name": host, "version": "1"}
        self.session: str | None = None
        self.next_id = 0

    async def user_id(self) -> int:
        return await service.user_for_token(self.token)

    async def rpc(self, method: str, params: dict | None = None) -> dict:
        """Sends one request and returns its JSON-RPC response."""
        params = dict(params or {})
        headers = {
            "accept": ACCEPT,
            "mcp-protocol-version": self.version,
            "mcp-method": method,
        }
        if self.version == MODERN:
            params["_meta"] = {
                f"{META}protocolVersion": MODERN,
                f"{META}clientCapabilities": self.caps,
                f"{META}clientInfo": self.info,
            }
        elif self.session is None:
            await self._initialize()
        if self.session:
            headers["mcp-session-id"] = self.session
        if method == "tools/call":
            headers["mcp-name"] = params["name"]
        self.next_id += 1
        response = await self.http.post(
            self.url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": self.next_id,
                "method": method,
                "params": params,
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    async def _initialize(self) -> None:
        headers = {"accept": ACCEPT}
        response = await self.http.post(
            self.url,
            headers=headers,
            json={
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": LEGACY,
                    "capabilities": self.caps,
                    "clientInfo": self.info,
                },
            },
        )
        self.session = response.headers["mcp-session-id"]
        headers["mcp-session-id"] = self.session
        await self.http.post(
            self.url,
            headers=headers,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        )

    async def call(self, name: str, **arguments) -> dict:
        """Calls a tool; plan_reply's JSON text is decoded into "data"."""
        result = await self.rpc(
            "tools/call", {"name": name, "arguments": arguments}
        )
        result = result["result"]
        if not result.get("isError") and name == "plan_reply":
            result["data"] = json.loads(result["content"][0]["text"])
        return result


@pytest_asyncio.fixture(scope="session")
async def server():
    """Opens the pool and runs the MCP session manager, once per process."""
    stop = asyncio.Event()

    async def run_manager():  # Its cancel scope must open and close here.
        async with mcp_server.mcp.session_manager.run():
            await stop.wait()

    async with await db.Connection.connect(
        os.environ["DATABASE_URL"], autocommit=True
    ) as conn:
        await conn.execute("DROP SCHEMA public CASCADE")
        await conn.execute("CREATE SCHEMA public")
    await db.migrate()
    await db.pool.open()
    task = asyncio.create_task(run_manager())
    await asyncio.sleep(0)
    yield
    stop.set()
    await task
    await db.pool.close()


@pytest_asyncio.fixture
async def pool(server):
    """Empties every table (keeping the schema) before each test."""
    del server
    async with db.pool.connection() as conn:
        tables = [
            r["tablename"]
            for r in await db.fetch(
                conn,
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' "
                "AND tablename <> 'schema_migrations'",
            )
        ]
        await db.execute(conn, "SET session_replication_role = replica")
        await db.execute(conn, f"TRUNCATE {", ".join(tables)} CASCADE")
        await db.execute(conn, "SET session_replication_role = DEFAULT")
    params_lib.cache.invalidate()
    return db.pool


@pytest_asyncio.fixture
async def http(pool):
    del pool
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost"
    ) as client:
        yield client


@pytest.fixture
def make_client(http):
    """Returns a factory of MCP clients, each with a new user."""

    async def make(
        apps: bool = True, version: str = MODERN, host: str = "claude-ai"
    ) -> McpClient:
        token = os.urandom(8).hex()
        async with db.pool.connection() as conn:
            await db.execute(
                conn,
                "INSERT INTO users (token_hash) VALUES (%s)",
                service.token_hash(token),
            )
        return McpClient(http, token, apps, version, host)

    return make
