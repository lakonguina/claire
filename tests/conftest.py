import asyncio
import json
import os

os.environ.setdefault("DATABASE_URL", "postgresql://lf:lf@localhost:55432/lf")
os.environ.setdefault("ADMIN_TOKEN", "admin-secret")

import httpx
import pytest
import pytest_asyncio

from app import params, service
from app.db import execute, migrate, pool
from app.main import app
from app.mcp_server import mcp

APPS_CAPS = {"extensions": {"io.modelcontextprotocol/ui": {"mimeTypes": ["text/html;profile=mcp-app"]}}}
MODERN = "2026-07-28"
LEGACY = "2025-11-25"


class McpClient:
    """Minimal JSON-RPC client over Streamable HTTP, for both protocol eras."""

    def __init__(self, http: httpx.AsyncClient, token: str, apps: bool, version: str = MODERN, host: str = "claude-ai"):
        self.http, self.url, self.version = http, f"/mcp/{token}", version
        self.caps, self.info = (APPS_CAPS if apps else {}), {"name": host, "version": "1"}
        self.session: str | None = None
        self.next_id = 0

    async def rpc(self, method: str, params: dict | None = None) -> dict:
        params = dict(params or {})
        headers = {"accept": "application/json, text/event-stream", "mcp-protocol-version": self.version,
                   "mcp-method": method}
        if self.version == MODERN:
            params["_meta"] = {"io.modelcontextprotocol/protocolVersion": MODERN,
                               "io.modelcontextprotocol/clientCapabilities": self.caps,
                               "io.modelcontextprotocol/clientInfo": self.info}
        elif self.session is None:
            await self._initialize()
        if self.session:
            headers["mcp-session-id"] = self.session
        if method == "tools/call":
            headers["mcp-name"] = params["name"]
        self.next_id += 1
        r = await self.http.post(self.url, headers=headers, json={"jsonrpc": "2.0", "id": self.next_id,
                                                                  "method": method, "params": params})
        assert r.status_code == 200, r.text
        return r.json()

    async def _initialize(self) -> None:
        headers = {"accept": "application/json, text/event-stream"}
        r = await self.http.post(self.url, headers=headers, json={
            "jsonrpc": "2.0", "id": 0, "method": "initialize",
            "params": {"protocolVersion": LEGACY, "capabilities": self.caps, "clientInfo": self.info}})
        self.session = r.headers["mcp-session-id"]
        headers["mcp-session-id"] = self.session
        await self.http.post(self.url, headers=headers, json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    async def call(self, name: str, **arguments) -> dict:
        """Tool result; plan_reply's JSON text is decoded."""
        result = (await self.rpc("tools/call", {"name": name, "arguments": arguments}))["result"]
        if not result.get("isError") and name == "plan_reply":
            result["data"] = json.loads(result["content"][0]["text"])
        return result


@pytest_asyncio.fixture(scope="session")
async def server():
    """The pool and the MCP session manager can each be started only once per process."""
    await pool.open()
    stop = asyncio.Event()

    async def run_manager():  # the manager's cancel scope must open and close in one task
        async with mcp.session_manager.run():
            await stop.wait()

    task = asyncio.create_task(run_manager())
    await asyncio.sleep(0)
    yield
    stop.set()
    await task
    await pool.close()


@pytest_asyncio.fixture
async def db(server):
    async with pool.connection() as conn:
        await execute(conn, "DROP SCHEMA public CASCADE")
        await execute(conn, "CREATE SCHEMA public")
    await migrate()
    params.invalidate()
    return pool


@pytest_asyncio.fixture
async def http(db):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        yield client


async def new_token(db) -> str:
    token = os.urandom(8).hex()
    async with db.connection() as conn:
        await execute(conn, "INSERT INTO users (token_hash) VALUES (%s)", service.token_hash(token))
    return token


@pytest.fixture
def make_client(http, db):
    async def make(apps: bool = True, version: str = MODERN, token: str | None = None, host: str = "claude-ai"):
        return McpClient(http, token or await new_token(db), apps, version, host)
    return make
