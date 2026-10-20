"""Fixtures: a fresh database per test and an MCP client for both eras."""

# Pytest injects fixtures by parameter name.
# pylint: disable=redefined-outer-name

import asyncio
import base64
import hashlib
import json
import os
import secrets
from urllib import parse

os.environ.setdefault(
    "DATABASE_URL", "postgresql://claire:claire@localhost:55432/claire"
)
os.environ.setdefault("ADMIN_TOKEN", "admin-secret")
# Random draws the tests decide themselves, through the settings.
for name in ("HOLDOUT_SHARE", "ABLATION_SHARE", "MESSAGE_SAMPLE"):
    os.environ.setdefault(name, "0")
os.environ.setdefault(
    "DATA_KEY", base64.b64encode(b"test-key-of-thirty-two-bytes!!!!").decode()
)

# pylint: disable=wrong-import-position
import httpx
import pytest
import pytest_asyncio
import sqlalchemy as sa

from app.core import db
from app import main
from app import mcp_server
from app import models
from app.routers import oauth
from app.services import params as params_lib

# pylint: enable=wrong-import-position

UI_EXTENSION = "io.modelcontextprotocol/ui"
APPS_CAPS = {
    "extensions": {UI_EXTENSION: {"mimeTypes": ["text/html;profile=mcp-app"]}}
}
MODERN = "2026-07-28"
LEGACY = "2025-11-25"
META = "io.modelcontextprotocol/"
ACCEPT = "application/json, text/event-stream"
CALLBACK = "https://claude.ai/api/mcp/auth_callback"
FORM = {"content-type": "application/x-www-form-urlencoded"}
STATE = "s1"
# Requests whose target travels in the mcp-name header (2026-07-28).
NAMED = {"tools/call": "name", "resources/read": "uri"}


class McpClient:
    """A minimal JSON-RPC client over Streamable HTTP, for both eras."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        access: str,
        apps: bool,
        version: str = MODERN,
        host: str = "claude-ai",
    ):
        """Calls /mcp with an OAuth access token."""
        self.http, self.access, self.version = http, access, version
        self.url = oauth.MCP_PATH
        self.auth = {"authorization": f"Bearer {access}"}
        self.caps = APPS_CAPS if apps else {}
        self.info = {"name": host, "version": "1"}
        self.session: str | None = None
        self.next_id = 0

    async def user_id(self) -> int:
        return await oauth.user_for_access_token(self.access)

    async def rpc(self, method: str, params: dict | None = None) -> dict:
        """Sends one request and returns its JSON-RPC response."""
        params = dict(params or {})
        headers = {
            **self.auth,
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
        if method in NAMED:
            headers["mcp-name"] = params[NAMED[method]]
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
        headers = {**self.auth, "accept": ACCEPT}
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


async def register(http: httpx.AsyncClient) -> str:
    """Registers an app as Claude does; returns its client id."""
    response = await http.post(
        "/register",
        json={
            "client_name": "Claude",
            "redirect_uris": [CALLBACK],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["client_id"]


def query(location: str) -> dict:
    return dict(parse.parse_qsl(parse.urlparse(location).query))


async def consent_url(http: httpx.AsyncClient, client_id: str, challenge: str):
    """Starts an authorization; returns the local URL of the consent page."""
    response = await http.get(
        "/authorize",
        params={
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": CALLBACK,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": STATE,
            "resource": oauth.RESOURCE,
        },
    )
    assert response.status_code == 302, response.text
    url = parse.urlparse(response.headers["location"])
    return f"{url.path}?{url.query}"


async def authorize(http: httpx.AsyncClient) -> tuple[str, str, str]:
    """Connects a new app up to its code, clicking Continue.

    Returns:
        The client id, the authorization code and the PKCE verifier.
    """
    client_id = await register(http)
    verifier = secrets.token_urlsafe(48)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    consent = await consent_url(http, client_id, challenge)
    allowed = await http.post(
        oauth.CONSENT_PATH,
        content=parse.urlencode(
            {
                "request": query(consent)["request"],
                "action": oauth.ALLOW,
            }
        ),
        headers=FORM,
    )
    assert allowed.status_code == 303
    back = query(allowed.headers["location"])
    assert back["state"] == STATE
    return client_id, back["code"], verifier


async def token(http: httpx.AsyncClient, client_id: str, **form):
    """Calls the token endpoint; returns the status and the JSON body."""
    response = await http.post(
        "/token",
        content=parse.urlencode({"client_id": client_id, **form}),
        headers=FORM,
    )
    return response.status_code, response.json()


async def redeem(http, client_id: str, code: str, verifier: str):
    """Exchanges an authorization code for tokens."""
    return await token(
        http,
        client_id,
        grant_type="authorization_code",
        code=code,
        redirect_uri=CALLBACK,
        code_verifier=verifier,
        resource=oauth.RESOURCE,
    )


@pytest_asyncio.fixture(scope="session")
async def server():
    """Migrates a fresh schema and runs the MCP session manager, once."""
    stop = asyncio.Event()

    async def run_manager():  # Its cancel scope must open and close here.
        async with mcp_server.mcp.session_manager.run():
            await stop.wait()

    async with db.engine.begin() as conn:
        await conn.execute(sa.text("DROP SCHEMA public CASCADE"))
        await conn.execute(sa.text("CREATE SCHEMA public"))
    await db.engine.dispose()
    await db.migrate()
    task = asyncio.create_task(run_manager())
    await asyncio.sleep(0)
    yield
    stop.set()
    await task
    await db.engine.dispose()


@pytest_asyncio.fixture
async def database(server):
    """Empties every table (keeping the schema) before each test."""
    del server
    tables = ", ".join(t.name for t in models.Base.metadata.sorted_tables)
    async with db.engine.begin() as conn:
        await conn.execute(sa.text(f"TRUNCATE {tables} RESTART IDENTITY"))
    params_lib.cache.invalidate()
    return db.session


@pytest_asyncio.fixture
async def http(database):
    del database
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://localhost"
    ) as client:
        yield client


@pytest.fixture
def make_client(http):
    """Returns a factory of MCP clients, each a new connector and user."""

    async def make(
        apps: bool = True,
        version: str = MODERN,
        host: str = "claude-ai",
    ) -> McpClient:
        _, tokens = await redeem(http, *await authorize(http))
        return McpClient(http, tokens["access_token"], apps, version, host)

    return make
