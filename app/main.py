"""FastAPI app: MCP at /mcp (OAuth), pages, admin and metrics."""

import contextlib
import logging
import pathlib
from urllib import parse

import fastapi
from fastapi import staticfiles
from mcp.server import transport_security
import sqlalchemy as sa

from app.core import config
from app.core import db
from app import mcp_server
from app.routers import admin
from app.routers import dashboard
from app.routers import landing
from app.routers import oauth

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("access")
BEARER = "Bearer "
MCP = oauth.MCP_PATH
# The landing page stylesheet, built by site/ (Dockerfile).
SITE = pathlib.Path(__file__).resolve().parent / "static" / "site"
LOCAL_HOSTS = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "app:*"]


async def _unauthorized(send, invalid: bool) -> None:
    """Answers 401 with the challenge that starts the OAuth flow."""
    challenge = oauth.challenge(invalid).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"text/plain"),
                (b"www-authenticate", challenge),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b"Unauthorized"})


class McpAuth:
    """Serves /mcp to the user behind an OAuth access token (SPEC 4.7).

    Every other path goes to the API. The user id travels in a header that
    only this wrapper sets: any copy sent by the client is dropped.
    """

    def __init__(self, mcp, other):
        self.mcp, self.other = mcp, other

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"].rstrip("/") != MCP:
            return await self.other(scope, receive, send)
        bearer = dict(scope["headers"]).get(b"authorization", b"").decode()
        access = (
            bearer.removeprefix(BEARER) if bearer.startswith(BEARER) else ""
        )
        user_id = await oauth.user_for_access_token(access) if access else None
        if user_id is None:
            return await _unauthorized(send, invalid=bool(access))
        header = mcp_server.TOKEN_HEADER.encode()
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != header]
        headers.append((header, str(user_id).encode()))
        await self.mcp(
            dict(scope, path="/", raw_path=b"/", headers=headers), receive, send
        )


class AccessLog:
    """Access log without the query string, which may carry a credential."""

    def __init__(self, inner):
        self.app = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        status = {}

        async def capture(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        await self.app(scope, receive, capture)
        log.info(
            "%s %s %s",
            scope["method"],
            scope["path"],
            status.get("code"),
        )


@contextlib.asynccontextmanager
async def lifespan(_):
    await db.migrate()
    async with mcp_server.mcp.session_manager.run():
        yield
    await db.engine.dispose()


settings = config.settings
mcp_app = mcp_server.mcp.streamable_http_app(
    streamable_http_path="/",
    json_response=True,
    transport_security=transport_security.TransportSecuritySettings(
        allowed_hosts=[
            parse.urlparse(settings.public_url).netloc,
            *LOCAL_HOSTS,
        ],
        allowed_origins=[
            settings.public_url,
            *settings.allowed_origins,
            "http://localhost:*",
        ],
    ),
)
api = fastapi.FastAPI(
    title="Claire",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
api.include_router(admin.router)
api.include_router(dashboard.page)
api.include_router(dashboard.api)
api.include_router(oauth.router)
api.include_router(landing.router)
api.mount(
    "/static/site",
    staticfiles.StaticFiles(directory=SITE, check_dir=False),
)
api.mount("/assets", staticfiles.StaticFiles(directory=landing.ASSETS))
api.router.routes.extend(oauth.routes())
app = AccessLog(McpAuth(mcp_app, api))


@api.get("/health")
async def health() -> dict:
    async with db.session() as session:
        await session.execute(sa.select(1))
    return {"ok": True}
