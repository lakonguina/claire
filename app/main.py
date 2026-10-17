"""FastAPI app: MCP at /mcp/<token>, profile page, admin and metrics."""

import contextlib
import logging
import re
from urllib import parse

import fastapi
from mcp.server import transport_security
import pydantic

from app import config
from app import dashboard
from app import db
from app import mcp_server
from app import metrics
from app import profile
from app import service
from app import vocab

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("access")
REDACT = re.compile(r"^/mcp/[^/]+")
REDACTED = "/mcp/<redacted>"
LOCAL_HOSTS = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "app:*"]


async def _send_text(send, status: int, body: bytes) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"text/plain")],
        }
    )
    await send({"type": "http.response.body", "body": body})


class McpAuth:
    """Resolves /mcp/<token> to a user and serves the MCP app.

    The user id travels in a header that only this wrapper sets: any copy
    sent by the client is dropped.
    """

    def __init__(self, inner):
        self.app = inner
        self.users: dict[str, int] = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        root = scope.get("root_path", "")
        token = scope["path"].removeprefix(root).strip("/").split("/")[0]
        user_id = self.users.get(token) or (
            await service.user_for_token(token) if token else None
        )
        if user_id is None:
            log.info(
                "%s %s rejected: unknown connector token",
                scope["method"],
                REDACTED,
            )
            return await _send_text(send, 404, b"Not found")
        self.users[token] = user_id
        header = mcp_server.TOKEN_HEADER.encode()
        headers = [(k, v) for k, v in scope["headers"] if k.lower() != header]
        headers.append((header, str(user_id).encode()))
        path = root + "/"
        await self.app(
            dict(scope, path=path, raw_path=path.encode(), headers=headers),
            receive,
            send,
        )


class AccessLog:
    """Access log with the token redacted and no query string."""

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
            REDACT.sub(REDACTED, scope["path"]),
            status.get("code"),
        )


@contextlib.asynccontextmanager
async def lifespan(_):
    await db.migrate()
    await db.pool.open()
    async with mcp_server.mcp.session_manager.run():
        yield
    await db.pool.close()


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
    title="Lingua Franca",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
api.include_router(dashboard.page)
api.include_router(dashboard.api)
api.include_router(profile.router)
api.mount("/mcp", McpAuth(mcp_app))
app = AccessLog(api)
admin = [fastapi.Depends(dashboard.admin)]


class GoldLabel(pydantic.BaseModel):
    turn_id: str = pydantic.Field(pattern=rf"^{service.TURN_PREFIX}\d+$")
    outcome: vocab.Outcome


@api.get("/health")
async def health() -> dict:
    async with db.pool.connection() as conn:
        await db.execute(conn, "SELECT 1")
    return {"ok": True}


@api.post("/gold_label", dependencies=admin)
async def gold_label(label: GoldLabel) -> dict:
    async with db.pool.connection() as conn:
        await db.execute(
            conn,
            "INSERT INTO gold_labels (turn_id, outcome) VALUES (%s, %s)",
            int(label.turn_id.removeprefix(service.TURN_PREFIX)),
            label.outcome,
        )
    return {"ok": True}


@api.get("/metrics", dependencies=admin)
async def get_metrics() -> dict:
    async with db.pool.connection() as conn:
        return await metrics.compute(conn)
