"""FastAPI app: MCP mounted at /mcp/<token>, plus health, gold labels, metrics and the admin dashboard."""

import contextlib
import logging
import re
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from app import dashboard, metrics, service
from app.config import settings
from app.db import execute, migrate, pool
from app.mcp_server import TOKEN_HEADER, mcp
from app.model.reaction import OUTCOMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("access")
REDACT = re.compile(r"^/mcp/[^/]+")


class McpAuth:
    """Resolve /mcp/<token> to a user, pass its id in a header only this wrapper sets, and serve the MCP app."""

    def __init__(self, app):
        self.app = app
        self.users: dict[str, int] = {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        root = scope.get("root_path", "")
        token = scope["path"].removeprefix(root).strip("/").split("/")[0]
        user_id = self.users.get(token) or (await service.user_for_token(token) if token else None)
        if user_id is None:
            await send({"type": "http.response.start", "status": 404, "headers": [(b"content-type", b"text/plain")]})
            return await send({"type": "http.response.body", "body": b"Not found"})
        self.users[token] = user_id
        headers = [(k, v) for k, v in scope["headers"] if k.decode().lower() != TOKEN_HEADER]
        headers.append((TOKEN_HEADER.encode(), str(user_id).encode()))
        await self.app(dict(scope, path=root + "/", raw_path=(root + "/").encode(), headers=headers), receive, send)


class AccessLog:
    """Access log with the token redacted and no query string (uvicorn's own access log is disabled)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        status = {}

        async def capture(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        await self.app(scope, receive, capture)
        log.info("%s %s %s", scope["method"], REDACT.sub("/mcp/<redacted>", scope["path"]), status.get("code"))


@contextlib.asynccontextmanager
async def lifespan(_app):
    await pool.open()
    await migrate()
    async with mcp.session_manager.run():
        yield
    await pool.close()


public_host = urlparse(settings.public_url).netloc
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    json_response=True,
    transport_security=TransportSecuritySettings(
        allowed_hosts=[public_host, "localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*", "app:*"],
        allowed_origins=[settings.public_url, "https://claude.ai", "https://chatgpt.com", "http://localhost:*"],
    ),
)
api = FastAPI(title="Lingua Franca", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
api.include_router(dashboard.page)
api.include_router(dashboard.api)
api.mount("/mcp", McpAuth(mcp_app))
app = AccessLog(api)


class GoldLabel(BaseModel):
    turn_id: str
    outcome: str


@api.get("/health")
async def health():
    async with pool.connection() as conn:
        await execute(conn, "SELECT 1")
    return {"ok": True}


@api.post("/gold_label", dependencies=[Depends(dashboard.admin)])
async def gold_label(label: GoldLabel):
    if label.outcome not in OUTCOMES or not label.turn_id.removeprefix("t_").isdigit():
        raise HTTPException(status_code=422, detail="Invalid turn_id or outcome")
    async with pool.connection() as conn:
        await execute(conn, "INSERT INTO gold_labels (turn_id, outcome) VALUES (%s, %s)",
                      int(label.turn_id.removeprefix("t_")), label.outcome)
    return {"ok": True}


@api.get("/metrics", dependencies=[Depends(dashboard.admin)])
async def get_metrics():
    async with pool.connection() as conn:
        return await metrics.compute(conn)
