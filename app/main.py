"""FastAPI app: MCP mounted at /mcp/<token>, plus health, gold labels, debug page and metrics."""

import contextlib
import html
import logging
import re
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

from app import metrics, service
from app.config import settings
from app.db import execute, fetch, migrate, pool
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
api.mount("/mcp", McpAuth(mcp_app))
app = AccessLog(api)


def admin(request: Request) -> None:
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ") or request.query_params.get("key")
    if not settings.admin_token or supplied != settings.admin_token:
        raise HTTPException(status_code=401)


class GoldLabel(BaseModel):
    turn_id: str
    outcome: str


@api.get("/health")
async def health():
    async with pool.connection() as conn:
        await execute(conn, "SELECT 1")
    return {"ok": True}


@api.post("/gold_label", dependencies=[Depends(admin)])
async def gold_label(label: GoldLabel):
    if label.outcome not in OUTCOMES or not label.turn_id.removeprefix("t_").isdigit():
        raise HTTPException(status_code=422, detail="Invalid turn_id or outcome")
    async with pool.connection() as conn:
        await execute(conn, "INSERT INTO gold_labels (turn_id, outcome) VALUES (%s, %s)",
                      int(label.turn_id.removeprefix("t_")), label.outcome)
    return {"ok": True}


@api.get("/metrics", dependencies=[Depends(admin)])
async def get_metrics():
    async with pool.connection() as conn:
        return await metrics.compute(conn)


@api.get("/debug", dependencies=[Depends(admin)], response_class=HTMLResponse)
async def debug(request: Request):
    async with pool.connection() as conn:
        rows = await fetch(conn, """
            SELECT t.id, t.topic, o.user_quote, o.observed, i.reaction_probs,
                   (SELECT outcome FROM gold_labels g WHERE g.turn_id = t.id ORDER BY created_at DESC LIMIT 1) AS gold
            FROM turns t JOIN turn_inference i ON i.turn_id = t.id
            LEFT JOIN outcomes o ON o.turn_id = t.id AND o.source = 'assistant'
            ORDER BY t.id DESC LIMIT 100""")
    options = "".join(f"<option>{o}</option>" for o in OUTCOMES)
    body = "".join(
        f"<tr><td>t_{r['id']}</td><td>{html.escape(r['topic'])}</td><td>{html.escape(r['user_quote'] or '')}</td>"
        f"<td>{r['observed'] or ''}</td><td>{max(r['reaction_probs'], key=r['reaction_probs'].get)}</td>"
        f"<td>{r['gold'] or ''}</td><td><select id='g{r['id']}'>{options}</select>"
        f"<button onclick=\"label({r['id']})\">Save</button></td></tr>"
        for r in rows)
    key = html.escape(request.query_params.get("key", ""))
    return f"""<!doctype html><meta charset="utf-8"><title>Lingua Franca turns</title>
<style>body{{font-family:system-ui;margin:16px}}td,th{{padding:4px 8px;border-bottom:1px solid #ddd}}</style>
<table><tr><th>Turn</th><th>Topic</th><th>Quote</th><th>Assistant</th><th>Inferred</th><th>Gold</th><th></th></tr>{body}</table>
<script>
async function label(id) {{
  const outcome = document.getElementById("g" + id).value;
  const r = await fetch("/gold_label", {{method: "POST", headers: {{"content-type": "application/json",
    authorization: "Bearer {key}"}}, body: JSON.stringify({{turn_id: "t_" + id, outcome}})}});
  if (r.ok) location.reload();
}}
</script>"""
