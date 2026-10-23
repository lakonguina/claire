"""FastAPI entrypoint: landing site + MCP connector.

On startup we verify the encryption key and migrate the database, then
hand control to the MCP session manager for Streamable HTTP.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import auth as oauth
from app.admin import router as admin_router
from app.api.router import router as api_router
from app.core import crypto, db
from app.mcp.server import mcp, mcp_app

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "static"
_TEMPLATES = Jinja2Templates(directory=_ROOT / "templates")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run process-wide setup, then keep the MCP session manager alive.

    Args:
        _app: FastAPI instance (unused; required by the framework).

    Yields:
        Control while the application serves requests.
    """
    crypto.check_key()
    await db.migrate()
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="claire.pm", lifespan=lifespan)
app.include_router(api_router)
app.include_router(admin_router)
app.include_router(oauth.router)
app.router.routes.extend(oauth.routes())
app.mount("/static", StaticFiles(directory=_STATIC), name="static")
app.mount(oauth.MCP_PATH, mcp_app)


@app.exception_handler(404)
async def not_found(request: Request, _exc: Exception):
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="404.html",
        context={},
        status_code=404,
    )
