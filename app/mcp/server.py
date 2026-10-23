"""MCP server: tools + App UI resource. Logic lives in :mod:`app.mcp.turn`."""

from __future__ import annotations

import hashlib
from pathlib import Path

from mcp.server import MCPServer
from mcp.server import apps as mcp_apps
from mcp.server.auth.provider import ProviderTokenVerifier
from mcp.server.mcpserver.context import Context
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from app.auth import SETTINGS, provider
from app.core import db
from app.mcp import turn
from app.mcp.schemas import (
    PlanReplyIn,
    PlanReplyOut,
    RecordEventIn,
    ShowReplyIn,
)

_WIDGET = Path(__file__).resolve().parents[1] / "static" / "reply.html"
_WIDGET_HTML = _WIDGET.read_text(encoding="utf-8") if _WIDGET.exists() else ""
_UI_PREFIX = "ui://claire/"
UI_URI = (
    f"{_UI_PREFIX}reply-"
    f"{hashlib.sha256(_WIDGET_HTML.encode()).hexdigest()[:12]}.html"
)
_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    open_world_hint=False,
)

# What the model reads back: never the reply itself, or it writes it again.
_SHOWN = [
    TextContent(
        type="text",
        text="The user now sees your reply in the widget. Write nothing else.",
    )
]

apps = mcp_apps.Apps()


@apps.tool(resource_uri=UI_URI, annotations=_ANNOTATIONS)
async def claire_show_reply(ctx: Context, show: ShowReplyIn) -> CallToolResult:
    """Write your entire reply here, in Markdown, and nothing in the chat.

    Call exactly once per reply, after claire_plan_reply.
    """
    mode = "app" if mcp_apps.client_supports_apps(ctx) else "fallback"
    async with db.Session() as session, session.begin():
        person_id, info = await turn.bind_caller(session, ctx)
        shown = await turn.run_show(
            session,
            person_id=person_id,
            show=show,
            request=info,
            mode=mode,
        )
    return CallToolResult(
        content=_SHOWN, structured_content=shown.model_dump(mode="json")
    )


@apps.tool(
    resource_uri=UI_URI,
    visibility=["app"],
    annotations=_ANNOTATIONS,
)
async def claire_record_event(ctx: Context, event: RecordEventIn) -> dict:
    """Log a widget event (``reply_view``, ``render_error``, …)."""
    async with db.Session() as session, session.begin():
        person_id, info = await turn.bind_caller(session, ctx)
        await turn.record_widget(
            session,
            person_id=person_id,
            turn_id=event.turn_id,
            kind=event.kind,
            value=event.value,
            request=info,
        )
    return {"ok": True}


apps.add_html_resource(
    UI_URI,
    _WIDGET_HTML,
    name="reply",
    permissions=mcp_apps.ResourcePermissions(clipboard_write={}),
    prefers_border=False,
)


class ClaireServer(MCPServer):
    """Serve current widget HTML under any ``ui://claire/`` fingerprint."""

    async def read_resource(self, uri, context=None):  # type: ignore[no-untyped-def]
        if str(uri).startswith(_UI_PREFIX):
            uri = UI_URI
        return await super().read_resource(uri, context)


mcp = ClaireServer(
    "Claire",
    auth=SETTINGS,
    token_verifier=ProviderTokenVerifier(provider),
    extensions=[apps],
)


@mcp.tool(annotations=_ANNOTATIONS)
async def claire_plan_reply(ctx: Context, plan: PlanReplyIn) -> PlanReplyOut:
    """Plan the reply. Call exactly once per reply, before writing it."""
    async with db.Session() as session, session.begin():
        person_id, info = await turn.bind_caller(session, ctx)
        return await turn.run_plan(
            session, person_id=person_id, plan=plan, request=info
        )


mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
