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
from app.core import db, vocab
from app.mcp import turn
from app.mcp.schemas import (
    PlanReplyIn,
    PlanReplyOut,
    RecordEventIn,
    ShowReplyIn,
    ShowReplyOut,
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


def _display_for(ctx: Context) -> vocab.Display:
    """Widget vs chat — fixed from host capabilities (known at connect)."""
    if mcp_apps.client_supports_apps(ctx):
        return vocab.Display.WIDGET
    return vocab.Display.FALLBACK


def _show_ack(shown: ShowReplyOut) -> CallToolResult:
    """Ack only: never echo reply text (hosts would rewrite it in chat).

    The widget reads parts from tool *arguments* via ``ontoolinput``.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=shown.note)],
        structured_content={
            "ok": shown.ok,
            "turn_id": shown.turn_id,
            "display": shown.display,
            "note": shown.note,
        },
    )


apps = mcp_apps.Apps()


@apps.tool(resource_uri=UI_URI, annotations=_ANNOTATIONS)
async def claire_show_reply(ctx: Context, show: ShowReplyIn) -> CallToolResult:
    """Write your entire reply here when plan.display is widget.

    Call only if claire_plan_reply said display=widget. Do not call this in
    fallback mode (write in the chat instead).
    """
    display = _display_for(ctx)
    async with db.Session() as session, session.begin():
        person_id, info = await turn.bind_caller(session, ctx)
        shown = await turn.run_show(
            session,
            person_id=person_id,
            show=show,
            request=info,
            display=display,
        )
    return _show_ack(shown)


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
    """Plan the reply. Call exactly once per reply, before writing it.

    Follow ``display`` and ``instruction`` in the result: widget →
    claire_show_reply only; fallback → write once in the chat.
    """
    display = _display_for(ctx)
    async with db.Session() as session, session.begin():
        person_id, info = await turn.bind_caller(session, ctx)
        return await turn.run_plan(
            session,
            person_id=person_id,
            plan=plan,
            request=info,
            display=display,
        )


mcp_app = mcp.streamable_http_app(
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
    ),
)
