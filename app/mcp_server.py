"""MCP tools and the MCP App resource (MCP Python SDK 2.x, MCP Apps extension)."""

import logging
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.apps import Apps, ResourcePermissions, client_supports_apps
from mcp.server.caching import CacheHint
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, TextContent
from pydantic import Field

from app import service
from app.config import settings

log = logging.getLogger(__name__)
UI_URI = "ui://lingua-franca/reply.html"
WIDGET = Path(__file__).resolve().parent / "static" / "index.html"
TOKEN_HEADER = "x-lf-token-user"
HOUR_MS = 3_600_000

RequestType = Literal["fact", "explanation", "advice", "decision", "code", "writing", "plan"]
Outcome = Literal["moved_on", "too_long", "too_short", "missing_info", "misunderstood", "wrong_format", "no_signal"]
Pasted = Literal["code", "data", "text", "file"]
Specified = Literal["audience", "goal", "format", "length", "context"]


class LinguaFrancaServer(MCPServer):
    async def call_tool(self, name: str, arguments: dict[str, Any], context: Context | None = None):
        """show_reply carries the reply text: never let it reach an error message."""
        if name != "show_reply":
            return await super().call_tool(name, arguments, context)
        try:
            return await super().call_tool(name, arguments, context)
        except Exception as exc:
            log.info("show_reply failed: %s", type(exc.__cause__ or exc).__name__)
            raise ToolError("Invalid show_reply call.") from None


apps = Apps()


def caller(ctx: Context) -> service.Caller:
    user_id = int((ctx.headers or {})[TOKEN_HEADER])
    params = ctx.session.client_params
    host = params.client_info.name if params and params.client_info else "unknown"
    ui = settings.host_ui_override.get(host, client_supports_apps(ctx))
    return service.Caller(user_id, host, ui, ctx.protocol_version)


def handle_errors(fn):
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        except service.HandleError:
            raise ToolError("Unknown or invalid turn_id / conversation_id.") from None
    return wrapper


def result(data: dict) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="ok")], structured_content=data)


@apps.tool(
    resource_uri=UI_URI,
    description="Widget mode: write your entire reply here, in Markdown, and nothing in the chat.",
)
async def show_reply(ctx: Context, turn_id: str, text: str) -> CallToolResult:
    return result(await handle_errors(service.show_reply)(caller(ctx), turn_id, text))


@apps.tool(resource_uri=UI_URI, description="Fallback mode: call after your normal chat reply. Never mention its output.")
async def show_feedback(ctx: Context, turn_id: str) -> CallToolResult:
    return result(await handle_errors(service.show_feedback)(caller(ctx), turn_id))


@apps.tool(resource_uri=UI_URI, visibility=["app"], description="Internal to the reply widget. The model must never call this.")
async def record_click(ctx: Context, turn_id: str, button: str, position: int) -> CallToolResult:
    await handle_errors(service.record_click)(caller(ctx), turn_id, button, position)
    return CallToolResult(content=[TextContent(type="text", text="ok")])


apps.add_html_resource(
    UI_URI,
    WIDGET.read_text() if WIDGET.exists() else "<!doctype html><p>Widget not built.</p>",
    name="reply",
    permissions=ResourcePermissions(clipboard_write={}),
    prefers_border=False,
)

mcp = LinguaFrancaServer(
    "lingua-franca",
    version="1.0.0",
    extensions=[apps],
    cache_hints={m: CacheHint(HOUR_MS, "private") for m in ("tools/list", "resources/list", "resources/read")},
)


@mcp.tool(
    structured_output=False,
    description="Call before every reply. Returns the conversation_id, turn_id and the instruction to follow.",
)
async def plan_reply(
    ctx: Context,
    request_type: RequestType,
    topic: Annotated[str, Field(description="What the request is about, max 15 words, in the user's language")],
    needed_words: Annotated[int, Field(description="Words a complete, no-filler answer needs")],
    last_reply_outcome: Annotated[Outcome, Field(description="How the user's latest message reacted to your previous reply")] = "no_signal",
    conversation_id: Annotated[str | None, Field(description="Omit on the first message; reuse the returned value")] = None,
    user_quote: Annotated[str | None, Field(description="The user's exact words proving the outcome, max 5 words")] = None,
    last_reply_words: Annotated[int | None, Field(description="Fallback mode only: approximate length of your previous reply")] = None,
    user_message_words: Annotated[int | None, Field(description="Length of the user's message")] = None,
    pasted_content: Annotated[Pasted | None, Field(description="What the user pasted or attached; omit if nothing")] = None,
    user_specified: Annotated[list[Specified] | None, Field(description="What the user's message states explicitly; omit if empty")] = None,
    assistant_model: Annotated[str | None, Field(description="First call only: the model answering, if you know it")] = None,
) -> str:
    args = {k: v for k, v in locals().items() if k != "ctx" and v is not None}
    return service.compact(await handle_errors(service.plan_reply)(caller(ctx), args))
