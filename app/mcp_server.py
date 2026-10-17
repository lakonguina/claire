"""MCP tools and the MCP App resource (MCP Python SDK 2.x, MCP Apps)."""

from collections.abc import Awaitable, Callable
import functools
import logging
import pathlib
from typing import Annotated, Any

from mcp.server import apps as mcp_apps
from mcp.server import caching
from mcp.server import mcpserver
from mcp.server.mcpserver import exceptions
import mcp_types
import pydantic

from app import config
from app import i18n
from app import service
from app import vocab

log = logging.getLogger(__name__)
UI_URI = "ui://lingua-franca/reply.html"
WIDGET = pathlib.Path(__file__).resolve().parent / "static" / "index.html"
TOKEN_HEADER = "x-lf-token-user"
SESSION_HEADER = "mcp-session-id"
UNKNOWN_HOST = "unknown"
CACHEABLE = ("tools/list", "resources/list", "resources/read")
TOOLS = i18n.catalog(i18n.FALLBACK)["tools"]
FIELDS = TOOLS["plan_reply"]["fields"]
OK = [mcp_types.TextContent(type="text", text="ok")]


def _field(name: str, **kwargs: Any) -> pydantic.fields.FieldInfo:
    return pydantic.Field(description=FIELDS[name], **kwargs)


class LinguaFrancaServer(mcpserver.MCPServer):
    """MCPServer whose show_reply errors never carry the reply text."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: mcpserver.Context | None = None,
    ):
        if name != "show_reply":
            return await super().call_tool(name, arguments, context)
        try:
            return await super().call_tool(name, arguments, context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            log.info(
                "show_reply failed: %s", type(exc.__cause__ or exc).__name__
            )
            raise exceptions.ToolError(TOOLS["errors"]["show_reply"]) from None


def caller(ctx: mcpserver.Context) -> service.Caller:
    """Returns who calls: the user behind the token, the host and its mode."""
    headers = ctx.headers or {}
    client = ctx.session.client_params
    host = (
        client.client_info.name
        if client and client.client_info
        else UNKNOWN_HOST
    )
    ui = config.settings.host_ui_override.get(
        host, mcp_apps.client_supports_apps(ctx)
    )
    return service.Caller(
        int(headers[TOKEN_HEADER]),
        host,
        ui,
        ctx.protocol_version,
        headers.get(SESSION_HEADER),
    )


def _handled(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Turns an invalid handle into the generic tool error."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except service.HandleError:
            raise exceptions.ToolError(TOOLS["errors"]["handle"]) from None

    return wrapper


apps = mcp_apps.Apps()


@apps.tool(resource_uri=UI_URI, description=TOOLS["show_reply"]["description"])
async def show_reply(
    ctx: mcpserver.Context, turn_id: str, text: str
) -> mcp_types.CallToolResult:
    data = await _handled(service.show_reply)(caller(ctx), turn_id, text)
    return mcp_types.CallToolResult(content=OK, structured_content=data)


@apps.tool(
    resource_uri=UI_URI,
    visibility=["app"],
    description=TOOLS["record_click"]["description"],
)
async def record_click(
    ctx: mcpserver.Context, turn_id: str, button: str, position: int
) -> mcp_types.CallToolResult:
    await _handled(service.record_click)(caller(ctx), turn_id, button, position)
    return mcp_types.CallToolResult(content=OK)


@apps.tool(
    resource_uri=UI_URI,
    visibility=["app"],
    description=TOOLS["report_context"]["description"],
)
async def report_context(
    ctx: mcpserver.Context,
    turn_id: str,
    platform: str | None = None,
    locale: str | None = None,
) -> mcp_types.CallToolResult:
    await _handled(service.report_context)(
        caller(ctx), turn_id, platform, locale
    )
    return mcp_types.CallToolResult(content=OK)


apps.add_html_resource(
    UI_URI,
    WIDGET.read_text() if WIDGET.exists() else "",
    name="reply",
    permissions=mcp_apps.ResourcePermissions(clipboard_write={}),
    prefers_border=False,
)

mcp = LinguaFrancaServer(
    "lingua-franca",
    version="2.0.0",
    extensions=[apps],
    cache_hints={
        m: caching.CacheHint(config.settings.cache_ttl_ms, "private")
        for m in CACHEABLE
    },
)


@mcp.tool(
    structured_output=False, description=TOOLS["plan_reply"]["description"]
)
async def plan_reply(
    ctx: mcpserver.Context,
    request_type: Annotated[vocab.RequestType, _field("request_type")],
    topic: Annotated[str, _field("topic")],
    language: Annotated[str, _field("language", pattern=r"^[a-z]{2}$")],
    needed_words: Annotated[int, _field("needed_words")],
    last_reply_outcome: Annotated[
        vocab.Outcome, _field("last_reply_outcome")
    ] = vocab.Outcome.NO_SIGNAL,
    conversation_id: Annotated[str | None, _field("conversation_id")] = None,
    user_quote: Annotated[str | None, _field("user_quote")] = None,
    format_issue: Annotated[
        vocab.FormatIssue | None, _field("format_issue")
    ] = None,
    ambiguous: Annotated[bool | None, _field("ambiguous")] = None,
    last_reply_words: Annotated[int | None, _field("last_reply_words")] = None,
    user_message_words: Annotated[
        int | None, _field("user_message_words")
    ] = None,
) -> str:
    # The signature is the tool schema; the arguments sent are kept as the raw
    # payload (SPEC 5.4), so they are collected as received.
    args = {k: v for k, v in locals().items() if k != "ctx" and v is not None}
    plan = await _handled(service.plan_reply)(caller(ctx), args)
    return service.compact(plan)
