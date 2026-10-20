"""MCP tools and the MCP App resource (MCP Python SDK 2.x, MCP Apps)."""

from collections.abc import Awaitable, Callable
import functools
import hashlib
import logging
import pathlib
from typing import Annotated, Any

from mcp.server import apps as mcp_apps
from mcp.server import caching
from mcp.server import mcpserver
from mcp.server.mcpserver import exceptions
import mcp_types
import pydantic

from app.core import config
from app.core import i18n
from app.core import vocab
from app.services import replies

log = logging.getLogger(__name__)
WIDGET = pathlib.Path(__file__).resolve().parent / "static" / "index.html"
WIDGET_HTML = WIDGET.read_text() if WIDGET.exists() else ""
# The widget's address holds its fingerprint: a new widget is a new address,
# which hosts must fetch instead of serving their cached copy.
UI_PREFIX = "ui://claire/"
UI_URI = (
    f"{UI_PREFIX}reply-"
    f"{hashlib.sha256(WIDGET_HTML.encode()).hexdigest()[:12]}.html"
)
TOKEN_HEADER = "x-claire-token-user"
SESSION_HEADER = "mcp-session-id"
UNKNOWN_HOST = "unknown"
CACHEABLE = ("tools/list", "resources/list", "resources/read")
TOOLS = i18n.catalog(i18n.FALLBACK)["tools"]
FIELDS = TOOLS["plan_reply"]["fields"]
OK = [mcp_types.TextContent(type="text", text="ok")]
# Without it the model may think nothing appeared and repeat the reply.
SHOWN = [mcp_types.TextContent(type="text", text=TOOLS["shown"])]
# ChatGPT requires these hints (Apps SDK reference): the tools only shape the
# conversation, they never touch the user's data elsewhere nor the internet.
ANNOTATIONS = mcp_types.ToolAnnotations(
    read_only_hint=True, destructive_hint=False, open_world_hint=False
)


def _field(name: str, **kwargs: Any) -> pydantic.fields.FieldInfo:
    description = FIELDS[name].format(
        max_message_words=config.settings.max_message_words
    )
    return pydantic.Field(description=description, **kwargs)


# The tools whose arguments carry a person's text: an error never echoes it.
TEXT_TOOLS = ("plan_reply", "show_reply")


class ClaireError(exceptions.ToolError):
    """An error whose message is ours, holding none of the arguments."""


class ClaireServer(mcpserver.MCPServer):
    """MCPServer that hides texts in errors and outlives cached widgets.

    Hosts cache tool definitions: a host may still ask for the widget at an
    older address, which serves the current widget.
    """

    async def read_resource(
        self, uri: Any, context: mcpserver.Context | None = None
    ):
        if str(uri).startswith(UI_PREFIX):
            uri = UI_URI
        return await super().read_resource(uri, context)

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: mcpserver.Context | None = None,
    ):
        if name not in TEXT_TOOLS:
            return await super().call_tool(name, arguments, context)
        try:
            return await super().call_tool(name, arguments, context)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            cause = exc.__cause__ or exc
            log.info("%s failed: %s", name, type(cause).__name__)
            if isinstance(cause, ClaireError):
                raise ClaireError(str(cause)) from None
            if isinstance(cause, replies.ReplyShownError):
                raise ClaireError(TOOLS["errors"]["shown"]) from None
            raise ClaireError(TOOLS["errors"][name]) from None


def caller(ctx: mcpserver.Context) -> replies.Caller:
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
    capabilities = client.capabilities if client else None
    version = (
        client.client_info.version if client and client.client_info else None
    )
    if not ui:
        log.info(
            "No MCP Apps for %s %s: extensions=%s",
            host,
            version or "",
            capabilities.extensions if capabilities else None,
        )
    return replies.Caller(
        int(headers[TOKEN_HEADER]),
        host,
        ui,
        ctx.protocol_version,
        headers.get(SESSION_HEADER),
        version,
        capabilities.model_dump(mode="json", exclude_none=True)
        if capabilities
        else {},
    )


def _handled(
    fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Turns an invalid handle into the generic tool error."""

    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await fn(*args, **kwargs)
        except replies.HandleError:
            raise ClaireError(TOOLS["errors"]["handle"]) from None

    return wrapper


apps = mcp_apps.Apps()


@apps.tool(
    resource_uri=UI_URI,
    description=TOOLS["show_reply"]["description"],
    annotations=ANNOTATIONS,
)
async def show_reply(
    ctx: mcpserver.Context,
    turn_id: str,
    text: str,
    # A field is never renamed nor removed: hosts keep old schemas.
    # No max_length: an extra option must not lose the reply.
    choices: Annotated[
        list[str] | None,
        pydantic.Field(description=TOOLS["show_reply"]["fields"]["choices"]),
    ] = None,
) -> mcp_types.CallToolResult:
    data = await _handled(replies.show_reply)(
        caller(ctx), turn_id, text, choices
    )
    return mcp_types.CallToolResult(content=SHOWN, structured_content=data)


@apps.tool(
    resource_uri=UI_URI,
    visibility=["app"],
    description=TOOLS["record_event"]["description"],
    annotations=ANNOTATIONS,
)
async def record_event(
    ctx: mcpserver.Context,
    turn_id: str,
    kind: str,
    value: str | int | float | None = None,
) -> mcp_types.CallToolResult:
    await _handled(replies.record_event)(caller(ctx), turn_id, kind, value)
    return mcp_types.CallToolResult(content=OK)


@apps.tool(
    resource_uri=UI_URI,
    visibility=["app"],
    description=TOOLS["report_context"]["description"],
    annotations=ANNOTATIONS,
)
async def report_context(
    ctx: mcpserver.Context,
    turn_id: str,
    platform: str | None = None,
    locale: str | None = None,
    time_zone: str | None = None,
) -> mcp_types.CallToolResult:
    await _handled(replies.report_context)(
        caller(ctx), turn_id, platform, locale, time_zone
    )
    return mcp_types.CallToolResult(content=OK)


apps.add_html_resource(
    UI_URI,
    WIDGET_HTML,
    name="reply",
    permissions=mcp_apps.ResourcePermissions(clipboard_write={}),
    prefers_border=False,
)

mcp = ClaireServer(
    "claire",
    version="2.0.0",
    instructions=TOOLS["server"],
    extensions=[apps],
    cache_hints={
        m: caching.CacheHint(config.settings.cache_ttl_ms, "private")
        for m in CACHEABLE
    },
)


@mcp.tool(
    structured_output=False,
    description=TOOLS["plan_reply"]["description"],
    annotations=ANNOTATIONS,
)
async def plan_reply(
    ctx: mcpserver.Context,
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
    ambiguity: Annotated[
        float | None, _field("ambiguity", ge=0.0, le=1.0)
    ] = None,
    last_reply_words: Annotated[int | None, _field("last_reply_words")] = None,
    user_message_words: Annotated[
        int | None, _field("user_message_words")
    ] = None,
    goal: Annotated[str | None, _field("goal")] = None,
    goal_reached: Annotated[bool | None, _field("goal_reached")] = None,
    last_turn_id: Annotated[str | None, _field("last_turn_id")] = None,
    messages_since_last_call: Annotated[
        int | None, _field("messages_since_last_call", ge=0)
    ] = None,
    user_message: Annotated[str | None, _field("user_message")] = None,
) -> str:
    # The signature is the tool schema; the arguments sent are kept as the raw
    # payload, so they are collected as received.
    args = {k: v for k, v in locals().items() if k != "ctx" and v is not None}
    plan = await _handled(replies.plan_reply)(caller(ctx), args)
    return replies.compact(plan)
