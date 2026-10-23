"""Dump everything MCP + HTTP expose about the caller into one JSON blob.

Stored on every event under ``meta["request"]``. No whitelist: if the
transport or SDK handed it to us, it goes in the blob (JSON-safe copy).
"""

from __future__ import annotations

from typing import Any, Mapping

from mcp.server.mcpserver.context import Context
from pydantic import BaseModel, ConfigDict, Field

try:
    from starlette.requests import Request as StarletteRequest
except ImportError:  # pragma: no cover
    StarletteRequest = Any  # type: ignore[misc, assignment]


def _jsonable(value: Any, *, _depth: int = 0) -> Any:
    """Best-effort JSON-safe copy of SDK / Starlette objects."""
    if _depth > 12:
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.hex()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json", by_alias=True)
        except TypeError:
            try:
                return dump()
            except Exception:
                return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item, _depth=_depth + 1) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        public = {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
        if public:
            return _jsonable(public, _depth=_depth + 1)
    return str(value)


def _header_map(headers: Mapping[str, str] | None) -> dict[str, str]:
    if not headers:
        return {}
    return {str(key).lower(): str(value) for key, value in headers.items()}


def _pick_ip(headers: Mapping[str, str], request: Any) -> str | None:
    for key in ("cf-connecting-ip", "x-real-ip"):
        if headers.get(key):
            return headers[key].split(",")[0].strip()
    forwarded = headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    if client is not None and getattr(client, "host", None):
        return str(client.host)
    return None


def _http_blob(request: Any, headers: Mapping[str, str]) -> dict[str, Any]:
    """Everything visible on the Starlette / ASGI request."""
    blob: dict[str, Any] = {"headers": dict(headers)}
    if request is None:
        return blob
    url = getattr(request, "url", None)
    if url is not None:
        blob["url"] = str(url)
        blob["path"] = getattr(url, "path", None)
        blob["scheme"] = getattr(url, "scheme", None)
    method = getattr(request, "method", None)
    if method:
        blob["method"] = method
    client = getattr(request, "client", None)
    if client is not None:
        blob["client"] = {
            "host": getattr(client, "host", None),
            "port": getattr(client, "port", None),
        }
    query = getattr(request, "query_params", None)
    if query is not None:
        try:
            blob["query"] = dict(query)
        except Exception:
            blob["query"] = str(query)
    path_params = getattr(request, "path_params", None)
    if path_params:
        blob["path_params"] = _jsonable(path_params)
    return blob


class RequestInfo(BaseModel):
    """Full caller snapshot. ``data`` is what we persist; helpers dig into it."""

    model_config = ConfigDict(extra="ignore")

    data: dict[str, Any] = Field(default_factory=dict)

    def as_meta(self) -> dict[str, Any]:
        """Return the full JSON blob stored under ``meta["request"]``."""
        return dict(self.data)

    @property
    def ip(self) -> str | None:
        http = self.data.get("http") or {}
        if isinstance(http, dict) and http.get("ip"):
            return str(http["ip"])
        return None

    @property
    def user_agent(self) -> str | None:
        http = self.data.get("http") or {}
        headers = http.get("headers") if isinstance(http, dict) else None
        if isinstance(headers, dict) and headers.get("user-agent"):
            return str(headers["user-agent"])
        return None

    @property
    def mcp_client_name(self) -> str | None:
        info = self._client_info()
        name = info.get("name") if info else None
        return str(name) if name else None

    @property
    def mcp_client_version(self) -> str | None:
        info = self._client_info()
        version = info.get("version") if info else None
        return str(version) if version else None

    @property
    def mcp_protocol_version(self) -> str | None:
        value = self.data.get("protocol_version")
        return str(value) if value else None

    def _client_info(self) -> dict[str, Any]:
        params = self.data.get("client_params")
        if isinstance(params, dict):
            info = params.get("clientInfo") or params.get("client_info")
            if isinstance(info, dict):
                return info
        return {}


def collect_http_request_info(request: StarletteRequest) -> RequestInfo:
    """Full dump for a plain HTTP request (consent / OAuth pages)."""
    headers = _header_map(request.headers)
    http = _http_blob(request, headers)
    http["ip"] = _pick_ip(headers, request)
    return RequestInfo(data={"http": http, "source": "http"})


def collect_request_info(ctx: Context) -> RequestInfo:
    """Dump everything available on the MCP ``Context`` for this tool call."""
    data: dict[str, Any] = {"source": "mcp"}

    try:
        data["request_id"] = ctx.request_id
    except (ValueError, AttributeError):
        pass

    try:
        data["protocol_version"] = ctx.protocol_version
    except (ValueError, AttributeError):
        pass

    try:
        data["input_responses"] = _jsonable(ctx.input_responses)
    except (ValueError, AttributeError):
        pass

    try:
        data["request_state"] = ctx.request_state
    except (ValueError, AttributeError):
        pass

    # Per-message request context (method, raw params incl. _meta, …).
    try:
        rc = ctx.request_context
        data["method"] = getattr(rc, "method", None)
        data["params"] = _jsonable(getattr(rc, "params", None))
        data["meta"] = _jsonable(getattr(rc, "meta", None))
        if getattr(rc, "request_id", None) is not None:
            data["request_id"] = str(rc.request_id)
        if getattr(rc, "protocol_version", None):
            data["protocol_version"] = rc.protocol_version
    except (ValueError, AttributeError):
        pass

    # Initialize handshake snapshot for this session.
    try:
        session = ctx.session
        data["client_params"] = _jsonable(session.client_params)
        data["client_capabilities"] = _jsonable(session.client_capabilities)
        if getattr(session, "protocol_version", None):
            data.setdefault("protocol_version", session.protocol_version)
    except (ValueError, AttributeError):
        try:
            data["client_capabilities"] = _jsonable(ctx.client_capabilities)
        except (ValueError, AttributeError):
            pass

    # HTTP transport envelope — every header, URL, peer, …
    request = None
    try:
        request = ctx.request_context.request
    except (ValueError, AttributeError):
        request = None

    try:
        headers = _header_map(ctx.headers)
    except (ValueError, AttributeError):
        headers = _header_map(getattr(request, "headers", None))

    http = _http_blob(request, headers)
    http["ip"] = _pick_ip(headers, request)
    data["http"] = http

    # Drop empty / null leaves so the blob stays readable.
    return RequestInfo(data=_prune(data))


def _prune(value: Any) -> Any:
    if isinstance(value, dict):
        out = {
            key: pruned
            for key, item in value.items()
            if (pruned := _prune(item)) not in (None, {}, [])
        }
        return out
    if isinstance(value, list):
        return [pruned for item in value if (pruned := _prune(item)) not in (None, {}, [])]
    return value
