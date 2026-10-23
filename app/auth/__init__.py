"""OAuth 2.1 for the MCP connector (anonymous consent)."""

from app.auth.oauth import MCP_PATH, SETTINGS, provider, router, routes

__all__ = ["MCP_PATH", "SETTINGS", "provider", "router", "routes"]
