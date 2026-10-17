"""Configuration from environment variables."""

import json
import os
from dataclasses import dataclass, field


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in os.environ.get(name, "").split(",") if v.strip()]


@dataclass(frozen=True)
class Settings:
    database_url: str = os.environ.get("DATABASE_URL", "postgresql://lf:lf@localhost:5432/lf")
    admin_token: str = os.environ.get("ADMIN_TOKEN", "")
    public_url: str = os.environ.get("PUBLIC_URL", "http://localhost:8000")
    policy: str = os.environ.get("POLICY", "bayes-v1")
    shadow_policies: list[str] = field(default_factory=lambda: _env_list("SHADOW_POLICIES"))
    control_unit: str = os.environ.get("CONTROL_UNIT", "conversation")
    control_share: float = float(os.environ.get("CONTROL_SHARE", "0"))  # > 0 only for an opt-in study
    q: float = float(os.environ.get("TARGET_QUANTILE", "0.8"))
    # ttlMs on tools/list and resources; set to 0 while developing the widget so hosts refetch it.
    cache_ttl_ms: int = int(os.environ.get("CACHE_TTL_MS", "3600000"))
    # Per-host override of MCP Apps support, e.g. {"some-host": false}.
    host_ui_override: dict[str, bool] = field(
        default_factory=lambda: json.loads(os.environ.get("HOST_UI_OVERRIDE", "{}"))
    )


SCHEMA_VERSION = 1
settings = Settings()
