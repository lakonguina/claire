"""Configuration from environment variables."""

import dataclasses
import json
import os

from app import vocab

SCHEMA_VERSION = 2


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_list(name: str) -> list[str]:
    return [v.strip() for v in _env(name, "").split(",") if v.strip()]


@dataclasses.dataclass(frozen=True)
class Settings:
    """Runtime settings; see .env.example."""

    database_url: str = _env(
        "DATABASE_URL", "postgresql://lf:lf@localhost:5432/lf"
    )
    admin_token: str = _env("ADMIN_TOKEN", "")
    public_url: str = _env("PUBLIC_URL", "http://localhost:8000")
    policy: str = _env("POLICY", "bayes-v1")
    shadow_policies: list[str] = dataclasses.field(
        default_factory=lambda: _env_list("SHADOW_POLICIES")
    )
    control_unit: vocab.RandomizationUnit = vocab.RandomizationUnit(
        _env("CONTROL_UNIT", vocab.RandomizationUnit.CONVERSATION)
    )
    # Share of new conversations (or users) per comparison arm: keep 0
    # except for an opt-in study (SPEC 4.1).
    arm_shares: dict[vocab.Arm, float] = dataclasses.field(
        default_factory=lambda: {
            vocab.Arm.CONTROL: float(_env("CONTROL_SHARE", "0")),
            vocab.Arm.CONCISE: float(_env("CONCISE_SHARE", "0")),
        }
    )
    q: float = float(_env("TARGET_QUANTILE", "0.8"))
    # ttlMs of tools/list and resources; 0 while developing the widget.
    cache_ttl_ms: int = int(_env("CACHE_TTL_MS", "3600000"))
    # Origins allowed to call the MCP endpoint (DNS-rebinding protection).
    allowed_origins: list[str] = dataclasses.field(
        default_factory=lambda: _env_list("ALLOWED_ORIGINS")
        or ["https://claude.ai", "https://chatgpt.com"]
    )
    # Per-host override of MCP Apps support, e.g. {"some-host": false}.
    host_ui_override: dict[str, bool] = dataclasses.field(
        default_factory=lambda: json.loads(_env("HOST_UI_OVERRIDE", "{}"))
    )


settings = Settings()
