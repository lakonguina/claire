"""Configuration from environment variables (see .env.example)."""

from typing import Annotated

import pydantic
import pydantic_settings

from app import vocab

SCHEMA_VERSION = 2
CommaList = Annotated[list[str], pydantic_settings.NoDecode]


class Settings(pydantic_settings.BaseSettings):
    """Runtime settings, read from the environment by field name."""

    model_config = pydantic_settings.SettingsConfigDict(frozen=True)

    database_url: str = "postgresql://lf:lf@localhost:5432/lf"
    admin_token: str = ""
    public_url: str = "http://localhost:8000"
    policy: str = "bayes-v1"
    shadow_policies: CommaList = []
    control_unit: vocab.RandomizationUnit = vocab.RandomizationUnit.CONVERSATION
    # Shares of new conversations (or users) held back in a comparison arm:
    # keep 0 except for an opt-in study (SPEC 4.1).
    control_share: float = 0.0
    concise_share: float = 0.0
    q: float = pydantic.Field(0.8, validation_alias="TARGET_QUANTILE")
    # Local multilingual sentence model for topic domains (SPEC 7.1).
    embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # ttlMs of tools/list and resources; 0 while developing the widget.
    cache_ttl_ms: int = 3_600_000
    # Origins allowed to call the MCP endpoint (DNS-rebinding protection).
    allowed_origins: CommaList = ["https://claude.ai", "https://chatgpt.com"]
    # Per-host override of MCP Apps support, e.g. {"some-host": false}.
    host_ui_override: dict[str, bool] = {}

    @pydantic.field_validator(
        "shadow_policies", "allowed_origins", mode="before"
    )
    @classmethod
    def _split(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @property
    def arm_shares(self) -> dict[vocab.Arm, float]:
        """The share of each comparison arm."""
        return {
            vocab.Arm.CONTROL: self.control_share,
            vocab.Arm.CONCISE: self.concise_share,
        }


settings = Settings()
