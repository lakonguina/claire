"""Configuration from environment variables (see .env.example)."""

from typing import Annotated

import pydantic
import pydantic_settings

from app.core import vocab

SCHEMA_VERSION = 3
CommaList = Annotated[list[str], pydantic_settings.NoDecode]


class Settings(pydantic_settings.BaseSettings):
    """Runtime settings, read from the environment by field name."""

    model_config = pydantic_settings.SettingsConfigDict(frozen=True)

    database_url: str = "postgresql://claire:claire@localhost:5432/claire"
    admin_token: str = ""
    public_url: str = "http://localhost:8000"
    policy: str = "bayes-v1"
    shadow_policies: CommaList = []
    control_unit: vocab.RandomizationUnit = vocab.RandomizationUnit.AUTO
    # AUTO randomises per person from this many people active in 30 days.
    per_user_from: int = 100
    # Shares of new conversations (or people) held back in a comparison arm:
    # the holdout is the permanent baseline; the concise arm is opt-in.
    holdout_share: float = 0.03
    concise_share: float = 0.0
    # Share of conversations whose user messages are stored (with consent).
    message_sample: float = 0.1
    # Longer messages are stored as their length only.
    max_message_words: int = 150
    # Share of reached goals followed by the "did you get it?" question.
    ground_truth_share: float = 0.1
    # Calibrated confidence above which the clarification bandit may ask.
    ambiguity_threshold: float = 0.5
    # Share of conversations where one tool field is not asked, to measure
    # what it is worth; the fields that may be ablated.
    ablation_share: float = 0.1
    ablation_fields: CommaList = [
        "ambiguity",
        "last_turn_id",
        "messages_since_last_call",
        "user_message_words",
    ]
    # Output tokens plan_reply arguments should stay under on a normal turn.
    token_budget: int = 60
    # Share of conversations on the candidate prompt catalog, when there is
    # one (locales/candidate.json): an instruction change is a launch.
    candidate_share: float = 0.0
    # Scripted-conversation check of the instruction: "<provider>:<model>".
    eval_models: CommaList = ["anthropic:claude-sonnet-5-5"]
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    # Stored text is deleted after this many days.
    retention_days: int = 3 * 365
    # Base64 of the 32-byte AES-256-GCM key that encrypts stored text.
    data_key: str = ""
    privacy_email: str = "privacy@claire.pm"
    q: float = pydantic.Field(0.8, validation_alias="TARGET_QUANTILE")
    # Local multilingual sentence model for topic domains.
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
        "shadow_policies",
        "allowed_origins",
        "ablation_fields",
        "eval_models",
        mode="before",
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
            vocab.Arm.HOLDOUT: self.holdout_share,
            vocab.Arm.CONCISE: self.concise_share,
        }


settings = Settings()
