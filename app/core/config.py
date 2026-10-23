"""Runtime settings loaded from environment variables.

Think of this as the control panel for Claire: database address, secret key,
holdout share, etc. Values come from the process environment (and optional
``.env`` file) via pydantic-settings — we do not hard-code secrets in code.
"""

from __future__ import annotations

import pydantic_settings

# Bumped when the shape of tool arguments / event payloads changes in a
# backward-incompatible way. Old events keep their original version.
SCHEMA_VERSION = 1


class Settings(pydantic_settings.BaseSettings):
    """Immutable process settings (frozen after load).

    Attributes:
        database_url: Postgres connection string (who hosts our memory).
        data_key: Base64 of a 32-byte AES key; locks private payloads at rest.
        public_url: Canonical public base URL of this deployment.
        policy_version: Label written on every logged decision (for audits).
        holdout_share: Fraction of new chats that get no Claire adaptation
            (the control group for measuring if we help).
        user_message_words: At or below this length (words), the assistant
            always sends the full user message; above it, only a short
            request / summary (no random ``copy_long`` draw).
        cell_prior_strength: How strongly a new person inherits the global
            average before we have their own feedback.
        admin_token: Shared secret for ``/admin``. Empty disables the UI.
    """

    model_config = pydantic_settings.SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
    )

    database_url: str = "postgresql://claire:claire@127.0.0.1:5432/claire"
    data_key: str = ""
    public_url: str = "http://127.0.0.1:8000"
    policy_version: str = "1.0"
    holdout_share: float = 0.03
    user_message_words: int = 150
    cell_prior_strength: float = 5.0
    admin_token: str = ""


settings = Settings()
