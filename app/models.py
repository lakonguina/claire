"""SQLAlchemy models: people and the append-only event log.

The ``events`` table is the source of truth. We never edit the past: every
tool call, decision, and reply is a new row. Derived state (bandits, etc.)
is rebuilt by replaying these rows — that is how we prove the online
brain matches history.

``people`` holds the stable identity plus a *cache of the last connection*
(UA, IP, MCP client). Full history of connects still goes into
``token_refresh`` events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM tables."""


class Person(Base):
    """Someone who connected Claire (one row per account / subject).

    Connection facts come from two layers:

    * **MCP** ``initialize`` / request ``_meta``: ``clientInfo.name`` /
      ``version`` (self-reported host app — Claude, ChatGPT, …). Not for
      security decisions.
    * **HTTP** (Streamable HTTP): ``User-Agent``, client IP
      (``Request.client.host`` / ``X-Forwarded-For`` behind a proxy).

    Attributes:
        id: Internal numeric id (never shown as a public handle).
        created_at: When they first appeared in our system.
        last_seen_at: Last successful connect / token refresh.
        last_user_agent: Last HTTP ``User-Agent`` (browser / host app).
        last_ip: Last client IP we observed.
        country: ISO country code of the browser that gave consent.
        mcp_client_name: Last MCP ``clientInfo.name``.
        mcp_client_version: Last MCP ``clientInfo.version``.
        mcp_protocol_version: Last negotiated / declared protocol version.
        form_counts: Dirichlet weights for form arms (updated by rewards).
        words_counts: Dirichlet weights for word buckets (updated by rewards).
    """

    __tablename__ = "people"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    last_user_agent: Mapped[str | None] = mapped_column(Text)
    last_ip: Mapped[str | None] = mapped_column(String(45))
    country: Mapped[str | None] = mapped_column(String(2))
    mcp_client_name: Mapped[str | None] = mapped_column(String(128))
    mcp_client_version: Mapped[str | None] = mapped_column(String(64))
    mcp_protocol_version: Mapped[str | None] = mapped_column(String(32))
    form_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=func.jsonb_build_object(),
    )
    words_counts: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=func.jsonb_build_object(),
    )


class Event(Base):
    """One immutable fact about what happened.

    Attributes:
        id: Row id.
        person_id: Who this belongs to.
        conversation_id: Chat handle (``c_…``), if any.
        turn_id: Turn handle (``t_…``), if any.
        type: Value from :class:`app.core.vocab.EventType`.
        schema_version: Payload shape version at write time.
        meta: Non-secret structured fields (host, policy, UA, …) as JSON.
        payload_enc: Optional AES-GCM ciphertext of sensitive JSON.
        created_at: Server timestamp when the row was appended.
    """

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    person_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("people.id"), index=True, nullable=False
    )
    conversation_id: Mapped[str | None] = mapped_column(
        String(64), index=True
    )
    turn_id: Mapped[str | None] = mapped_column(String(64), index=True)
    type: Mapped[str] = mapped_column(String(32), index=True, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=func.jsonb_build_object(),
    )
    payload_enc: Mapped[bytes | None] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False,
    )


class OAuthClient(Base):
    """An AI host registered via OAuth dynamic client registration.

    Attributes:
        client_id: Public client id from registration.
        info: Full RFC 7591 client metadata as JSON.
        created_at: When the client was registered.
    """

    __tablename__ = "oauth_clients"

    client_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    info: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class OAuthGrant(Base):
    """One connector: a host authorized to act as a person until revoked.

    Attributes:
        id: Grant id.
        person_id: Anonymous person created at consent.
        client_id: Which host holds the tokens.
        created_at: When the person clicked allow.
        revoked_at: When the grant died (revoke or token reuse).
        consent_version: Hash of the consent wording they accepted.
    """

    __tablename__ = "oauth_grants"

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True
    )
    person_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("people.id"), index=True, nullable=False
    )
    client_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("oauth_clients.client_id"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    consent_version: Mapped[str] = mapped_column(String(64), nullable=False)


class OAuthCredential(Base):
    """Pending authorize request, code, access or refresh token (by hash).

    Cleartext tokens never hit the database. Replaying a spent code or
    refresh token revokes the whole grant (RFC 9700 §4.14).

    Attributes:
        token_hash: SHA-256 hex of the credential.
        kind: Value from :class:`app.core.vocab.Credential`.
        client_id: Owning client.
        grant_id: Owning grant (``None`` for a pending request).
        data: Extra SDK fields (scopes, PKCE, resource, …).
        expires_at: Hard expiry.
        used_at: When a one-shot credential was spent.
    """

    __tablename__ = "oauth_credentials"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    client_id: Mapped[str] = mapped_column(
        String(128), ForeignKey("oauth_clients.client_id"), nullable=False
    )
    grant_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("oauth_grants.id"), index=True
    )
    data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=func.jsonb_build_object(),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
