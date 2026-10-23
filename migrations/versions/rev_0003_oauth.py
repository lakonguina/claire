"""OAuth clients, grants, and hashed credentials

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "oauth_clients",
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column("info", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("client_id"),
    )
    op.create_table(
        "oauth_grants",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("person_id", sa.BigInteger(), nullable=False),
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consent_version", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(["client_id"], ["oauth_clients.client_id"]),
        sa.ForeignKeyConstraint(["person_id"], ["people.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_oauth_grants_person_id"),
        "oauth_grants",
        ["person_id"],
        unique=False,
    )
    op.create_table(
        "oauth_credentials",
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("client_id", sa.String(length=128), nullable=False),
        sa.Column("grant_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "data",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("jsonb_build_object()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["client_id"], ["oauth_clients.client_id"]),
        sa.ForeignKeyConstraint(["grant_id"], ["oauth_grants.id"]),
        sa.PrimaryKeyConstraint("token_hash"),
    )
    op.create_index(
        op.f("ix_oauth_credentials_grant_id"),
        "oauth_credentials",
        ["grant_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_oauth_credentials_grant_id"), table_name="oauth_credentials"
    )
    op.drop_table("oauth_credentials")
    op.drop_index(op.f("ix_oauth_grants_person_id"), table_name="oauth_grants")
    op.drop_table("oauth_grants")
    op.drop_table("oauth_clients")
