"""people: last connection cache (UA, IP, MCP client)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "people",
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column("people", sa.Column("last_user_agent", sa.Text(), nullable=True))
    op.add_column(
        "people", sa.Column("last_ip", sa.String(length=45), nullable=True)
    )
    op.add_column(
        "people",
        sa.Column("mcp_client_name", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "people",
        sa.Column("mcp_client_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "people",
        sa.Column("mcp_protocol_version", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("people", "mcp_protocol_version")
    op.drop_column("people", "mcp_client_version")
    op.drop_column("people", "mcp_client_name")
    op.drop_column("people", "last_ip")
    op.drop_column("people", "last_user_agent")
    op.drop_column("people", "last_seen_at")
