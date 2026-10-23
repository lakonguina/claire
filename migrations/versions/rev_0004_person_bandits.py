"""people: form_counts and words_counts for Thompson bandits

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "people",
        sa.Column(
            "form_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("jsonb_build_object()"),
            nullable=False,
        ),
    )
    op.add_column(
        "people",
        sa.Column(
            "words_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("jsonb_build_object()"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("people", "words_counts")
    op.drop_column("people", "form_counts")
