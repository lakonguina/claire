"""people: country of the browser that gave consent

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-23
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, Sequence[str], None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("people", sa.Column("country", sa.String(length=2)))


def downgrade() -> None:
    op.drop_column("people", "country")
