"""Events stay append-only by convention, not by trigger.

The application still never updates nor deletes an event; the owner can now
correct history by hand (then rerun the nightly job, which replays it).

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-18 20:30:00.000000

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0007'
down_revision: Union[str, Sequence[str], None] = '0006'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

APPEND_ONLY = (
    'users',
    'conversations',
    'turns',
    'replies',
    'outcomes',
    'buttons_shown',
    'clicks',
    'gold_labels',
    'shadow_decisions',
    'tool_calls',
    'turn_context',
    'profile_edits',
    'topic_embeddings',
)


def upgrade() -> None:
    """Upgrade schema."""
    for table in APPEND_ONLY:
        op.execute(f'DROP TRIGGER {table}_append_only ON {table}')
    op.execute('DROP FUNCTION forbid_change()')


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(
        'CREATE FUNCTION forbid_change() RETURNS trigger LANGUAGE plpgsql AS '
        "$$ BEGIN RAISE EXCEPTION 'table % is append-only', TG_TABLE_NAME; "
        'END $$'
    )
    for table in APPEND_ONLY:
        op.execute(
            f'CREATE TRIGGER {table}_append_only BEFORE UPDATE OR DELETE OR '
            f'TRUNCATE ON {table} FOR EACH STATEMENT '
            'EXECUTE FUNCTION forbid_change()'
        )
