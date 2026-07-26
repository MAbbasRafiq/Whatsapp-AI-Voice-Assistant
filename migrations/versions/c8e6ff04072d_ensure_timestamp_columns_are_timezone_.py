"""ensure timestamp columns are timezone-aware

Revision ID: c8e6ff04072d
Revises: c36104edea14
Create Date: 2026-07-26 11:56:48.618299

Converts timestamp columns to TIMESTAMP WITH TIME ZONE when they are still
stored as timestamp without time zone. Environments that already use
timestamptz (including the initial Phase 3 migration) are left unchanged.
"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "c8e6ff04072d"
down_revision: Union[str, Sequence[str], None] = "c36104edea14"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# (table, column) pairs that must be timezone-aware in the ORM and DB.
_TIMESTAMP_COLUMNS = (
    ("users", "created_at"),
    ("users", "last_seen"),
    ("messages", "received_at"),
    ("transcripts", "created_at"),
)


def upgrade() -> None:
    """Upgrade naive timestamp columns to TIMESTAMP WITH TIME ZONE."""
    # Conditional ALTER avoids double-converting values that are already
    # timestamptz (`AT TIME ZONE` on timestamptz yields timestamp without tz).
    for table, column in _TIMESTAMP_COLUMNS:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = '{table}'
                      AND column_name = '{column}'
                      AND data_type = 'timestamp without time zone'
                ) THEN
                    ALTER TABLE {table}
                        ALTER COLUMN {column}
                        TYPE TIMESTAMP WITH TIME ZONE
                        USING {column} AT TIME ZONE 'UTC';
                END IF;
            END $$;
            """
        )


def downgrade() -> None:
    """Downgrade timestamptz columns back to timestamp without time zone."""
    for table, column in _TIMESTAMP_COLUMNS:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = '{table}'
                      AND column_name = '{column}'
                      AND data_type = 'timestamp with time zone'
                ) THEN
                    ALTER TABLE {table}
                        ALTER COLUMN {column}
                        TYPE TIMESTAMP WITHOUT TIME ZONE
                        USING {column} AT TIME ZONE 'UTC';
                END IF;
            END $$;
            """
        )
