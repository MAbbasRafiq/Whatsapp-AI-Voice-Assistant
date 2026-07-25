"""create initial tables

Revision ID: c36104edea14
Revises:
Create Date: 2026-07-25 15:39:22.061127

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c36104edea14'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: create users, messages, usage_daily, transcripts."""
    op.create_table(
        "users",
        sa.Column("wa_phone", sa.String(), primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("last_seen", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("preferred_language", sa.String(), nullable=True),
        sa.Column("is_blocked", sa.Boolean(), server_default=sa.text("false"), nullable=False),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("wa_message_id", sa.String(), nullable=False),
        sa.Column("wa_phone", sa.String(), sa.ForeignKey("users.wa_phone"), nullable=False),
        sa.Column("received_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="received"),
    )
    op.create_unique_constraint("uq_messages_wa_message_id", "messages", ["wa_message_id"])
    op.create_index("ix_messages_wa_phone", "messages", ["wa_phone"])

    op.create_table(
        "usage_daily",
        sa.Column("wa_phone", sa.String(), sa.ForeignKey("users.wa_phone"), primary_key=True),
        sa.Column("day", sa.Date(), primary_key=True),
        sa.Column("voice_count", sa.Integer(), server_default="0", nullable=False),
    )

    op.create_table(
        "transcripts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("wa_phone", sa.String(), sa.ForeignKey("users.wa_phone"), nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("language", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
    )
    op.create_index("ix_transcripts_wa_phone", "transcripts", ["wa_phone"])


def downgrade() -> None:
    """Downgrade schema: drop all Phase 3 tables in dependency order."""
    op.drop_index("ix_transcripts_wa_phone", table_name="transcripts")
    op.drop_table("transcripts")
    op.drop_table("usage_daily")
    op.drop_index("ix_messages_wa_phone", table_name="messages")
    op.drop_constraint("uq_messages_wa_message_id", "messages", type_="unique")
    op.drop_table("messages")
    op.drop_table("users")
