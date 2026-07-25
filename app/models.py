"""
SQLAlchemy ORM models for the 4 Phase 3 tables: users, messages,
usage_daily, transcripts.

Kept intentionally simple (no relationships/joins needed yet) — each
service module queries these directly via `AsyncSession`. See
`migrations/` for the Alembic migration that creates these tables; this
module is the source of truth for the schema, but does NOT create tables
itself (no `Base.metadata.create_all()` anywhere) — that's Alembic's job.
"""

from datetime import date, datetime

from sqlalchemy import Boolean, Date, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    """One row per WhatsApp user who has ever messaged the bot."""

    __tablename__ = "users"

    wa_phone: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    last_seen: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    # "urdu" | "english" | "roman" | None (no preference set yet).
    preferred_language: Mapped[str | None] = mapped_column(String, nullable=True)
    is_blocked: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)


class Message(Base):
    """
    One row per incoming WhatsApp message, used purely for webhook
    deduplication (Meta retries webhook deliveries, which would otherwise
    cause the same voice note to be processed/replied-to twice).
    """

    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_wa_phone", "wa_phone"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wa_message_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    wa_phone: Mapped[str] = mapped_column(ForeignKey("users.wa_phone"), nullable=False)
    received_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    # received | skipped_dup | rate_limited | queued | succeeded | failed
    status: Mapped[str] = mapped_column(String, nullable=False, default="received")


class UsageDaily(Base):
    """Per-user, per-UTC-day voice note counter, used for the daily quota."""

    __tablename__ = "usage_daily"

    wa_phone: Mapped[str] = mapped_column(ForeignKey("users.wa_phone"), primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    voice_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)


class Transcript(Base):
    """
    Cleaned transcript history — used for debugging bad outputs, a future
    `/history` command, and language-usage analytics. Audio is never
    stored, only the final cleaned text.
    """

    __tablename__ = "transcripts"
    __table_args__ = (Index("ix_transcripts_wa_phone", "wa_phone"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    wa_phone: Mapped[str] = mapped_column(ForeignKey("users.wa_phone"), nullable=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now(), nullable=False)
    language: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
