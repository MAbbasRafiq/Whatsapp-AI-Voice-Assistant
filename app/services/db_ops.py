"""
Small, focused async database operations used by the webhook handler,
voice note pipeline, and command handlers.

Each function opens and closes its own short-lived session rather than
accepting a shared one from the caller. This keeps every operation
independently atomic and avoids the complexity of passing a single
AsyncSession across FastAPI's request handler and `BackgroundTasks`
(which run after the request/response cycle — sharing a session across
that boundary is fragile). The extra connection-pool churn this causes is
negligible at this app's scale and the pool (see app/database.py) is
sized accordingly.
"""

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from app.database import AsyncSessionLocal
from app.models import Message, Transcript, UsageDaily, User

logger = logging.getLogger(__name__)


async def touch_user(wa_phone: str) -> tuple[User, bool]:
    """
    Ensure a `users` row exists for `wa_phone` and update `last_seen`.

    Returns (user, is_new_user). `is_new_user` is True only the very
    first time this phone number is ever seen.
    """
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.wa_phone == wa_phone))
        user = result.scalar_one_or_none()

        if user is not None:
            user.last_seen = datetime.now(timezone.utc)
            await session.commit()
            return user, False

        user = User(wa_phone=wa_phone)
        session.add(user)
        try:
            await session.commit()
            return user, True
        except IntegrityError:
            # Lost a race with a concurrent insert for the same brand-new
            # phone number (extremely unlikely given the 10s rate limit,
            # but cheap to handle correctly).
            await session.rollback()
            result = await session.execute(select(User).where(User.wa_phone == wa_phone))
            user = result.scalar_one()
            user.last_seen = datetime.now(timezone.utc)
            await session.commit()
            return user, False


async def is_blocked(wa_phone: str) -> bool:
    """Return True if this user is on the manual blocklist."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User.is_blocked).where(User.wa_phone == wa_phone))
        value = result.scalar_one_or_none()
        return bool(value)


async def record_message(wa_message_id: str, wa_phone: str, status: str = "received") -> bool:
    """
    Insert a row into `messages` for deduplication purposes.

    Returns True if this is the first time we've seen this
    `wa_message_id` (i.e. genuinely new, safe to process). Returns False
    if a row with this `wa_message_id` already exists — Meta retried a
    webhook delivery we already handled, and the caller should skip
    processing entirely (no reply, just acknowledge with 200).
    """
    async with AsyncSessionLocal() as session:
        message = Message(wa_message_id=wa_message_id, wa_phone=wa_phone, status=status)
        session.add(message)
        try:
            await session.commit()
            return True
        except IntegrityError:
            await session.rollback()
            logger.info("Duplicate webhook delivery detected for wa_message_id=%s", wa_message_id)
            return False


async def update_message_status(wa_message_id: str, status: str) -> None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Message).where(Message.wa_message_id == wa_message_id))
        message = result.scalar_one_or_none()
        if message is not None:
            message.status = status
            await session.commit()


async def try_consume_daily_usage(wa_phone: str, limit: int) -> tuple[bool, int]:
    """
    Atomically reserve one daily voice-note slot for `wa_phone` if under
    `limit`.

    Returns (allowed, count):
      - (True, new_count) when a unit was consumed (new_count is 1..limit).
      - (False, current_count) when already at/over limit — the counter is
        NOT incremented, so rejected spam cannot inflate usage forever.

    Uses PostgreSQL `INSERT ... ON CONFLICT DO UPDATE ... WHERE` so
    concurrent consumes cannot race past the cap.
    """
    # Use UTC explicitly (not date.today(), which is local server time) —
    # the daily quota must reset at midnight UTC regardless of where the
    # app happens to be deployed.
    today = datetime.now(timezone.utc).date()

    async with AsyncSessionLocal() as session:
        stmt = (
            pg_insert(UsageDaily)
            .values(wa_phone=wa_phone, day=today, voice_count=1)
            .on_conflict_do_update(
                index_elements=[UsageDaily.wa_phone, UsageDaily.day],
                set_={"voice_count": UsageDaily.voice_count + 1},
                where=UsageDaily.voice_count < limit,
            )
            .returning(UsageDaily.voice_count)
        )
        result = await session.execute(stmt)
        new_count = result.scalar_one_or_none()
        if new_count is not None:
            await session.commit()
            return True, int(new_count)

        await session.rollback()
        current = await session.execute(
            select(UsageDaily.voice_count).where(
                UsageDaily.wa_phone == wa_phone,
                UsageDaily.day == today,
            )
        )
        current_count = current.scalar_one_or_none()
        return False, int(current_count if current_count is not None else limit)


async def refund_daily_usage(wa_phone: str) -> None:
    """
    Best-effort undo of one `try_consume_daily_usage` reservation when
    the voice pipeline fails before a successful transcript reply.
    Never raises — quota accounting must not block error handling.
    """
    today = datetime.now(timezone.utc).date()
    try:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(UsageDaily).where(
                    UsageDaily.wa_phone == wa_phone,
                    UsageDaily.day == today,
                )
            )
            row = result.scalar_one_or_none()
            if row is None or row.voice_count <= 0:
                return
            row.voice_count -= 1
            await session.commit()
    except Exception:
        logger.exception(
            "Failed to refund daily usage | wa_phone=%s",
            wa_phone,
        )


async def save_transcript(wa_phone: str, language: str, text: str) -> None:
    async with AsyncSessionLocal() as session:
        session.add(Transcript(wa_phone=wa_phone, language=language, text=text))
        await session.commit()
