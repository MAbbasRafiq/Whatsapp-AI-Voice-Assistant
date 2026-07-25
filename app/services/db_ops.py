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
    first time this phone number is ever seen — used to decide whether
    to show the first-time onboarding block (Part 5) vs the regular
    footer (Part 6).
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


async def get_preferred_language(wa_phone: str) -> str | None:
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User.preferred_language).where(User.wa_phone == wa_phone))
        return result.scalar_one_or_none()


async def set_preferred_language(wa_phone: str, language: str) -> None:
    """language must be one of 'urdu' | 'english' | 'roman'."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(User).where(User.wa_phone == wa_phone))
        user = result.scalar_one_or_none()
        if user is None:
            # Shouldn't normally happen (touch_user runs first for every
            # message), but guard against it defensively.
            user = User(wa_phone=wa_phone)
            session.add(user)
        user.preferred_language = language
        await session.commit()


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


async def increment_daily_usage(wa_phone: str) -> int:
    """
    Atomically increment today's (UTC) voice note count for `wa_phone`
    and return the new count.

    Uses PostgreSQL's `INSERT ... ON CONFLICT DO UPDATE` so concurrent
    increments for the same user/day can't race and undercount (a plain
    SELECT-then-UPDATE would have a lost-update race).
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
            )
            .returning(UsageDaily.voice_count)
        )
        result = await session.execute(stmt)
        new_count = result.scalar_one()
        await session.commit()
        return new_count


async def save_transcript(wa_phone: str, language: str, text: str) -> None:
    async with AsyncSessionLocal() as session:
        session.add(Transcript(wa_phone=wa_phone, language=language, text=text))
        await session.commit()
