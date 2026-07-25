"""
Async SQLAlchemy engine + session setup.

This module owns the single async engine/sessionmaker for the whole app.
Everything else (models, request handlers, background tasks) imports
`AsyncSessionLocal` from here rather than constructing its own engine.

Migrations are NOT run automatically anywhere in this app (see
`migrations/README` and the main README's "Deployment" section) — they
must be run manually via `alembic upgrade head` before first deploy and
after every schema change. `main.py` only tests connectivity on startup
and logs success/failure; it never creates/alters tables itself.
"""

import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models (see app/models.py)."""


# pool_size/max_overflow are kept deliberately small: Railway's free-tier
# Postgres (and the free tier of the app service itself) has modest
# connection limits, and this app is a single web process handling
# webhook traffic + a handful of background tasks — it doesn't need a
# large pool. pool_pre_ping guards against stale connections that Railway
# or the DB may have silently closed (common with managed free-tier DBs
# that recycle idle connections).
_engine = create_async_engine(
    settings.database_url or "postgresql+asyncpg://invalid/invalid",
    echo=False,
    pool_size=5,
    max_overflow=5,
    pool_pre_ping=True,
    pool_recycle=300,
)

AsyncSessionLocal = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    autoflush=False,
)


async def test_connection() -> bool:
    """
    Verify the database is reachable by running a trivial query.

    Called once at app startup (see `app/main.py`). Intentionally never
    raises — a DB outage at boot shouldn't crash the whole app, since the
    webhook can still acknowledge Meta with 200 (and skip DB-dependent
    features) even while the database is down. Returns True/False so the
    caller can log an appropriate message.
    """
    if not settings.database_url:
        logger.warning("DATABASE_URL is not configured; skipping DB connectivity check.")
        return False

    try:
        async with _engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("Database connectivity check failed.")
        return False


async def dispose_engine() -> None:
    """Cleanly close all pooled connections on app shutdown."""
    await _engine.dispose()
