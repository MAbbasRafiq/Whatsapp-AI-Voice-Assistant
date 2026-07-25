"""
FastAPI application entrypoint.

Run locally with:
    uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import dispose_engine, test_connection
from app.routers import webhook

# --- Logging setup ----------------------------------------------------------
# Configured once, at import time, before the app starts handling requests.
# All modules use `logging.getLogger(__name__)` and inherit this config.
logging.basicConfig(
    level=logging.DEBUG if settings.app_env == "development" else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# httpx/httpcore emit very verbose DEBUG logs (raw TLS handshake steps,
# connection pool internals, etc.) that drown out our own app logs without
# adding much value day-to-day. Keep them at INFO+ regardless of app_env —
# httpx's INFO level already logs one clean line per outbound request
# (e.g. "HTTP Request: POST .../messages 200 OK"), which is exactly the
# level of detail useful for debugging send_text_message() calls.
logging.getLogger("httpx").setLevel(logging.INFO)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# The "analytics" logger (see app/utils.py::log_voice_note_event) gets its
# own handler with a bare "%(message)s" formatter and propagate=False —
# every line it emits is a single, directly-parseable JSON object (Phase
# 3, Part 10), not mixed in with the timestamp/level-prefixed app logs.
_analytics_logger = logging.getLogger("analytics")
_analytics_logger.setLevel(logging.INFO)
_analytics_handler = logging.StreamHandler()
_analytics_handler.setFormatter(logging.Formatter("%(message)s"))
_analytics_logger.addHandler(_analytics_handler)
_analytics_logger.propagate = False

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup/shutdown hook.

    Startup logs config sanity warnings and tests the database
    connection. Per Phase 3's design, migrations are NEVER run
    automatically here (see README "Deployment" section — run
    `alembic upgrade head` manually before first deploy and after every
    schema change), and a failed DB connection does NOT crash the app:
    the webhook can still acknowledge Meta with 200 and serve non-DB
    functionality even while the database is down, which beats the whole
    process refusing to start.
    """
    logger.info("VoiceNotes starting up | app_env=%s", settings.app_env)
    if not settings.whatsapp_token or not settings.whatsapp_phone_number_id:
        logger.warning(
            "WHATSAPP_TOKEN / WHATSAPP_PHONE_NUMBER_ID are not set. "
            "Sending messages will fail until these are configured in .env."
        )
    if not settings.whatsapp_verify_token:
        logger.warning(
            "WHATSAPP_VERIFY_TOKEN is not set. Webhook verification (GET "
            "/webhook) will fail until this is configured in .env."
        )
    if not settings.app_secret:
        logger.warning(
            "APP_SECRET is not set. All incoming webhook POSTs will be "
            "rejected with 403 until this is configured in .env."
        )

    if await test_connection():
        logger.info("Database connection established.")
    else:
        logger.error(
            "Database connection failed. Continuing startup anyway — "
            "DB-dependent features (dedup, blocklist, usage limits, "
            "transcripts, preferences) will not work until this is fixed."
        )

    yield  # App runs while suspended here.

    await dispose_engine()
    logger.info("VoiceNotes shutting down.")


# --- FastAPI app --------------------------------------------------------------
app = FastAPI(
    title="VoiceNotes WhatsApp Bot",
    description=(
        "WhatsApp Cloud API bot that transcribes voice notes (Groq Whisper) "
        "and cleans them up with an LLM. Phase 3 adds PostgreSQL "
        "persistence, webhook signature verification/dedup/blocklist, rate "
        "+ daily usage limiting, language preferences, and slash commands."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# CORS: this API is only ever called by Meta's servers (webhook) and,
# later, maybe an internal dashboard — not by a browser frontend on a
# different origin. We keep CORS permissive for now for ease of local
# development/testing; tighten `allow_origins` before shipping anything
# public-facing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the WhatsApp webhook routes (GET/POST /webhook).
app.include_router(webhook.router)


@app.get("/health")
async def health_check() -> dict:
    """Simple liveness check to confirm the server is up and reachable."""
    return {"status": "ok"}
