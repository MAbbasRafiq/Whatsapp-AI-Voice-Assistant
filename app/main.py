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

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook. Currently just logs config sanity warnings."""
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

    yield  # App runs while suspended here.

    logger.info("VoiceNotes shutting down.")


# --- FastAPI app --------------------------------------------------------------
app = FastAPI(
    title="VoiceNotes WhatsApp Bot",
    description=(
        "Phase 1: project skeleton + WhatsApp Cloud API webhook for "
        "receiving/sending text messages. Audio transcription and LLM "
        "cleanup land in later phases."
    ),
    version="0.1.0",
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
