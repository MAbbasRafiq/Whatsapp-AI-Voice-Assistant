"""
Centralized user-facing error messages for WhatsApp replies.

All production failure replies that the user sees should go through
`send_user_error` so wording stays consistent and we never leak stack
traces, API bodies, secrets, or internal details to WhatsApp.

Call sites keep their existing try/except structure; this module only
owns the message text + the send helper + exception classification.
"""

from __future__ import annotations

import logging
import subprocess
from enum import Enum
from typing import Any

import httpx
from groq import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from sqlalchemy.exc import SQLAlchemyError

from app.services.whatsapp import send_text_message
from app.utils import mask_phone

logger = logging.getLogger(__name__)

# Re-export for callers that want the enum without importing Enum machinery.
__all__ = [
    "ErrorType",
    "send_user_error",
    "classify_exception",
    "classify_for_voice_pipeline",
    "classify_for_request",
    "classify_for_translate",
]


class ErrorType(str, Enum):
    """Stable keys for every user-facing error reply."""

    # --- Already handled (wording preserved exactly) --------------------
    RATE_LIMITED = "rate_limited"
    DAILY_LIMIT = "daily_limit"
    FILE_TOO_LARGE = "file_too_large"
    DURATION_TOO_LONG = "duration_too_long"
    EMPTY_TRANSCRIPT = "empty_transcript"
    VOICE_PROCESSING_FAILED = "voice_processing_failed"
    REQUEST_FAILED = "request_failed"
    NO_TRANSCRIPT = "no_transcript"
    UNSUPPORTED_LANGUAGE = "unsupported_language"
    EMPTY_LANGUAGE = "empty_language"
    WAITING_EXPIRED = "waiting_expired"

    # --- Newly classified gaps ------------------------------------------
    GROQ_UNAVAILABLE = "groq_unavailable"
    GROQ_TIMEOUT = "groq_timeout"
    GROQ_RATE_LIMIT = "groq_rate_limit"
    GROQ_AUTH = "groq_auth"
    WHATSAPP_TEMP_FAILURE = "whatsapp_temp_failure"
    WHATSAPP_RATE_LIMIT = "whatsapp_rate_limit"
    AUDIO_DOWNLOAD_FAILED = "audio_download_failed"
    MEDIA_EXPIRED = "media_expired"
    UNSUPPORTED_MEDIA_FORMAT = "unsupported_media_format"
    AUDIO_DECODE_FAILED = "audio_decode_failed"
    DATABASE_UNAVAILABLE = "database_unavailable"
    UNEXPECTED_INTERACTIVE = "unexpected_interactive"
    INTERNAL_UNEXPECTED = "internal_unexpected"


# Exact existing wording preserved for previously handled cases.
_ERROR_MESSAGES: dict[ErrorType, str] = {
    ErrorType.RATE_LIMITED: (
        "⏳ Please wait a few seconds before sending another voice note."
    ),
    ErrorType.DAILY_LIMIT: (
        "⚠️ You've used all {limit} voice notes for today.\n"
        "Resets at midnight UTC. See you tomorrow! 🌙"
    ),
    ErrorType.FILE_TOO_LARGE: (
        "⚠️ Voice note is too large (max 25MB).\nPlease send a shorter recording."
    ),
    ErrorType.DURATION_TOO_LONG: (
        "⚠️ Voice note is too long (max 3 minutes).\nPlease split it into shorter parts."
    ),
    ErrorType.EMPTY_TRANSCRIPT: (
        "🤔 I couldn't make out any speech in that voice note. "
        "Could you try sending it again?"
    ),
    ErrorType.VOICE_PROCESSING_FAILED: (
        "⚠️ Something went wrong while processing your voice note.\n"
        "Please try again in a minute."
    ),
    ErrorType.REQUEST_FAILED: (
        "⚠️ Something went wrong while processing your request. "
        "Please try again in a minute."
    ),
    ErrorType.NO_TRANSCRIPT: (
        "I couldn't find a recent transcript.\n"
        "Remember voice expires after 15 min\n"
        "Send a new voice note."
    ),
    ErrorType.UNSUPPORTED_LANGUAGE: (
        "⚠️ I couldn't translate into that language. "
        "Please try a common language name (e.g. Spanish, German, Turkish)."
    ),
    ErrorType.EMPTY_LANGUAGE: ("🌍 Type the language you'd like to translate into."),
    ErrorType.WAITING_EXPIRED: (
        "⏳ That language request expired. "
        "Send a voice note or tap Translate again."
    ),
    # New specific messages for previously-generic failure paths.
    ErrorType.GROQ_UNAVAILABLE: (
        "⚠️ Our transcription service is temporarily unavailable.\n"
        "Please try again in a few minutes."
    ),
    ErrorType.GROQ_TIMEOUT: (
        "⚠️ That took too long to process.\nPlease try again in a minute."
    ),
    ErrorType.GROQ_RATE_LIMIT: (
        "⚠️ We're getting a lot of requests right now.\n"
        "Please wait a moment and try again."
    ),
    ErrorType.GROQ_AUTH: (
        "⚠️ Voice processing is temporarily misconfigured on our side.\n"
        "Please try again later."
    ),
    ErrorType.WHATSAPP_TEMP_FAILURE: (
        "⚠️ WhatsApp is having trouble right now.\nPlease try again in a minute."
    ),
    ErrorType.WHATSAPP_RATE_LIMIT: (
        "⚠️ WhatsApp is rate-limiting messages right now.\n"
        "Please wait a moment and try again."
    ),
    ErrorType.AUDIO_DOWNLOAD_FAILED: (
        "⚠️ I couldn't download your voice note.\nPlease send it again."
    ),
    ErrorType.MEDIA_EXPIRED: (
        "⚠️ That voice note is no longer available to download.\n"
        "Please send it again."
    ),
    ErrorType.UNSUPPORTED_MEDIA_FORMAT: (
        "⚠️ I couldn't process that audio format.\n"
        "Please send a normal WhatsApp voice note."
    ),
    ErrorType.AUDIO_DECODE_FAILED: (
        "⚠️ I couldn't read that audio file.\n"
        "It may be corrupted — please try recording again."
    ),
    ErrorType.DATABASE_UNAVAILABLE: (
        "⚠️ I'm having a temporary issue saving your request.\n"
        "Please try again in a minute."
    ),
    ErrorType.UNEXPECTED_INTERACTIVE: (
        "🎤 Send me a voice note and I'll convert it to clean text!\n\n"
        "Or tap Help after a transcript for tips."
    ),
    ErrorType.INTERNAL_UNEXPECTED: (
        "⚠️ Something went wrong on my side.\nPlease try again in a minute."
    ),
}


async def send_user_error(
    to: str,
    error_type: ErrorType | str,
    **format_kwargs: Any,
) -> bool:
    """
    Send a standardized friendly WhatsApp error reply.

    Logs which error_type was sent (for correlation with the caller's
    detailed exception log). Never includes exception text, stack traces,
    or API payloads in the WhatsApp body — callers should `logger.exception`
    separately for the full traceback.
    """
    if isinstance(error_type, str):
        try:
            error_type = ErrorType(error_type)
        except ValueError:
            logger.error("Unknown error_type=%r; falling back to internal", error_type)
            error_type = ErrorType.INTERNAL_UNEXPECTED

    template = _ERROR_MESSAGES[error_type]
    try:
        message = template.format(**format_kwargs) if format_kwargs else template
    except KeyError:
        logger.exception(
            "Missing format kwargs for error_type=%s kwargs=%r",
            error_type.value,
            format_kwargs,
        )
        message = _ERROR_MESSAGES[ErrorType.INTERNAL_UNEXPECTED]

    logger.warning(
        "User error reply | to=%s | error_type=%s",
        mask_phone(to),
        error_type.value,
    )

    return await send_text_message(to, message)


def classify_exception(exc: BaseException) -> ErrorType:
    """
    Map a caught exception to the most specific user-facing ErrorType.

    Order matters: more specific subclasses / status codes first.
    Unknown failures fall back to INTERNAL_UNEXPECTED (callers that used
    a domain-specific generic message historically can remap that).
    """
    # --- Database -------------------------------------------------------
    if isinstance(exc, SQLAlchemyError):
        return ErrorType.DATABASE_UNAVAILABLE

    # --- Groq -----------------------------------------------------------
    if isinstance(exc, RateLimitError):
        return ErrorType.GROQ_RATE_LIMIT
    if isinstance(exc, APITimeoutError):
        return ErrorType.GROQ_TIMEOUT
    if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
        return ErrorType.GROQ_AUTH
    if isinstance(exc, BadRequestError):
        detail = _exception_text(exc).lower()
        if any(
            token in detail
            for token in ("format", "codec", "audio", "file type", "unsupported", "media type")
        ):
            return ErrorType.UNSUPPORTED_MEDIA_FORMAT
        return ErrorType.GROQ_UNAVAILABLE
    if isinstance(exc, InternalServerError):
        return ErrorType.GROQ_UNAVAILABLE
    if isinstance(exc, APIConnectionError):
        return ErrorType.GROQ_UNAVAILABLE
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return ErrorType.GROQ_RATE_LIMIT
        if status in (401, 403):
            return ErrorType.GROQ_AUTH
        return ErrorType.GROQ_UNAVAILABLE

    # --- ffmpeg / local audio decode ------------------------------------
    if isinstance(exc, subprocess.CalledProcessError):
        return ErrorType.AUDIO_DECODE_FAILED

    # --- WhatsApp Graph / media download (httpx) ------------------------
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        url = str(exc.request.url) if exc.request is not None else ""
        if code == 429:
            return ErrorType.WHATSAPP_RATE_LIMIT
        if code in (401, 403):
            # Token / permission issue talking to Graph — treat as WA temp
            # misconfig from the user's perspective (same soft message).
            return ErrorType.WHATSAPP_TEMP_FAILURE
        if code == 404:
            return ErrorType.MEDIA_EXPIRED
        if code >= 500:
            return ErrorType.WHATSAPP_TEMP_FAILURE
        # Other 4xx on media endpoints → download / media problem.
        if "graph.facebook.com" in url or "/media" in url:
            return ErrorType.AUDIO_DOWNLOAD_FAILED
        return ErrorType.WHATSAPP_TEMP_FAILURE

    if isinstance(exc, httpx.TimeoutException):
        return ErrorType.AUDIO_DOWNLOAD_FAILED

    if isinstance(exc, httpx.RequestError):
        return ErrorType.AUDIO_DOWNLOAD_FAILED

    # --- Config / media lookup helpers in whatsapp.py -------------------
    if isinstance(exc, RuntimeError):
        text = _exception_text(exc).lower()
        if "whatsapp_token" in text or "not configured" in text:
            return ErrorType.WHATSAPP_TEMP_FAILURE
        if "no 'url' field" in text or "media" in text:
            return ErrorType.MEDIA_EXPIRED

    return ErrorType.INTERNAL_UNEXPECTED


def classify_for_voice_pipeline(exc: BaseException) -> ErrorType:
    """
    Classify an exception raised inside the voice-note pipeline.

    Preserves the historical generic voice-processing message as the
    fallback when nothing more specific matches.
    """
    error_type = classify_exception(exc)
    if error_type == ErrorType.INTERNAL_UNEXPECTED:
        return ErrorType.VOICE_PROCESSING_FAILED
    return error_type


def classify_for_request(exc: BaseException) -> ErrorType:
    """
    Classify an exception raised inside command / interactive / language
    handlers. Preserves the historical generic request-failed fallback.
    """
    error_type = classify_exception(exc)
    if error_type == ErrorType.INTERNAL_UNEXPECTED:
        return ErrorType.REQUEST_FAILED
    return error_type


def classify_for_translate(exc: BaseException) -> ErrorType:
    """
    Classify translate failures.

    Infra / API failures get specific messages. Anything else keeps the
    historical "unsupported language" reply (previous catch-all behavior
    for non-classified errors).
    """
    error_type = classify_exception(exc)
    if error_type == ErrorType.INTERNAL_UNEXPECTED:
        return ErrorType.UNSUPPORTED_LANGUAGE
    return error_type


def _exception_text(exc: BaseException) -> str:
    """Best-effort string form for keyword checks — never sent to users."""
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if body is not None:
        parts.append(str(body))
    message = getattr(exc, "message", None)
    if message is not None:
        parts.append(str(message))
    return " ".join(parts)
