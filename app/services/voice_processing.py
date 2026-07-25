"""
End-to-end voice note processing pipeline.

Orchestrates: daily-quota check -> download audio from WhatsApp -> file
size / duration pre-checks -> convert to WAV -> transcribe with Groq
Whisper -> clean up with a Groq LLM (honoring any saved language
preference) -> save transcript + reply to the user on WhatsApp -> log a
structured analytics event.

Designed to run as a FastAPI `BackgroundTask`, scheduled from the webhook
router *after* WhatsApp has already been acknowledged with a 200 (see
`app/routers/webhook.py`). Rate limiting, deduplication, and the
blocklist check all happen in the webhook router *before* this function
is ever called — this module assumes the caller has already decided
"yes, actually process this voice note".

Every step can fail independently (media URL expired, corrupt audio,
ffmpeg error, Groq API down/rate-limited, etc). On any failure, we log
the full exception for debugging and send the user a single friendly
error message instead of leaving them without any reply at all (Part 9)
— never a raw/internal error message.
"""

import logging
import tempfile
import time
from pathlib import Path

from app.config import settings
from app.services import db_ops
from app.services.llm import cleanup_transcript, resolve_effective_language
from app.services.transcript_cache import set_last_transcript
from app.services.transcription import convert_to_wav, probe_duration_seconds, transcribe_audio
from app.services.whatsapp import download_media, get_media_info, send_long_message, send_text_message
from app.utils import log_voice_note_event, mask_phone

logger = logging.getLogger(__name__)

PROCESSING_MESSAGE = "⏳ Processing your voice note..."

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
MAX_DURATION_SECONDS = 3 * 60  # 3 minutes

_ERROR_MESSAGE = (
    "⚠️ Something went wrong while processing your voice note.\nPlease try again in a minute."
)

_EMPTY_TRANSCRIPT_MESSAGE = (
    "🤔 I couldn't make out any speech in that voice note. Could you try sending it again?"
)

_FILE_TOO_LARGE_MESSAGE = "⚠️ Voice note is too large (max 25MB).\nPlease send a shorter recording."

_DURATION_TOO_LONG_MESSAGE = (
    "⚠️ Voice note is too long (max 3 minutes).\nPlease split it into shorter parts."
)

_DAILY_LIMIT_MESSAGE_TEMPLATE = (
    "⚠️ You've used all {limit} voice notes for today.\nResets at midnight UTC. See you tomorrow! 🌙"
)

_ONBOARDING_FOOTER_TEMPLATE = """\

💡 I used {language} for this transcript.
Want to set a permanent preference?

/urdu — Always reply in اردو
/english — Always reply in English
/roman — Always reply in Roman Urdu

You can change this anytime."""

_STANDARD_FOOTER = "──────────────\n💡 /translate • /summarize • /roman • /help"

_LANGUAGE_DISPLAY_LABELS = {
    "urdu": "Urdu",
    "english": "English",
    "roman": "Roman Urdu",
}


async def process_voice_note(
    media_id: str,
    sender: str,
    wa_message_id: str,
    is_new_user: bool,
    preferred_language: str | None,
) -> None:
    """
    Full pipeline for a single incoming voice/audio message.

    Args:
        media_id: The WhatsApp media ID from the incoming message payload.
        sender: The sender's WhatsApp phone number.
        wa_message_id: This message's WhatsApp ID, used to update its
            `messages.status` row as processing proceeds (Part 3b).
        is_new_user: Whether this is the very first message we've ever
            seen from this phone number (decides onboarding vs. regular
            footer — Part 5/6).
        preferred_language: The user's saved `preferred_language`
            ("urdu"/"english"/"roman"), or None if not set yet.
    """
    start_time = time.monotonic()
    timings = {"transcription_time_sec": None, "llm_cleanup_time_sec": None}
    detected_language = None
    audio_duration_sec = None
    transcript_char_count = 0

    try:
        # --- Daily quota check (Part 3e) — before any download, so an
        # over-quota user doesn't cost us bandwidth/ffmpeg time. ---------
        new_count = await db_ops.increment_daily_usage(sender)
        if new_count > settings.daily_voice_limit:
            await send_text_message(
                sender, _DAILY_LIMIT_MESSAGE_TEMPLATE.format(limit=settings.daily_voice_limit)
            )
            await db_ops.update_message_status(wa_message_id, "failed")
            _log_result(sender, wa_message_id, "failed", None, None, timings, 0, "daily_limit_exceeded", start_time)
            return

        await send_text_message(sender, PROCESSING_MESSAGE)

        # --- Download + pre-checks (Part 4) -----------------------------
        media_url, reported_file_size = await get_media_info(media_id)

        if reported_file_size is not None and reported_file_size > MAX_FILE_SIZE_BYTES:
            await send_text_message(sender, _FILE_TOO_LARGE_MESSAGE)
            await db_ops.update_message_status(wa_message_id, "failed")
            _log_result(sender, wa_message_id, "failed", None, None, timings, 0, "file_too_large", start_time)
            return

        audio_bytes = await download_media(media_url)

        if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
            await send_text_message(sender, _FILE_TOO_LARGE_MESSAGE)
            await db_ops.update_message_status(wa_message_id, "failed")
            _log_result(sender, wa_message_id, "failed", None, None, timings, 0, "file_too_large", start_time)
            return

        with tempfile.TemporaryDirectory(prefix="voicenotes_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.ogg"
            output_path = tmp_path / "converted.wav"
            input_path.write_bytes(audio_bytes)

            audio_duration_sec = await probe_duration_seconds(input_path)
            if audio_duration_sec > MAX_DURATION_SECONDS:
                await send_text_message(sender, _DURATION_TOO_LONG_MESSAGE)
                await db_ops.update_message_status(wa_message_id, "failed")
                _log_result(
                    sender, wa_message_id, "failed", None, audio_duration_sec, timings, 0, "duration_too_long", start_time
                )
                return

            await convert_to_wav(input_path, output_path)

            transcription_start = time.monotonic()
            raw_transcript, detected_language = await transcribe_audio(output_path)
            timings["transcription_time_sec"] = round(time.monotonic() - transcription_start, 2)

        if not raw_transcript.strip():
            await send_text_message(sender, _EMPTY_TRANSCRIPT_MESSAGE)
            await db_ops.update_message_status(wa_message_id, "failed")
            _log_result(
                sender, wa_message_id, "failed", detected_language, audio_duration_sec, timings, 0, "empty_transcript", start_time
            )
            return

        # --- Cleanup, honoring a saved language preference (Part 5) ----
        effective_detected = resolve_effective_language(detected_language)
        target_language = None
        if preferred_language and preferred_language.strip().lower() != effective_detected.strip().lower():
            target_language = preferred_language.strip().lower()

        cleanup_start = time.monotonic()
        cleaned_text = await cleanup_transcript(raw_transcript, detected_language, target_language=target_language)
        timings["llm_cleanup_time_sec"] = round(time.monotonic() - cleanup_start, 2)
        transcript_char_count = len(cleaned_text)

        final_language_key = target_language or effective_detected.strip().lower()
        display_label = _LANGUAGE_DISPLAY_LABELS.get(final_language_key, final_language_key.title())

        # Persist + cache before replying — if sending the reply fails
        # partway through chunking, the transcript is still safely saved
        # and available to /translate, /summarize, and future /history.
        await db_ops.save_transcript(sender, final_language_key, cleaned_text)
        set_last_transcript(sender, cleaned_text, final_language_key)

        footer = (
            _ONBOARDING_FOOTER_TEMPLATE.format(language=display_label)
            if is_new_user
            else f"\n\n{_STANDARD_FOOTER}"
        )
        reply = f"🌐 Language: {display_label}\n\n{cleaned_text}\n\n{footer.strip()}"

        await send_long_message(sender, reply)
        await db_ops.update_message_status(wa_message_id, "succeeded")
        _log_result(
            sender, wa_message_id, "success", detected_language, audio_duration_sec, timings, transcript_char_count, None, start_time
        )

    except Exception as exc:
        logger.exception(
            "Voice note processing failed | media_id=%s | sender=%s", media_id, mask_phone(sender)
        )
        await send_text_message(sender, _ERROR_MESSAGE)
        await db_ops.update_message_status(wa_message_id, "failed")
        _log_result(
            sender,
            wa_message_id,
            "failed",
            detected_language,
            audio_duration_sec,
            timings,
            transcript_char_count,
            type(exc).__name__,
            start_time,
        )


def _log_result(
    wa_phone: str,
    wa_message_id: str,
    status: str,
    detected_language: str | None,
    audio_duration_sec: float | None,
    timings: dict,
    transcript_char_count: int,
    error: str | None,
    start_time: float,
) -> None:
    """Emit the Part 10 structured analytics log line for one attempt."""
    log_voice_note_event(
        event="voice_note_processed",
        wa_phone=mask_phone(wa_phone),
        wa_message_id=wa_message_id,
        status=status,
        detected_language=detected_language,
        audio_duration_sec=audio_duration_sec,
        transcription_time_sec=timings.get("transcription_time_sec"),
        llm_cleanup_time_sec=timings.get("llm_cleanup_time_sec"),
        total_time_sec=round(time.monotonic() - start_time, 2),
        transcript_char_count=transcript_char_count,
        error=error,
    )
