"""
End-to-end voice note processing pipeline.

Orchestrates: daily-quota check -> download audio from WhatsApp -> file
size / duration pre-checks -> convert to WAV -> transcribe with Groq
Whisper -> clean up with a Groq LLM (always in the original detected
language — users translate themselves via Translate) -> save transcript
+ reply on WhatsApp with interactive Reply Buttons -> log a structured
analytics event.

Designed to run as a FastAPI `BackgroundTask`, scheduled from the webhook
router *after* WhatsApp has already been acknowledged with a 200 (see
`app/routers/webhook.py`). Rate limiting, deduplication, and the
blocklist check all happen in the webhook router *before* this function
is ever called — this module assumes the caller has already decided
"yes, actually process this voice note".

Every step can fail independently (media URL expired, corrupt audio,
ffmpeg error, Groq API down/rate-limited, etc). On any failure, we log
the full exception for debugging and send the user a single friendly
error message instead of leaving them without any reply at all —
never a raw/internal error message.
"""

import logging
import tempfile
import time
from pathlib import Path

from app.config import settings
from app.services import db_ops
from app.services.commands import send_post_transcript_actions, slash_command_fallback_footer
from app.services.llm import cleanup_transcript, resolve_effective_language
from app.services.transcript_cache import set_last_transcript
from app.services.transcription import convert_to_wav, probe_duration_seconds, transcribe_audio
from app.services.user_errors import ErrorType, classify_for_voice_pipeline, send_user_error
from app.services.user_state import clear_waiting_for_language
from app.services.voice_storage import schedule_voice_upload
from app.services.whatsapp import download_media, get_media_info, send_long_message, send_text_message
from app.utils import log_voice_note_event, mask_phone

logger = logging.getLogger(__name__)

PROCESSING_MESSAGE = "⏳ Processing your voice note..."

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25MB
MAX_DURATION_SECONDS = 3 * 60  # 3 minutes

_LANGUAGE_DISPLAY_LABELS = {
    "urdu": "Urdu",
    "english": "English",
    "roman": "Roman Urdu",
}


async def process_voice_note(
    media_id: str,
    sender: str,
    wa_message_id: str,
) -> None:
    """
    Full pipeline for a single incoming voice/audio message.

    Args:
        media_id: The WhatsApp media ID from the incoming message payload.
        sender: The sender's WhatsApp phone number.
        wa_message_id: This message's WhatsApp ID, used to update its
            `messages.status` row as processing proceeds.
    """
    start_time = time.monotonic()
    timings = {"transcription_time_sec": None, "llm_cleanup_time_sec": None}
    detected_language = None
    audio_duration_sec = None
    transcript_char_count = 0
    quota_reserved = False
    pipeline_succeeded = False

    # A new voice note always cancels any pending "Other language..." wait.
    clear_waiting_for_language(sender)

    try:
        # --- Daily quota — reserve one slot before any download so an
        # over-quota user doesn't cost us bandwidth/ffmpeg time. Failed
        # pipelines refund below so flaky media/Groq doesn't burn the day.
        allowed, _count = await db_ops.try_consume_daily_usage(
            sender, settings.daily_voice_limit
        )
        if not allowed:
            await send_user_error(
                sender, ErrorType.DAILY_LIMIT, limit=settings.daily_voice_limit
            )
            await _safe_update_status(wa_message_id, "failed")
            _log_result(
                sender, wa_message_id, "failed", None, None, timings, 0, "daily_limit_exceeded", start_time
            )
            return
        quota_reserved = True

        processing_sent = await send_text_message(sender, PROCESSING_MESSAGE)
        if not processing_sent:
            logger.error(
                "Failed to send processing acknowledgement | sender=%s | id=%s",
                mask_phone(sender),
                wa_message_id,
            )

        # --- Download + pre-checks --------------------------------------
        media_url, reported_file_size, mime_type = await get_media_info(media_id)

        if reported_file_size is not None and reported_file_size > MAX_FILE_SIZE_BYTES:
            await send_user_error(sender, ErrorType.FILE_TOO_LARGE)
            await _safe_update_status(wa_message_id, "failed")
            _log_result(sender, wa_message_id, "failed", None, None, timings, 0, "file_too_large", start_time)
            return

        audio_bytes = await download_media(media_url)

        if len(audio_bytes) > MAX_FILE_SIZE_BYTES:
            await send_user_error(sender, ErrorType.FILE_TOO_LARGE)
            await _safe_update_status(wa_message_id, "failed")
            _log_result(sender, wa_message_id, "failed", None, None, timings, 0, "file_too_large", start_time)
            return

        # Archive raw audio in the background — never awaited. Failures
        # are logged inside voice_storage and cannot delay transcription
        # or WhatsApp replies.
        schedule_voice_upload(audio_bytes, sender, wa_message_id, mime_type)

        with tempfile.TemporaryDirectory(prefix="voicenotes_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_path = tmp_path / "input.ogg"
            output_path = tmp_path / "converted.wav"
            input_path.write_bytes(audio_bytes)

            audio_duration_sec = await probe_duration_seconds(input_path)
            if not audio_duration_sec:
                logger.warning(
                    "Could not determine audio duration, skipping duration check | sender=%s | id=%s",
                    mask_phone(sender),
                    wa_message_id,
                )
            if audio_duration_sec and audio_duration_sec > MAX_DURATION_SECONDS:
                await send_user_error(sender, ErrorType.DURATION_TOO_LONG)
                await _safe_update_status(wa_message_id, "failed")
                _log_result(
                    sender, wa_message_id, "failed", None, audio_duration_sec, timings, 0, "duration_too_long", start_time
                )
                return

            await convert_to_wav(input_path, output_path)

            transcription_start = time.monotonic()
            raw_transcript, detected_language = await transcribe_audio(output_path)
            timings["transcription_time_sec"] = round(time.monotonic() - transcription_start, 2)

        if not raw_transcript.strip():
            await send_user_error(sender, ErrorType.EMPTY_TRANSCRIPT)
            await _safe_update_status(wa_message_id, "failed")
            _log_result(
                sender, wa_message_id, "failed", detected_language, audio_duration_sec, timings, 0, "empty_transcript", start_time
            )
            return

        # --- Cleanup in the ORIGINAL detected language (no auto-translate).
        # Users pick a target themselves via Translate after the reply.
        effective_detected = resolve_effective_language(detected_language)

        cleanup_start = time.monotonic()
        cleaned_text = await cleanup_transcript(
            raw_transcript, detected_language, target_language=None
        )
        timings["llm_cleanup_time_sec"] = round(time.monotonic() - cleanup_start, 2)

        if not cleaned_text.strip():
            logger.error(
                "LLM cleanup returned empty result for non-empty transcript | "
                "sender=%s | id=%s | raw_chars=%d",
                mask_phone(sender),
                wa_message_id,
                len(raw_transcript),
            )
            await send_user_error(sender, ErrorType.VOICE_PROCESSING_FAILED)
            await _safe_update_status(wa_message_id, "failed")
            _log_result(
                sender,
                wa_message_id,
                "failed",
                detected_language,
                audio_duration_sec,
                timings,
                0,
                "empty_cleanup_result",
                start_time,
            )
            return

        transcript_char_count = len(cleaned_text)

        final_language_key = effective_detected.strip().lower()
        display_label = _LANGUAGE_DISPLAY_LABELS.get(final_language_key, final_language_key.title())

        # Persist + cache before replying — if sending the reply fails
        # partway through chunking, the transcript is still safely saved
        # and available to Translate / Summarize. Quota is considered
        # earned once we have a usable transcript.
        await db_ops.save_transcript(sender, final_language_key, cleaned_text)
        set_last_transcript(sender, cleaned_text, final_language_key)
        pipeline_succeeded = True

        reply = f"🌐 Language: {display_label}\n\n{cleaned_text}"
        transcript_sent = await send_long_message(sender, reply)
        if not transcript_sent:
            logger.error(
                "Failed to deliver transcript message | sender=%s | id=%s",
                mask_phone(sender),
                wa_message_id,
            )
            await send_user_error(sender, ErrorType.WHATSAPP_TEMP_FAILURE)
            await _safe_update_status(wa_message_id, "failed")
            _log_result(
                sender,
                wa_message_id,
                "failed",
                detected_language,
                audio_duration_sec,
                timings,
                transcript_char_count,
                "transcript_send_failed",
                start_time,
            )
            return

        # Second message: interactive Reply Buttons. If Meta rejects the
        # interactive payload (permissions, API issues, etc.), fall back
        # to a short text note so the user isn't stuck.
        buttons_sent = await send_post_transcript_actions(sender)
        if not buttons_sent:
            logger.warning(
                "Reply buttons failed; sending text fallback | sender=%s",
                mask_phone(sender),
            )
            fallback_sent = await send_text_message(sender, slash_command_fallback_footer())
            if not fallback_sent:
                logger.error(
                    "Failed to send buttons-unavailable fallback after transcript | sender=%s",
                    mask_phone(sender),
                )

        await _safe_update_status(wa_message_id, "succeeded")
        _log_result(
            sender, wa_message_id, "success", detected_language, audio_duration_sec, timings, transcript_char_count, None, start_time
        )

    except Exception as exc:
        logger.exception(
            "Voice note processing failed | media_id=%s | sender=%s", media_id, mask_phone(sender)
        )
        error_type = classify_for_voice_pipeline(exc)
        await send_user_error(sender, error_type)
        await _safe_update_status(wa_message_id, "failed")
        _log_result(
            sender,
            wa_message_id,
            "failed",
            detected_language,
            audio_duration_sec,
            timings,
            transcript_char_count,
            error_type.value,
            start_time,
        )
    finally:
        if quota_reserved and not pipeline_succeeded:
            await db_ops.refund_daily_usage(sender)


async def _safe_update_status(wa_message_id: str, status: str) -> None:
    """Best-effort status write — never raises into the pipeline."""
    try:
        await db_ops.update_message_status(wa_message_id, status)
    except Exception:
        logger.exception(
            "Failed to update message status | id=%s | status=%s",
            wa_message_id,
            status,
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
    """Emit the structured analytics log line for one attempt."""
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
