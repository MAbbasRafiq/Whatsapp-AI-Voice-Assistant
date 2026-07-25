"""
End-to-end voice note processing pipeline.

Orchestrates: download audio from WhatsApp -> convert to WAV -> transcribe
with Groq Whisper -> clean up with a Groq LLM -> reply to the user on
WhatsApp.

This is designed to run as a FastAPI `BackgroundTask`, scheduled from the
webhook router *after* WhatsApp has already been acknowledged with a 200
(see `app/routers/webhook.py`). That's important: every step in here
(network calls, ffmpeg, LLM inference) is slow enough that doing it inline
in the webhook handler would violate WhatsApp's "respond fast" requirement
and risk Meta retrying the delivery.

Every step can fail independently (media URL expired, corrupt audio,
ffmpeg error, Groq API down/rate-limited, etc.). On any failure, we log
the details for debugging and send the user a single friendly error
message instead of leaving them without any reply at all.
"""

import logging
import tempfile
from pathlib import Path

from app.services.llm import cleanup_transcript, resolve_effective_language
from app.services.transcription import convert_to_wav, transcribe_audio
from app.services.whatsapp import download_media, get_media_url, send_text_message

logger = logging.getLogger(__name__)

PROCESSING_MESSAGE = "⏳ Processing your voice note..."

_ERROR_MESSAGE = (
    "😔 Sorry, I couldn't process that voice note. Please try again in a "
    "moment — if the problem continues, try sending a shorter or clearer "
    "recording."
)

_EMPTY_TRANSCRIPT_MESSAGE = (
    "🤔 I couldn't make out any speech in that voice note. Could you try "
    "sending it again?"
)


async def process_voice_note(media_id: str, sender: str) -> None:
    """
    Full pipeline for a single incoming voice/audio message.

    Args:
        media_id: The WhatsApp media ID from the incoming message payload
            (see `app/routers/webhook.py::_parse_single_message`).
        sender: The sender's WhatsApp phone number, used to send both the
            "processing" notice and the final reply (or error message).
    """
    # Let the user know we're working on it immediately — transcription +
    # cleanup take a few real seconds, unlike Phase 1's instant text echo,
    # so silence here would feel broken even though it's working fine.
    await send_text_message(sender, PROCESSING_MESSAGE)

    try:
        cleaned_text, detected_language = await _download_transcribe_and_clean(media_id)
    except Exception:
        logger.exception(
            "Voice note processing failed | media_id=%s | sender=%s",
            media_id,
            sender,
        )
        await send_text_message(sender, _ERROR_MESSAGE)
        return

    if not cleaned_text.strip():
        logger.warning("Transcription produced empty text | media_id=%s", media_id)
        await send_text_message(sender, _EMPTY_TRANSCRIPT_MESSAGE)
        return

    # Use the *effective* language (e.g. Whisper's "hindi" mislabel is
    # corrected to "urdu" — see resolve_effective_language) so the label
    # shown to the user matches the script actually used in cleaned_text.
    display_language = resolve_effective_language(detected_language)
    reply = f"🌐 Language: {display_language.title()}\n\n{cleaned_text}"
    await send_text_message(sender, reply)


async def _download_transcribe_and_clean(media_id: str) -> tuple[str, str]:
    """
    Download, convert, transcribe, and clean up a single voice note.

    Returns a (cleaned_text, detected_language) tuple — the detected
    language is surfaced back to `process_voice_note` so it can be shown
    to the user alongside the cleaned transcript.

    Raises on any failure (network error, ffmpeg failure, Groq API error,
    etc.) — the caller (`process_voice_note`) is responsible for turning
    that into a single user-facing error message, so we deliberately don't
    catch anything here.
    """
    media_url = await get_media_url(media_id)
    audio_bytes = await download_media(media_url)

    # Use a temp directory (auto-cleaned on exit, even on exceptions)
    # rather than juggling individual temp file cleanup ourselves.
    with tempfile.TemporaryDirectory(prefix="voicenotes_") as tmp_dir:
        tmp_path = Path(tmp_dir)

        # ffmpeg detects the actual codec from the file contents, not the
        # extension, so a fixed placeholder name is fine even though we
        # don't know the exact original filename/extension from WhatsApp.
        input_path = tmp_path / "input.ogg"
        output_path = tmp_path / "converted.wav"

        input_path.write_bytes(audio_bytes)

        await convert_to_wav(input_path, output_path)
        raw_transcript, detected_language = await transcribe_audio(output_path)

    cleaned_text = await cleanup_transcript(raw_transcript, detected_language)
    return cleaned_text, detected_language
