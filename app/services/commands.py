"""
Text-command and interactive-message handling.

Primary UX is WhatsApp Reply Buttons + List Messages after each
transcript. Slash commands (`/translate`, `/summarize`, `/help`, etc.)
remain as a fallback when interactive sends fail or a user types them
directly.

Any incoming text message starting with "/" is treated as a command
(case-insensitive, surrounding whitespace ignored). Interactive button/
list replies are routed through `handle_interactive`. Plain text while
`waiting_for_language` is handled by `handle_language_input`.
"""

import logging
import re

from app.config import settings
from app.services import db_ops
from app.services.llm import resolve_effective_language, summarize_transcript, translate_transcript
from app.services.rate_limiter import release_tts, try_acquire_tts
from app.services.transcript_cache import get_last_transcript
from app.services.tts import UnsupportedTtsLanguageError, detect_language_from_text, generate_speech
from app.services.tts_cache import get_tts_cache, set_tts_cache
from app.services.user_errors import (
    ErrorType,
    classify_for_request,
    classify_for_translate,
    send_user_error,
)
from app.services.user_state import (
    clear_waiting_for_language,
    is_waiting_for_language,
    set_waiting_for_language,
)
from app.services.whatsapp import (
    send_audio_message,
    send_list_message,
    send_listen_button,
    send_long_message,
    send_reply_buttons,
    send_text_message,
    upload_media,
)
from app.utils import mask_phone

logger = logging.getLogger(__name__)

# --- Interactive message IDs (must match what we send) --------------------
BTN_TRANSLATE = "action_translate"
BTN_SUMMARIZE = "action_summarize"
BTN_HELP = "action_help"
BTN_TTS_PLAY = "tts_play"

LANG_ENGLISH = "lang_english"
LANG_URDU = "lang_urdu"
LANG_ROMAN = "lang_roman"
LANG_ARABIC = "lang_arabic"
LANG_FRENCH = "lang_french"
LANG_CHINESE = "lang_chinese"
LANG_OTHER = "lang_other"

_LANG_ID_TO_TARGET: dict[str, str] = {
    LANG_ENGLISH: "English",
    LANG_URDU: "Urdu",
    LANG_ROMAN: "Roman Urdu",
    LANG_ARABIC: "Arabic",
    LANG_FRENCH: "French",
    LANG_CHINESE: "Chinese",
}

_POST_TRANSCRIPT_BUTTONS: list[tuple[str, str]] = [
    (BTN_TRANSLATE, "🌍 Translate"),
    (BTN_SUMMARIZE, "📝 Summarize"),
    (BTN_HELP, "❓ Help"),
]

_LANGUAGE_LIST_ROWS: list[tuple[str, str]] = [
    (LANG_ENGLISH, "English"),
    (LANG_URDU, "Urdu"),
    (LANG_ROMAN, "🔤 Roman Urdu"),
    (LANG_ARABIC, "Arabic"),
    (LANG_FRENCH, "French"),
    (LANG_CHINESE, "Chinese"),
    (LANG_OTHER, "✍️ Other language..."),
]

# Kept as a public alias — same wording as ErrorType.NO_TRANSCRIPT.
NO_TRANSCRIPT_REPLY = (
    "I couldn't find a recent transcript.\n"
    "Remember voice expires after 15 min\n"
    "Send a new voice note."
)

_ASK_OTHER_LANGUAGE = "🌍 Type the language you'd like to translate into."

DEFAULT_REPLY = (
    "🎤 Send me a voice note and I'll convert it into clean text.\n\n"
    "🔊 Want to convert text into speech?\n"
    "Use:\n"
    "`/voice Your text here`\n\n"
    "Example:\n"
    "`/voice Hello everyone`"
)

_SLASH_FALLBACK_FOOTER = "──────────────\n💡 /translate • /summarize • /roman • /help"

_PREFERENCE_SAVED_REPLIES = {
    "urdu": "✅ Preference saved! I'll use اردو for all future transcripts.",
    "english": "✅ Preference saved! I'll use English for all future transcripts.",
    "roman": "✅ Preference saved! I'll use Roman Urdu for all future transcripts.",
}

_HELP_TEXT = """\
🤖 Voice Assistant
• Send a voice note to receive a transcript.
• Use Translate to convert it into another language.
• Use Summarize to generate key points.
• Use /voice <text> to convert any text into speech.
• The voice message expires after 15 min

To translate into any language not listed,
choose Other language... and simply type the language name."""

_HELP_TEXT_SLASH_TEMPLATE = """\
🎤 VoiceNotes — Commands

/translate — Translate last transcript to English
/summarize — Summarize last transcript
/voice <text> — Convert any text to voice
/urdu — Set language to Urdu (اردو)
/english — Set language to English
/roman — Set language to Roman Urdu
/help — Show this message

Send any voice note to get started!
Max 3 min • Max {daily_limit} notes/day"""

_TTS_EXPIRED_REPLY = (
    "⌛ The audio has expired (15 min).\n"
    "Please request a new summary or translation."
)

_TTS_FAILED_REPLY = (
    "⚠️ I couldn't generate audio right now.\n"
    "Please try again in a moment."
)

_TTS_USAGE_REPLY = (
    "Please provide text after /voice\n"
    "Example: /voice Hello everyone"
)

_TTS_RENAMED_REPLY = (
    "🔄 /tts has been renamed to /voice\n\n"
    "Example: /voice Hello everyone"
)

_TTS_BUSY_REPLY = "⏳ Audio is already being generated. Please wait a moment."

_TTS_RATE_LIMITED_REPLY = (
    "⏳ Please wait a few seconds before generating more audio."
)

_TTS_TRUNCATED_NOTICE = (
    "⚠️ Note: the audio was shortened to fit the length limit."
)

_TTS_UNSUPPORTED_LANGUAGE_TEMPLATE = (
    "⚠️ I can't generate speech for {language} yet.\n"
    "Supported: English, Urdu, Roman Urdu, Arabic, French, German, "
    "Hindi, Chinese, Turkish, Russian, Spanish, Italian, Portuguese, Korean."
)


def is_command(text: str) -> bool:
    """True if `text` looks like a bot command (starts with '/')."""
    return text.strip().startswith("/")


async def send_post_transcript_actions(wa_phone: str) -> bool:
    """
    Send the three Reply Buttons that follow every successful transcript.

    Returns True if the interactive message was accepted. Callers should
    fall back to the slash-command footer text when this returns False.
    """
    return await send_reply_buttons(
        wa_phone,
        body="What would you like to do next?",
        buttons=_POST_TRANSCRIPT_BUTTONS,
    )


def slash_command_fallback_footer() -> str:
    """Text footer used when Reply Buttons cannot be delivered."""
    return _SLASH_FALLBACK_FOOTER


async def handle_command(wa_phone: str, text: str) -> None:
    """
    Parse and execute a single slash-command message.

    Unknown "/whatever" commands fall back to the same generic nudge as
    non-command text, rather than a confusing silent no-op.
    """
    command = text.strip().lower().lstrip("/").split()[0] if text.strip().lstrip("/") else ""

    try:
        if command == "translate":
            await _translate_last(wa_phone, "English")
        elif command == "summarize":
            await _summarize_last(wa_phone)
        elif command == "voice":
            await _handle_tts_command(wa_phone, text)
        elif command == "tts":
            sent = await send_text_message(wa_phone, _TTS_RENAMED_REPLY)
            if not sent:
                logger.error(
                    "Failed to send /tts rename reply | wa_phone=%s",
                    mask_phone(wa_phone),
                )
        elif command in ("urdu", "english", "roman"):
            await _handle_set_preference(wa_phone, command)
        elif command == "help":
            help_text = _HELP_TEXT_SLASH_TEMPLATE.format(daily_limit=settings.daily_voice_limit)
            sent = await send_text_message(wa_phone, help_text)
            if not sent:
                logger.error("Failed to send /help reply | wa_phone=%s", mask_phone(wa_phone))
        else:
            sent = await send_text_message(wa_phone, DEFAULT_REPLY)
            if not sent:
                logger.error("Failed to send default command reply | wa_phone=%s", mask_phone(wa_phone))
    except Exception as exc:
        logger.exception(
            "Command handling failed | wa_phone=%s | command=%s",
            mask_phone(wa_phone),
            command,
        )
        await send_user_error(wa_phone, classify_for_request(exc))


async def handle_interactive(wa_phone: str, reply_id: str, reply_title: str = "") -> None:
    """
    Handle a WhatsApp interactive button_reply or list_reply.

    `reply_id` is the stable id we set when sending the interactive
    message; `reply_title` is only used for logging / unknown-id fallback.
    """
    try:
        if reply_id == BTN_TRANSLATE:
            await _start_translate_flow(wa_phone)
        elif reply_id == BTN_SUMMARIZE:
            await _summarize_last(wa_phone)
        elif reply_id == BTN_HELP:
            sent = await send_text_message(wa_phone, _HELP_TEXT)
            if not sent:
                logger.error("Failed to send Help reply | wa_phone=%s", mask_phone(wa_phone))
        elif reply_id == BTN_TTS_PLAY:
            await _handle_tts_play(wa_phone)
        elif reply_id == LANG_OTHER:
            await _prompt_other_language(wa_phone)
        elif reply_id in _LANG_ID_TO_TARGET:
            await _translate_last(wa_phone, _LANG_ID_TO_TARGET[reply_id])
        else:
            logger.warning(
                "Unknown interactive reply id=%r title=%r | wa_phone=%s",
                reply_id,
                reply_title,
                mask_phone(wa_phone),
            )
            await send_user_error(wa_phone, ErrorType.UNEXPECTED_INTERACTIVE)
    except Exception as exc:
        logger.exception(
            "Interactive handling failed | wa_phone=%s | reply_id=%s",
            mask_phone(wa_phone),
            reply_id,
        )
        await send_user_error(wa_phone, classify_for_request(exc))


async def handle_language_input(wa_phone: str, language_text: str) -> None:
    """
    Handle the plain-text reply that follows "Other language...".

    Clears pending state whether translation succeeds or fails, so the
    user isn't stuck in waiting mode. If the waiting flag already expired,
    tells them to start Translate again.
    """
    if not is_waiting_for_language(wa_phone):
        await send_user_error(wa_phone, ErrorType.WAITING_EXPIRED)
        return

    clear_waiting_for_language(wa_phone)

    language = language_text.strip()
    if not language:
        sent = await send_user_error(wa_phone, ErrorType.EMPTY_LANGUAGE)
        if sent:
            set_waiting_for_language(wa_phone)
        return

    # Ignore accidental slash-looking input while waiting — treat the
    # whole string as a language name (e.g. user typed "/spanish" by habit).
    if language.startswith("/"):
        language = language.lstrip("/").strip()
        if not language:
            sent = await send_user_error(wa_phone, ErrorType.EMPTY_LANGUAGE)
            if sent:
                set_waiting_for_language(wa_phone)
            return

    try:
        await _translate_last(wa_phone, language)
    except Exception as exc:
        logger.exception(
            "Language-input translation failed | wa_phone=%s | language=%r",
            mask_phone(wa_phone),
            language,
        )
        await send_user_error(wa_phone, classify_for_request(exc))


async def _start_translate_flow(wa_phone: str) -> None:
    """Show the language list, or fall back to slash-style English translate."""
    if get_last_transcript(wa_phone) is None:
        await send_user_error(wa_phone, ErrorType.NO_TRANSCRIPT)
        return

    sent = await send_list_message(
        wa_phone,
        body="Pick a language for the translation:",
        button_text="Languages",
        rows=_LANGUAGE_LIST_ROWS,
        header="Choose target language",
        section_title="Languages",
    )
    if not sent:
        # Interactive list unavailable — fall back to English translate
        # (same as legacy /translate) so the user still gets a result.
        logger.warning(
            "List message failed; falling back to /translate English | wa_phone=%s",
            mask_phone(wa_phone),
        )
        notice_sent = await send_text_message(
            wa_phone,
            "Interactive menus unavailable. Translating to English…\n"
            f"(Or use {_SLASH_FALLBACK_FOOTER})",
        )
        if not notice_sent:
            logger.error(
                "Failed to send list-fallback notice | wa_phone=%s",
                mask_phone(wa_phone),
            )
        await _translate_last(wa_phone, "English")


async def _prompt_other_language(wa_phone: str) -> None:
    if get_last_transcript(wa_phone) is None:
        clear_waiting_for_language(wa_phone)
        await send_user_error(wa_phone, ErrorType.NO_TRANSCRIPT)
        return

    # Only enter waiting mode after the prompt is actually delivered —
    # otherwise the user is stuck treating their next message as a language
    # name without ever seeing the ask.
    sent = await send_text_message(wa_phone, _ASK_OTHER_LANGUAGE)
    if not sent:
        logger.error(
            "Failed to send 'type a language' prompt; not entering waiting mode | wa_phone=%s",
            mask_phone(wa_phone),
        )
        return
    set_waiting_for_language(wa_phone)


async def _send_post_action_buttons(wa_phone: str, *, context: str) -> None:
    """Send Reply Buttons after a successful result, with slash-footer fallback."""
    buttons_sent = await send_post_transcript_actions(wa_phone)
    if not buttons_sent:
        logger.warning(
            "Reply buttons failed after %s; sending slash-command fallback | wa_phone=%s",
            context,
            mask_phone(wa_phone),
        )
        fallback_sent = await send_text_message(wa_phone, slash_command_fallback_footer())
        if not fallback_sent:
            logger.error(
                "Failed to send slash-command fallback after %s | wa_phone=%s",
                context,
                mask_phone(wa_phone),
            )


async def _translate_last(wa_phone: str, target_language: str) -> None:
    cached = get_last_transcript(wa_phone)
    if cached is None:
        clear_waiting_for_language(wa_phone)
        await send_user_error(wa_phone, ErrorType.NO_TRANSCRIPT)
        return

    text, language = cached
    try:
        translated = await translate_transcript(text, language, target_language=target_language)
    except Exception as exc:
        logger.exception(
            "translate_transcript failed | wa_phone=%s | target=%r",
            mask_phone(wa_phone),
            target_language,
        )
        await send_user_error(wa_phone, classify_for_translate(exc))
        clear_waiting_for_language(wa_phone)
        return

    if not translated.strip():
        await send_user_error(wa_phone, ErrorType.UNSUPPORTED_LANGUAGE)
        clear_waiting_for_language(wa_phone)
        return

    clear_waiting_for_language(wa_phone)
    translated_sent = await send_long_message(wa_phone, translated)
    if not translated_sent:
        logger.error(
            "Failed to deliver translation | wa_phone=%s | target=%r",
            mask_phone(wa_phone),
            target_language,
        )
        await send_user_error(wa_phone, ErrorType.WHATSAPP_TEMP_FAILURE)
        return

    set_tts_cache(wa_phone, translated, target_language)
    await send_listen_button(wa_phone)
    await _send_post_action_buttons(wa_phone, context="translate")


async def _summarize_last(wa_phone: str) -> None:
    cached = get_last_transcript(wa_phone)
    if cached is None:
        await send_user_error(wa_phone, ErrorType.NO_TRANSCRIPT)
        return

    text, language = cached
    # Always summarize in the transcript's own language — never auto-translate
    # via preferred_language. Translation only happens when the user taps
    # Translate (or uses /translate).
    target_language = resolve_effective_language(language)
    try:
        summary = await summarize_transcript(text, target_language)
    except Exception as exc:
        # Classify here so Groq/infra failures get a specific reply instead
        # of only the outer generic request-failed message. The outer
        # try/except in handle_command / handle_interactive remains as a
        # safety net for anything else in this function.
        logger.exception(
            "summarize_transcript failed | wa_phone=%s",
            mask_phone(wa_phone),
        )
        await send_user_error(wa_phone, classify_for_request(exc))
        return

    if not summary.strip():
        await send_user_error(wa_phone, ErrorType.REQUEST_FAILED)
        return

    summary_sent = await send_long_message(wa_phone, summary)
    if not summary_sent:
        logger.error(
            "Failed to deliver summary | wa_phone=%s",
            mask_phone(wa_phone),
        )
        await send_user_error(wa_phone, ErrorType.WHATSAPP_TEMP_FAILURE)
        return

    set_tts_cache(wa_phone, summary, target_language)
    await send_listen_button(wa_phone)
    # Re-offer the same post-transcript Reply Buttons so the user can keep
    # interacting (Translate / Summarize again / Help) without resending.
    await _send_post_action_buttons(wa_phone, context="summarize")


def _detect_tts_language(text: str) -> str:
    """Infer language for /voice from script; default english for Latin text."""
    detected = detect_language_from_text(text)
    if detected == "unsupported":
        # Let generate_speech raise UnsupportedTtsLanguageError with context.
        return "unsupported"
    if detected:
        return detected
    return "english"


async def _generate_and_send_audio(wa_phone: str, text: str, language: str) -> None:
    """
    Shared TTS pipeline: gate → status → synthesize → upload → send audio.

    Raises on synthesis/upload/send failure so callers can show a friendly
    error. Busy / rate-limit rejections are handled here (user message sent,
    no raise).
    """
    gate = try_acquire_tts(wa_phone)
    if gate == "busy":
        sent = await send_text_message(wa_phone, _TTS_BUSY_REPLY)
        if not sent:
            logger.error(
                "Failed to send TTS busy reply | wa_phone=%s",
                mask_phone(wa_phone),
            )
        return
    if gate == "rate_limited":
        sent = await send_text_message(wa_phone, _TTS_RATE_LIMITED_REPLY)
        if not sent:
            logger.error(
                "Failed to send TTS rate-limit reply | wa_phone=%s",
                mask_phone(wa_phone),
            )
        return

    try:
        status_sent = await send_text_message(wa_phone, "🔊 Generating audio...")
        if not status_sent:
            logger.warning(
                "Failed to send TTS status message | wa_phone=%s",
                mask_phone(wa_phone),
            )

        result = await generate_speech(text, language)
        if result.truncated:
            notice_sent = await send_text_message(wa_phone, _TTS_TRUNCATED_NOTICE)
            if not notice_sent:
                logger.warning(
                    "Failed to send TTS truncation notice | wa_phone=%s",
                    mask_phone(wa_phone),
                )

        media_id = await upload_media(result.audio_bytes)
        audio_sent = await send_audio_message(wa_phone, media_id)
        if not audio_sent:
            raise RuntimeError("WhatsApp rejected audio message send")
    finally:
        release_tts(wa_phone)


async def _reply_tts_failure(wa_phone: str, exc: BaseException) -> None:
    if isinstance(exc, UnsupportedTtsLanguageError):
        message = _TTS_UNSUPPORTED_LANGUAGE_TEMPLATE.format(language=exc.language)
    else:
        message = _TTS_FAILED_REPLY
    sent = await send_text_message(wa_phone, message)
    if not sent:
        logger.error(
            "Failed to send TTS failure reply | wa_phone=%s",
            mask_phone(wa_phone),
        )


async def _handle_tts_play(wa_phone: str) -> None:
    """Handle the Listen Reply Button after a summary/translation."""
    cached = get_tts_cache(wa_phone)
    if cached is None:
        sent = await send_text_message(wa_phone, _TTS_EXPIRED_REPLY)
        if not sent:
            logger.error(
                "Failed to send TTS expired reply | wa_phone=%s",
                mask_phone(wa_phone),
            )
        return

    text, language = cached
    try:
        await _generate_and_send_audio(wa_phone, text, language)
    except Exception as exc:
        logger.exception(
            "TTS play failed | wa_phone=%s | language=%r",
            mask_phone(wa_phone),
            language,
        )
        await _reply_tts_failure(wa_phone, exc)


async def _handle_tts_command(wa_phone: str, raw_text: str) -> None:
    """
    /voice <text> — synthesize arbitrary text. No LLM reply; audio only.
    """
    match = re.match(r"(?i)^/voice\s*(.*)$", raw_text.strip(), flags=re.DOTALL)
    tts_text = (match.group(1) if match else "").strip()
    if not tts_text:
        sent = await send_text_message(wa_phone, _TTS_USAGE_REPLY)
        if not sent:
            logger.error(
                "Failed to send /voice usage reply | wa_phone=%s",
                mask_phone(wa_phone),
            )
        return

    language = _detect_tts_language(tts_text)
    try:
        await _generate_and_send_audio(wa_phone, tts_text, language)
    except Exception as exc:
        logger.exception(
            "/voice failed | wa_phone=%s | language=%r",
            mask_phone(wa_phone),
            language,
        )
        await _reply_tts_failure(wa_phone, exc)


async def _handle_set_preference(wa_phone: str, language: str) -> None:
    await db_ops.set_preferred_language(wa_phone, language)
    sent = await send_text_message(wa_phone, _PREFERENCE_SAVED_REPLIES[language])
    if not sent:
        logger.error(
            "Failed to send preference-saved confirmation | wa_phone=%s | language=%s",
            mask_phone(wa_phone),
            language,
        )
