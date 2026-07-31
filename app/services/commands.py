"""
Text-command and interactive-message handling.

Primary UX is WhatsApp Reply Buttons + List Messages:
  - After voice transcripts: Translate / Summarize / Help
  - After summary/translation: Listen
  - After eligible plain text: Convert to Voice

`/voice <text>` remains for backward compatibility only and is not
advertised in normal UX. Interactive button/list replies are routed
through `handle_interactive`. Plain text while `waiting_for_language`
is handled by `handle_language_input`.
"""

import logging
import re

from app.config import settings
from app.services.llm import (
    INVALID_LANGUAGE,
    resolve_effective_language,
    summarize_transcript,
    translate_transcript,
)
from app.services.rate_limiter import release_tts, try_acquire_tts
from app.services.text_to_voice_cache import get_pending_text, set_pending_text
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
    consume_waiting_for_language,
    clear_waiting_for_language,
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
BTN_TEXT_TO_VOICE = "text_to_voice"

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
    "⌛ Your previous transcript has expired.\n"
    "For your privacy, transcripts are kept for only 15 minutes.\n"
    "🎤 Send a new voice note to continue."
)

_ASK_OTHER_LANGUAGE = "🌍 Type the language you'd like to translate into."


def _invalid_language_message(user_input: str) -> str:
    return f'⚠️ "{user_input}" isn\'t a recognized language. Enter a valid language name.'

DEFAULT_REPLY = (
    "🎤 Send a voice note to get a clean transcript.\n\n"
    "Use the buttons to *Translate*, *Summarize*, or get *Help*.\n\n"
    "🔊 Or send any text to convert it into voice."
)


# Shown only when Reply Buttons cannot be delivered (Graph API rejection).
_BUTTONS_UNAVAILABLE_FOOTER = (
    "──────────────\n"
    "💡 Buttons unavailable right now — send another voice note when ready.\n"
    "Or send any text to convert it to voice."
)

_HELP_TEXT_TEMPLATE = """\
🤖 Voice Assistant
• Send a voice note to receive a transcript.
• Tap *Translate* to convert it into another language.
• Tap *Summarize* to generate key points.
• Tap *Listen* after a translation or summary to hear it as audio.
• Send any text and tap *Convert to Voice* to hear it spoken.
• Transcripts expire after 15 minutes.

💬 Feedback or issues: abbasrafiq82@gmail.com"""
# Max 3 min per note • Max {daily_limit} notes/day"""

_TEXT_TO_VOICE_PROMPT = "What would you like to do with this text?"
_TEXT_TO_VOICE_BUTTONS: list[tuple[str, str]] = [
    (BTN_TEXT_TO_VOICE, "🔊 Convert to Voice"),
]

_TTS_EXPIRED_REPLY = (
    "⌛ The audio has expired (15 min).\n"
    "Request a new summary or translation."
)

_TEXT_TO_VOICE_EXPIRED_REPLY = (
    "⌛ That text has expired (15 min).\n"
    "Send the text again to convert it to voice."
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

_TTS_BUSY_REPLY = "⏳ Audio is already being generated. Wait a moment."

_TTS_RATE_LIMITED_REPLY = (
    "⏳ Wait a few seconds before generating more audio."
)

_TTS_TRUNCATED_NOTICE = (
    "⚠️ Note: the audio was shortened to fit the length limit."
)

_TTS_UNSUPPORTED_LANGUAGE_TEMPLATE = (
    "⚠️ I can't generate speech for {language} yet.\n"
)


def _command_name(text: str) -> str:
    stripped = text.strip().lstrip("/")
    if not stripped:
        return ""
    return stripped.lower().split()[0]


def meaningful_char_count(text: str) -> int:
    """
    Count Unicode letters/numbers only (local, no I/O).

    Spaces, punctuation, symbols, and emojis are ignored.
    """
    return sum(1 for ch in text if ch.isalnum())


def is_command(text: str) -> bool:
    """
    True only for `/voice` (and legacy `/tts`) — kept for backward
    compatibility, not advertised in normal UX.
    """
    return _command_name(text) in ("voice", "tts")


async def handle_plain_text(wa_phone: str, text: str) -> bool:
    """
    Handle a normal (non-command, non-waiting) text message.

    Short texts (< 5 meaningful characters) get the existing nudge.
    Longer texts are cached and offered a Convert to Voice button.

    Returns True if a reply was sent successfully, False if sending failed.
    """
    stripped = text.strip()
    if meaningful_char_count(stripped) < 5:
        return await send_text_message(wa_phone, DEFAULT_REPLY)

    set_pending_text(wa_phone, stripped)
    sent = await send_reply_buttons(
        wa_phone,
        body=_TEXT_TO_VOICE_PROMPT,
        buttons=_TEXT_TO_VOICE_BUTTONS,
    )
    if sent:
        return True

    # Interactive buttons unavailable — keep the text cached and tell the
    # user what happened (no /voice advertising).
    logger.warning(
        "Convert-to-Voice buttons failed; sending text fallback | wa_phone=%s",
        mask_phone(wa_phone),
    )
    return await send_text_message(
        wa_phone,
        f"{_TEXT_TO_VOICE_PROMPT}\n\n"
        "Buttons unavailable right now — please try again in a moment.",
    )


async def send_post_transcript_actions(wa_phone: str) -> bool:
    """
    Send the three Reply Buttons that follow every successful transcript.

    Returns True if the interactive message was accepted. Callers should
    fall back to `slash_command_fallback_footer()` when this returns False.
    """
    return await send_reply_buttons(
        wa_phone,
        body="What would you like to do next?",
        buttons=_POST_TRANSCRIPT_BUTTONS,
    )


def slash_command_fallback_footer() -> str:
    """Text footer used when Reply Buttons cannot be delivered."""
    return _BUTTONS_UNAVAILABLE_FOOTER


def _help_text() -> str:
    return _HELP_TEXT_TEMPLATE


async def handle_command(wa_phone: str, text: str) -> None:
    """
    Handle the only supported slash command: `/voice` (plus `/tts` rename).
    """
    command = _command_name(text)

    try:
        if command == "voice":
            await _handle_tts_command(wa_phone, text)
        elif command == "tts":
            sent = await send_text_message(wa_phone, _TTS_RENAMED_REPLY)
            if not sent:
                logger.error(
                    "Failed to send /tts rename reply | wa_phone=%s",
                    mask_phone(wa_phone),
                )
        else:
            # Should not be reached when is_command() gates the webhook.
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
            sent = await send_text_message(wa_phone, _help_text())
            if not sent:
                logger.error("Failed to send Help reply | wa_phone=%s", mask_phone(wa_phone))
        elif reply_id == BTN_TTS_PLAY:
            await _handle_tts_play(wa_phone)
        elif reply_id == BTN_TEXT_TO_VOICE:
            await _handle_text_to_voice(wa_phone)
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

    Atomically claims the waiting flag so concurrent text deliveries cannot
    both translate. If the flag was already claimed/expired, tells them to
    start Translate again. On empty language input, re-enters waiting mode
    so they can try again.
    """
    if not consume_waiting_for_language(wa_phone):
        await send_user_error(wa_phone, ErrorType.WAITING_EXPIRED)
        return

    language = language_text.strip()
    if not language:
        sent = await send_user_error(wa_phone, ErrorType.EMPTY_LANGUAGE)
        if sent:
            set_waiting_for_language(wa_phone)
        return

    # Accidental "/spanish"-style input while waiting: strip the slash and
    # treat the rest as a language name. /voice is routed earlier in the
    # webhook and never reaches this handler.
    if language.startswith("/"):
        language = language.lstrip("/").strip()
        if not language:
            sent = await send_user_error(wa_phone, ErrorType.EMPTY_LANGUAGE)
            if sent:
                set_waiting_for_language(wa_phone)
            return

    try:
        await _translate_last(wa_phone, language, waiting_for_retry=True)
    except Exception as exc:
        logger.exception(
            "Language-input translation failed | wa_phone=%s | language=%r",
            mask_phone(wa_phone),
            language,
        )
        await send_user_error(wa_phone, classify_for_request(exc))


async def _start_translate_flow(wa_phone: str) -> None:
    """Show the language list, or fall back to an English translation."""
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
        # Interactive list unavailable — still deliver a useful result.
        logger.warning(
            "List message failed; falling back to English translate | wa_phone=%s",
            mask_phone(wa_phone),
        )
        notice_sent = await send_text_message(
            wa_phone,
            "Language menu unavailable right now. Translating to English…",
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
    """Send Reply Buttons after a successful result, with text fallback."""
    buttons_sent = await send_post_transcript_actions(wa_phone)
    if not buttons_sent:
        logger.warning(
            "Reply buttons failed after %s; sending text fallback | wa_phone=%s",
            context,
            mask_phone(wa_phone),
        )
        fallback_sent = await send_text_message(wa_phone, slash_command_fallback_footer())
        if not fallback_sent:
            logger.error(
                "Failed to send buttons-unavailable fallback after %s | wa_phone=%s",
                context,
                mask_phone(wa_phone),
            )


async def _translate_last(
    wa_phone: str,
    target_language: str,
    *,
    waiting_for_retry: bool = False,
) -> None:
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

    if translated.strip() == INVALID_LANGUAGE:
        # Free-text "Other language..." path: re-enter waiting so the user
        # can type another language name without tapping Translate again.
        sent = await send_text_message(wa_phone, _invalid_language_message(target_language))
        if waiting_for_retry and sent:
            set_waiting_for_language(wa_phone)
        else:
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
    # Always summarize in the transcript's own language — never auto-translate.
    # Translation only happens when the user taps Translate.
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


async def _handle_text_to_voice(wa_phone: str) -> None:
    """Handle the Convert to Voice button for cached plain text."""
    text = get_pending_text(wa_phone)
    if text is None:
        sent = await send_text_message(wa_phone, _TEXT_TO_VOICE_EXPIRED_REPLY)
        if not sent:
            logger.error(
                "Failed to send text-to-voice expired reply | wa_phone=%s",
                mask_phone(wa_phone),
            )
        return

    language = _detect_tts_language(text)
    try:
        await _generate_and_send_audio(wa_phone, text, language)
    except Exception as exc:
        logger.exception(
            "Text-to-voice failed | wa_phone=%s | language=%r",
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
