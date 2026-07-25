"""
Text-command handling.

Any incoming text message starting with "/" is treated as a command
(case-insensitive, surrounding whitespace ignored). Anything else that
isn't a voice/audio note falls through to a generic nudge reply — see
`DEFAULT_REPLY` below, sent by the webhook router directly.
"""

import logging

from app.config import settings
from app.services import db_ops
from app.services.llm import resolve_effective_language, summarize_transcript, translate_transcript
from app.services.transcript_cache import get_last_transcript
from app.services.whatsapp import send_long_message, send_text_message
from app.utils import mask_phone

logger = logging.getLogger(__name__)

NO_TRANSCRIPT_REPLY = "⚠️ No recent transcript found. Please send a voice note first."

_GENERIC_ERROR_REPLY = (
    "⚠️ Something went wrong while processing your request. Please try again in a minute."
)

DEFAULT_REPLY = (
    "🎤 Send me a voice note and I'll convert it to clean text!\n\n"
    "Or use /help to see available commands."
)

_PREFERENCE_SAVED_REPLIES = {
    "urdu": "✅ Preference saved! I'll use اردو for all future transcripts.",
    "english": "✅ Preference saved! I'll use English for all future transcripts.",
    "roman": "✅ Preference saved! I'll use Roman Urdu for all future transcripts.",
}

_HELP_TEXT_TEMPLATE = """\
🎤 VoiceNotes — Commands

/translate — Translate last transcript to English
/summarize — Summarize last transcript
/urdu — Set language to Urdu (اردو)
/english — Set language to English
/roman — Set language to Roman Urdu
/help — Show this message

Send any voice note to get started!
Max 3 min • Max {daily_limit} notes/day"""


def is_command(text: str) -> bool:
    """True if `text` looks like a bot command (starts with '/')."""
    return text.strip().startswith("/")


async def handle_command(wa_phone: str, text: str) -> None:
    """
    Parse and execute a single command message, sending the appropriate
    reply (or replies, for long /translate /summarize output) directly.

    Unknown "/whatever" commands fall back to the same generic nudge as
    non-command text, rather than a confusing silent no-op.
    """
    command = text.strip().lower().lstrip("/").split()[0] if text.strip().lstrip("/") else ""

    try:
        if command == "translate":
            await _handle_translate(wa_phone)
        elif command == "summarize":
            await _handle_summarize(wa_phone)
        elif command in ("urdu", "english", "roman"):
            await _handle_set_preference(wa_phone, command)
        elif command == "help":
            await _handle_help(wa_phone)
        else:
            await send_text_message(wa_phone, DEFAULT_REPLY)
    except Exception:
        logger.exception("Command handling failed | wa_phone=%s | command=%s", mask_phone(wa_phone), command)
        await send_text_message(wa_phone, _GENERIC_ERROR_REPLY)


async def _handle_translate(wa_phone: str) -> None:
    cached = get_last_transcript(wa_phone)
    if cached is None:
        await send_text_message(wa_phone, NO_TRANSCRIPT_REPLY)
        return

    text, language = cached
    translated = await translate_transcript(text, language)
    await send_long_message(wa_phone, translated)


async def _handle_summarize(wa_phone: str) -> None:
    cached = get_last_transcript(wa_phone)
    if cached is None:
        await send_text_message(wa_phone, NO_TRANSCRIPT_REPLY)
        return

    text, language = cached
    preferred_language = await db_ops.get_preferred_language(wa_phone)
    target_language = preferred_language or resolve_effective_language(language)
    summary = await summarize_transcript(text, target_language)
    await send_long_message(wa_phone, summary)


async def _handle_set_preference(wa_phone: str, language: str) -> None:
    await db_ops.set_preferred_language(wa_phone, language)
    await send_text_message(wa_phone, _PREFERENCE_SAVED_REPLIES[language])


async def _handle_help(wa_phone: str) -> None:
    help_text = _HELP_TEXT_TEMPLATE.format(daily_limit=settings.daily_voice_limit)
    await send_text_message(wa_phone, help_text)
