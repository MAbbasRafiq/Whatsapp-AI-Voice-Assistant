"""
Text-to-speech via Microsoft Edge's online TTS (edge-tts).

No API key required. Generates MP3 bytes for WhatsApp audio messages.
"""

import logging
import os
import re
import tempfile

import edge_tts

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 3000
DEFAULT_VOICE = "en-US-JennyNeural"

# language key (lowercase) → Edge neural voice
_VOICE_MAP: dict[str, str] = {
    "english": "en-US-JennyNeural",
    "urdu": "ur-PK-AsadNeural",
    "roman": "en-US-JennyNeural",
    "arabic": "ar-SA-HamedNeural",
    "french": "fr-FR-DeniseNeural",
    "german": "de-DE-KatjaNeural",
    "hindi": "hi-IN-SwaraNeural",
    "chinese": "zh-CN-XiaoxiaoNeural",
    "turkish": "tr-TR-EmelNeural",
    "russian": "ru-RU-SvetlanaNeural",
    "spanish": "es-ES-ElviraNeural",
    "italian": "it-IT-ElsaNeural",
    "portuguese": "pt-BR-FranciscaNeural",
    "korean": "ko-KR-SunHiNeural",
}


def _resolve_voice(language: str) -> str:
    """Map a language label to an Edge voice; never fail — unknown → default."""
    key = (language or "").strip().lower()
    if key.startswith("roman"):
        return _VOICE_MAP["roman"]
    if key in _VOICE_MAP:
        return _VOICE_MAP[key]
    # e.g. "Roman Urdu" already handled; "Mandarin Chinese" → try first token
    first = key.split()[0] if key else ""
    return _VOICE_MAP.get(first, DEFAULT_VOICE)


def _truncate_for_tts(text: str) -> str:
    """
    Cap text at MAX_TTS_CHARS, preferring a sentence boundary, then a space.
    Appends "..." when truncated.
    """
    if len(text) <= MAX_TTS_CHARS:
        return text

    logger.warning(
        "TTS text truncated | original_chars=%d | limit=%d",
        len(text),
        MAX_TTS_CHARS,
    )
    window = text[:MAX_TTS_CHARS]
    # Prefer last sentence end (. ! ? Urdu full stop) inside the window.
    matches = list(re.finditer(r"[.!?\u06d4]\s", window))
    if matches:
        cut = matches[-1].end()
        return window[:cut].rstrip() + "..."

    space = window.rfind(" ")
    if space > 0:
        return window[:space].rstrip() + "..."

    return window.rstrip() + "..."


async def generate_speech(text: str, language: str) -> bytes:
    """
    Synthesize `text` as MP3 using the voice for `language`.

    Returns raw MP3 bytes. Raises on failure — callers turn that into a
    user-facing error. Temp files are always cleaned up.
    """
    spoken = _truncate_for_tts(text)
    voice = _resolve_voice(language)
    logger.info(
        "TTS voice selected | voice=%s | language=%r | chars=%d",
        voice,
        language,
        len(spoken),
    )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        communicate = edge_tts.Communicate(spoken, voice)
        await communicate.save(tmp_path)

        with open(tmp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        if not audio_bytes:
            raise RuntimeError("edge-tts produced empty audio output")

        logger.info(
            "TTS generated | voice=%s | chars=%d | bytes=%d",
            voice,
            len(spoken),
            len(audio_bytes),
        )
        return audio_bytes
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Failed to delete TTS temp file | path=%s", tmp_path)
