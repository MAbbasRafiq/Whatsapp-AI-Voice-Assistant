"""
Text-to-speech via Microsoft Edge's online TTS (edge-tts).

No API key required. Generates MP3 bytes for WhatsApp audio messages.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tempfile
from dataclasses import dataclass

import edge_tts

logger = logging.getLogger(__name__)

MAX_TTS_CHARS = 3000
TTS_TIMEOUT_SECONDS = 45.0
DEFAULT_VOICE = "en-US-JennyNeural"
DEFAULT_LANGUAGE_KEY = "english"

# Canonical language key → Edge neural voice
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

# Free-text / ISO-ish labels → canonical keys in _VOICE_MAP
_LANGUAGE_ALIASES: dict[str, str] = {
    "en": "english",
    "eng": "english",
    "ur": "urdu",
    "urd": "urdu",
    "ar": "arabic",
    "ara": "arabic",
    "fr": "french",
    "fra": "french",
    "francais": "french",
    "français": "french",
    "de": "german",
    "deu": "german",
    "deutsch": "german",
    "hi": "hindi",
    "hin": "hindi",
    "zh": "chinese",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "mandarin": "chinese",
    "tr": "turkish",
    "tur": "turkish",
    "turkce": "turkish",
    "türkçe": "turkish",
    "ru": "russian",
    "rus": "russian",
    "es": "spanish",
    "spa": "spanish",
    "espanol": "spanish",
    "español": "spanish",
    "it": "italian",
    "ita": "italian",
    "pt": "portuguese",
    "por": "portuguese",
    "brazilian": "portuguese",
    "brazil": "portuguese",
    "ko": "korean",
    "kor": "korean",
    "roman urdu": "roman",
    "roman-urdu": "roman",
}

# Script → canonical key. Japanese kana has no mapped voice → unsupported.
_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF]")
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_HANGUL_RE = re.compile(r"[\uAC00-\uD7AF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_CJK_RE = re.compile(r"[\u4E00-\u9FFF]")
_JAPANESE_KANA_RE = re.compile(r"[\u3040-\u30FF]")


class UnsupportedTtsLanguageError(ValueError):
    """Raised when we cannot pick a suitable Edge voice without guessing wrongly."""

    def __init__(self, language: str) -> None:
        self.language = (language or "").strip() or "that language"
        super().__init__(f"No TTS voice for language={self.language!r}")


@dataclass(frozen=True)
class SpeechResult:
    audio_bytes: bytes
    truncated: bool
    voice: str
    language_key: str


def _normalize_language_key(language: str) -> str:
    key = (language or "").strip().lower()
    if not key:
        return ""
    if key.startswith("roman"):
        return "roman"
    if key in _VOICE_MAP:
        return key
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    # Multi-word lookups (e.g. "Brazilian Portuguese", "Mandarin Chinese")
    # are handled by splitting and matching each part below.
    for part in key.replace("-", " ").split():
        if part in _VOICE_MAP:
            return part
        if part in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[part]
    return key


def detect_language_from_text(text: str) -> str | None:
    """
    Infer a supported language key from script, or None if Latin/unknown.

    Returns the sentinel \"unsupported\" when the script is recognized but
    we have no Edge voice for it (e.g. Japanese kana).
    """
    if not text:
        return None
    if _JAPANESE_KANA_RE.search(text):
        return "unsupported"
    if _ARABIC_SCRIPT_RE.search(text):
        return "urdu"
    if _DEVANAGARI_RE.search(text):
        return "hindi"
    if _HANGUL_RE.search(text):
        return "korean"
    if _CYRILLIC_RE.search(text):
        return "russian"
    if _CJK_RE.search(text):
        return "chinese"
    return None


def resolve_language_key(language: str, text: str = "") -> str:
    """
    Resolve to a canonical _VOICE_MAP key.

    Prefer an explicit supported language label; otherwise infer from
    script. Never silently map an unsupported language onto English.
    """
    normalized = _normalize_language_key(language)
    if normalized in _VOICE_MAP:
        return normalized

    script_key = detect_language_from_text(text)
    if script_key == "unsupported":
        label = language.strip() if (language or "").strip() and language.strip().lower() != "unsupported" else "this text"
        raise UnsupportedTtsLanguageError(label)
    if script_key in _VOICE_MAP:
        return script_key

    # Explicit but unknown label (e.g. "Japanese", "Swahili") — do not
    # guess English; that produced wrong-sounding audio with no warning.
    if normalized and normalized != "unsupported":
        raise UnsupportedTtsLanguageError(language)

    if normalized == "unsupported":
        raise UnsupportedTtsLanguageError("this text")

    return DEFAULT_LANGUAGE_KEY


def _resolve_voice(language_key: str) -> str:
    return _VOICE_MAP.get(language_key, DEFAULT_VOICE)


def _truncate_for_tts(text: str) -> tuple[str, bool]:
    """
    Cap text at MAX_TTS_CHARS, preferring a sentence boundary, then a space.
    Appends "..." when truncated. Returns (spoken_text, was_truncated).
    """
    if len(text) <= MAX_TTS_CHARS:
        return text, False

    logger.warning(
        "TTS text truncated | original_chars=%d | limit=%d",
        len(text),
        MAX_TTS_CHARS,
    )
    window = text[:MAX_TTS_CHARS]
    matches = list(re.finditer(r"[.!?\u06d4]\s", window))
    if matches:
        cut = matches[-1].end()
        return window[:cut].rstrip() + "...", True

    space = window.rfind(" ")
    if space > 0:
        return window[:space].rstrip() + "...", True

    return window.rstrip() + "...", True


async def generate_speech(text: str, language: str) -> SpeechResult:
    """
    Synthesize `text` as MP3 using the voice for `language`.

    Returns SpeechResult. Raises UnsupportedTtsLanguageError when no safe
    voice exists; raises on synthesis/timeout failure otherwise.
    """
    language_key = resolve_language_key(language, text)
    spoken, truncated = _truncate_for_tts(text)
    voice = _resolve_voice(language_key)
    logger.info(
        "TTS voice selected | voice=%s | language=%r | resolved=%s | chars=%d | truncated=%s",
        voice,
        language,
        language_key,
        len(spoken),
        truncated,
    )

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        communicate = edge_tts.Communicate(spoken, voice)
        try:
            await asyncio.wait_for(
                communicate.save(tmp_path),
                timeout=TTS_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"edge-tts timed out after {TTS_TIMEOUT_SECONDS:.0f}s"
            ) from exc

        with open(tmp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        if not audio_bytes:
            raise RuntimeError("edge-tts produced empty audio output")

        logger.info(
            "TTS generated | voice=%s | chars=%d | bytes=%d | truncated=%s",
            voice,
            len(spoken),
            len(audio_bytes),
            truncated,
        )
        return SpeechResult(
            audio_bytes=audio_bytes,
            truncated=truncated,
            voice=voice,
            language_key=language_key,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                logger.warning("Failed to delete TTS temp file | path=%s", tmp_path)
