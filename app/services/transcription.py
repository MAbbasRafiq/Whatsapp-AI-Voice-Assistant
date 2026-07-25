"""
Audio conversion + speech-to-text transcription for voice notes.

WhatsApp voice notes arrive as OGG-container files using the Opus codec.
Groq's Whisper-compatible endpoint lists "ogg" as a supported format, but
OGG/Opus specifically has had recurring reliability reports on
Whisper-family APIs (silent failures on otherwise-valid files). To stay
safe, we always convert to WAV via ffmpeg before sending audio off for
transcription. ffmpeg itself is provided by the `imageio-ffmpeg` pip
package, which bundles a static binary — no manual ffmpeg install or PATH
setup required on any platform.
"""

import asyncio
import logging
import subprocess
from pathlib import Path

import imageio_ffmpeg
from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

# Model IDs are plain strings (not enums) per Groq's API — see
# https://console.groq.com/docs/speech-to-text and /docs/models.
#
# We initially used whisper-large-v3-turbo (faster/cheaper), but it's a
# pruned/distilled model with noticeably worse multilingual accuracy and
# language detection than the full model — in testing it mis-detected
# Urdu speech as English and transcribed it as English text entirely
# (not even a deliberate translation, just a language-ID failure). The
# full whisper-large-v3 model is slower but far more reliable for
# non-English/mixed-language audio, which this app depends on.
WHISPER_MODEL = "whisper-large-v3"

_groq_client: AsyncGroq | None = None


def _get_groq_client() -> AsyncGroq:
    """
    Lazily construct the Groq client on first use, rather than at import
    time. This avoids the client erroring out on missing/empty API keys
    before `app.config.settings` has had a chance to load `.env` — and
    means a single import of this module doesn't require GROQ_API_KEY to
    already be set (useful for Phase 1-only test runs, tooling, etc.).
    """
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


async def convert_to_wav(input_path: Path, output_path: Path) -> None:
    """
    Convert an input audio file (e.g. WhatsApp's .ogg/opus voice note) to
    a 16kHz mono WAV file using ffmpeg.

    - WAV is one of the most universally supported input formats across
      Whisper-family transcription APIs.
    - 16kHz mono matches what speech models are trained on — downsampling
      keeps the converted file small without losing anything useful for
      transcription (speech doesn't need stereo or high sample rates).

    Runs the (blocking) ffmpeg subprocess in a worker thread via
    `asyncio.to_thread` so it never blocks the event loop — important
    since this app also needs to keep handling other webhook requests
    while a conversion is in progress.

    Raises `subprocess.CalledProcessError` if ffmpeg fails (e.g. corrupt
    or unrecognized audio data).
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",  # overwrite output file if it somehow already exists
        "-i", str(input_path),
        "-ar", "16000",  # 16kHz sample rate
        "-ac", "1",  # mono
        str(output_path),
    ]

    def _run_ffmpeg() -> None:
        subprocess.run(cmd, capture_output=True, check=True)

    try:
        await asyncio.to_thread(_run_ffmpeg)
    except subprocess.CalledProcessError as exc:
        stderr_text = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        logger.error("ffmpeg conversion failed (exit=%s): %s", exc.returncode, stderr_text)
        raise


async def transcribe_audio(audio_path: Path) -> tuple[str, str]:
    """
    Transcribe an audio file using Groq's Whisper-compatible API
    (model: whisper-large-v3 — see WHISPER_MODEL comment above for why
    we don't use the turbo variant).

    Language is auto-detected by Whisper — we don't pass a `language`
    parameter — so English, Urdu, and mixed/code-switched speech are all
    handled automatically without extra configuration.

    Returns a (text, detected_language) tuple. `detected_language` is
    Whisper's own language guess (e.g. "english", "urdu") and is passed
    downstream to `app.services.llm.cleanup_transcript`, which needs it to
    keep the cleaned-up output in the *same* language/script rather than
    letting the cleanup LLM drift into English or another language.
    """
    client = _get_groq_client()

    # Passing a Path lets the async client read the file contents itself
    # (asynchronously) rather than us opening it with blocking file I/O.
    transcription = await client.audio.transcriptions.create(
        file=audio_path,
        model=WHISPER_MODEL,
        response_format="verbose_json",  # exposes .text and .language
    )

    detected_language = getattr(transcription, "language", "unknown") or "unknown"
    text = transcription.text or ""
    logger.info(
        "Transcription complete | detected_language=%s | chars=%d",
        detected_language,
        len(text),
    )

    return text, detected_language
