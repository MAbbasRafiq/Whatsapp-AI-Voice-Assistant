"""
LLM-based transcript cleanup.

Raw Whisper transcripts tend to contain filler words ("um", "uh", "like",
"you know"), run-on sentences with little/no punctuation, and no paragraph
structure. This module sends the raw transcript to a Groq-hosted LLM to
clean that up into readable text — without changing the meaning, tone, or
language of what was actually said.
"""

import logging
import re

from groq import AsyncGroq

from app.config import settings

logger = logging.getLogger(__name__)

# Unicode block for Devanagari script (used to detect Hindi-script leftovers
# after cleanup — see the retry loop in cleanup_transcript below). This is a
# deterministic safety net: the cleanup LLM follows the "no Devanagari" rule
# most of the time, but not always, especially for less common words it
# doesn't already know the natural Urdu equivalent for.
_DEVANAGARI_RUN_RE = re.compile(r"[\u0900-\u097F]+")

# One initial attempt + this many corrective retries before giving up and
# returning whatever we have (better to send an almost-all-Urdu message than
# no reply at all).
_MAX_DEVANAGARI_RETRIES = 2

# A larger, more instruction-following model. We initially used the
# smaller/faster llama-3.1-8b-instant, but it repeatedly ignored the
# "do not summarize" instruction and condensed full transcripts down to a
# one-line gist (e.g. turning a 120-character spoken sentence into a
# 6-word summary). llama-3.3-70b-versatile follows the "clean up, don't
# summarize" instruction much more reliably.
CLEANUP_MODEL = "llama-3.3-70b-versatile"

_groq_client: AsyncGroq | None = None


def _get_groq_client() -> AsyncGroq:
    """Lazily construct the Groq client on first use (see transcription.py for why)."""
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncGroq(api_key=settings.groq_api_key)
    return _groq_client


def resolve_effective_language(detected_language: str) -> str:
    """
    Map Whisper's raw detected language to the language we actually treat
    the transcript as, correcting a known Whisper mix-up: spoken Urdu and
    spoken Hindi ("Hindustani") sound nearly identical, and Whisper's
    language ID frequently mislabels Urdu speech as Hindi — Whisper then
    transcribes it phonetically into Devanagari script, which is wrong
    for our Urdu-speaking users (correct spoken content, wrong script).

    Since this app's users speak English/Urdu (not Hindi), we treat any
    "hindi" detection as Urdu. Used both for the cleanup prompt (so the
    LLM converts the Devanagari transcript into Urdu Arabic script) and
    for the "🌐 Language: ..." label shown back to the user.
    """
    language_key = (detected_language or "").strip().lower()
    if language_key == "hindi":
        return "urdu"
    return detected_language


def _build_cleanup_system_prompt(detected_language: str) -> str:
    """
    Build the cleanup system prompt with an explicit, strict language rule
    based on what Whisper detected. This exists because without being told
    the detected language up front, the cleanup LLM would sometimes drift
    into translating the text (e.g. into English) or, worse, into a
    completely unrelated language — the model would just "auto-continue"
    in whatever language felt statistically likely rather than preserving
    what was actually spoken.
    """
    original_language_key = (detected_language or "").strip().lower()
    effective_language = resolve_effective_language(detected_language)
    language_key = effective_language.strip().lower()

    if language_key == "urdu" and original_language_key == "hindi":
        # Whisper mislabeled Urdu speech as Hindi and wrote it phonetically
        # in Devanagari script. The spoken content is correct — only the
        # script is wrong — so this is a mechanical script conversion
        # (Devanagari -> Urdu Nastaliq/Arabic script), NOT a translation.
        language_rule = (
            "The detected language was reported as Hindi, written in "
            "Devanagari script. However, spoken Hindi and spoken Urdu "
            "(Hindustani) sound nearly identical, and this is actually "
            "Urdu speech that was mis-scripted. Rewrite the input using "
            "the Urdu Arabic script (اردو) instead of Devanagari — this "
            "is a script conversion of the same words, NOT a translation "
            "into a different language. NEVER use Roman Urdu (Latin "
            "letters), and do not translate the meaning into English or "
            "any other language."
        )
    elif language_key == "urdu":
        language_rule = (
            "The detected spoken language is Urdu. Your output MUST be "
            "written in Urdu using the Urdu Arabic script (اردو) only. "
            "NEVER use Roman Urdu (Urdu written with Latin/English "
            "letters), and NEVER translate the text into English or any "
            "other language."
        )
    elif language_key == "english":
        language_rule = "The detected spoken language is English. Write the output in English."
    else:
        language_rule = (
            f"The detected spoken language is '{effective_language}'. Keep "
            "the output in that same language and in its native script — "
            "do NOT translate it into English or any other language."
        )

    return f"""\
You clean up raw voice-to-text transcripts for a WhatsApp voice notes app.

You are a transcript FORMATTER, not a summarizer. Your only job is to
take the raw spoken-word transcript and lightly polish it — you are NOT
writing a summary, a gist, or a shorter version of what was said.

Detected language: {effective_language}

Rules:
- Remove filler words and verbal stumbles (um, uh, like, you know, etc.).
- Fix punctuation and capitalization, and break the text into clear
  paragraphs where natural pauses or topic changes occur.
- Preserve the original meaning and tone exactly as spoken.
- {language_rule}
- If the speech mixes languages (e.g. Urdu and English in the same
  recording), use Urdu Arabic script as the base, and keep any English
  words that naturally appeared in the speech exactly as-is — do not
  translate them into Urdu and do not transliterate them.
- NEVER transliterate any language into Roman/Latin script under any
  circumstances.
- CRITICAL — NO DEVANAGARI SCRIPT, EVER: your output must NEVER contain
  any Devanagari (Hindi-script) characters, even a single word, even if
  most of the input is already correctly in Urdu Arabic script. Spoken
  Urdu and spoken Hindi sound identical, so Whisper sometimes writes a
  few scattered words of the SAME Urdu speech in Devanagari script by
  mistake, even inside an otherwise-Urdu transcript. Whenever you see
  Devanagari characters anywhere in the input, they are mis-scripted
  Urdu, NOT a foreign-language insertion to preserve — rewrite every
  such word or phrase as the natural Urdu word a native Urdu speaker
  would actually use for that same meaning. This is NOT a letter-by-
  letter phonetic transliteration — Hindi and Urdu often use different
  vocabulary for the same everyday words (e.g. "उपलब्ध" is Hindi for
  "available", but a native Urdu speaker would say "دستیاب", not a
  phonetic spelling of "उपलब्ध"). Use your knowledge of natural,
  everyday spoken Urdu to pick the equivalent word or phrase, exactly
  as if the speaker had said that word in Urdu to begin with. Only
  genuine English words (Latin script) should ever be preserved as-is;
  Devanagari must always be fully converted, never preserved and never
  left mixed in — the final output must be 100% Urdu Arabic script
  (plus any genuine English words), with zero Devanagari characters.
- CRITICAL — DO NOT SUMMARIZE: the cleaned-up text must preserve every
  idea, sentence, and detail from the original speech, and must be
  roughly the same length as the raw transcript (only shorter by the
  filler words you removed). Turning multiple sentences into a single
  short summary sentence is WRONG, even if that sentence captures the
  "gist" — you must keep every distinct point that was made.
- Do not add information that wasn't said.
- Reply with ONLY the cleaned-up text. No preamble, no explanations, no
  wrapping quotation marks.
"""


# A concrete before/after example, given to the model as an actual
# conversation turn (not just described in the system prompt). Few-shot
# examples like this anchor behavior far more reliably than instructions
# alone — this is what stopped the model from summarizing in testing.
_EXAMPLE_RAW_TRANSCRIPT = (
    "Um, so like, I wanted to talk to you about the, the project timeline. "
    "Uh, I think we need to, you know, push the deadline back by like two "
    "weeks because the client hasn't given us the assets yet. And also, "
    "um, the design team is still waiting on feedback from marketing."
)

_EXAMPLE_CLEANED_TRANSCRIPT = (
    "I wanted to talk to you about the project timeline. I think we need "
    "to push the deadline back by two weeks because the client hasn't "
    "given us the assets yet.\n\n"
    "Also, the design team is still waiting on feedback from marketing."
)

# A second few-shot example targeting the partial-Devanagari-contamination
# failure mode specifically: a handful of stray Hindi-script words (using
# different vocabulary than natural Urdu, not just different letters)
# scattered inside an otherwise-correct Urdu transcript. Demonstrates that
# these must become the natural Urdu word/phrase for the same meaning, not
# be preserved as-is and not be phonetically transliterated letter-by-letter.
_EXAMPLE_MIXED_SCRIPT_RAW = (
    "بیٹا، آپ کا ڈیٹا سیٹ उपलब्ध है اور سورس کوڈ بھی उपलब्ध है، तो ایک "
    "ٹیبل بنا کر مجھے دکھائیں۔"
)

_EXAMPLE_MIXED_SCRIPT_CLEANED = (
    "بیٹا، آپ کا ڈیٹا سیٹ دستیاب ہے اور سورس کوڈ بھی دستیاب ہے، تو ایک "
    "ٹیبل بنا کر مجھے دکھائیں۔"
)


async def cleanup_transcript(raw_transcript: str, detected_language: str) -> str:
    """
    Send a raw transcript to a Groq-hosted LLM (llama-3.3-70b-versatile)
    for cleanup: removes filler words, fixes punctuation/paragraphs, and
    keeps the original language, meaning, and full content intact (no
    summarizing — see `_build_cleanup_system_prompt` and the few-shot
    example below for how that's enforced).

    Args:
        raw_transcript: The raw Whisper transcript text.
        detected_language: Whisper's detected language (e.g. "english",
            "urdu") from `app.services.transcription.transcribe_audio`.
            Passed explicitly so the cleanup LLM knows exactly what
            language/script to preserve, rather than guessing — see
            `_build_cleanup_system_prompt`.

    Returns the cleaned text. Raises on API failure — the caller (the
    voice note processing pipeline) is responsible for turning that into
    a user-facing error message.
    """
    if not raw_transcript.strip():
        # Nothing to clean up — let the caller decide how to handle an
        # empty transcript (e.g. silence-only audio) rather than making
        # an unnecessary LLM call.
        return ""

    client = _get_groq_client()
    system_prompt = _build_cleanup_system_prompt(detected_language)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        # Few-shot examples (as real conversation turns, not just described
        # in the prompt) — anchors two behaviors the model otherwise gets
        # wrong: (1) not summarizing, (2) converting stray Devanagari words
        # to their natural Urdu equivalent rather than leaving them as-is.
        {"role": "user", "content": _EXAMPLE_RAW_TRANSCRIPT},
        {"role": "assistant", "content": _EXAMPLE_CLEANED_TRANSCRIPT},
        {"role": "user", "content": _EXAMPLE_MIXED_SCRIPT_RAW},
        {"role": "assistant", "content": _EXAMPLE_MIXED_SCRIPT_CLEANED},
        {"role": "user", "content": raw_transcript},
    ]

    cleaned = ""
    for attempt in range(_MAX_DEVANAGARI_RETRIES + 1):
        completion = await client.chat.completions.create(
            model=CLEANUP_MODEL,
            messages=messages,
            temperature=0.2,
        )
        cleaned = (completion.choices[0].message.content or "").strip()

        leftover_words = sorted(set(_DEVANAGARI_RUN_RE.findall(cleaned)))
        if not leftover_words:
            return cleaned

        if attempt < _MAX_DEVANAGARI_RETRIES:
            # The model didn't fully comply with the "no Devanagari" rule —
            # rather than silently accepting Hindi-script leftovers, feed
            # the exact offending words back as explicit corrective
            # feedback and ask for a full rewrite. This is far more
            # reliable than hoping the same prompt produces a different
            # result on a plain retry, since it points at the specific
            # words the model missed.
            logger.warning(
                "Cleanup output still contains Devanagari words (attempt %d/%d): %s",
                attempt + 1,
                _MAX_DEVANAGARI_RETRIES,
                leftover_words,
            )
            messages.append({"role": "assistant", "content": cleaned})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your reply still contains these Devanagari "
                        f"(Hindi-script) words, which must not appear: "
                        f"{', '.join(leftover_words)}. Rewrite your "
                        "entire previous reply from scratch, keeping "
                        "everything else exactly the same, but replace "
                        "every one of those words with the natural Urdu "
                        "word or phrase a native Urdu speaker would use "
                        "for that same meaning (not a phonetic "
                        "transliteration). Output ONLY the fully "
                        "corrected text, with zero Devanagari characters "
                        "remaining."
                    ),
                }
            )
        else:
            logger.error(
                "Cleanup output still contains Devanagari words after %d retries, "
                "sending as-is: %s",
                _MAX_DEVANAGARI_RETRIES,
                leftover_words,
            )

    return cleaned
