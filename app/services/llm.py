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


def _build_cleanup_system_prompt(detected_language: str, target_language: str | None = None) -> str:
    """
    Build the cleanup system prompt with an explicit, strict language rule.

    Args:
        detected_language: Whisper's detected spoken language for this
            audio — always used to correctly interpret the input (e.g.
            the Hindi/Urdu Devanagari mislabel below).
        target_language: If None (default), the output must PRESERVE the
            input's language/script as-is (original Phase 2 behavior —
            used for a user's very first transcript, before they have a
            language preference, or whenever preferred_language matches
            what was actually spoken). If set to "english", "urdu", or
            "roman", the output is instead forced into that language/
            script regardless of what was spoken — used when a user has
            set a `preferred_language` in `users.preferred_language`
            that differs from the detected spoken language (Part 5).

    Without being told the detected language up front, the cleanup LLM
    would sometimes drift into translating the text (e.g. into English)
    or, worse, into a completely unrelated language — the model would
    just "auto-continue" in whatever language felt statistically likely
    rather than preserving what was actually spoken.
    """
    original_language_key = (detected_language or "").strip().lower()
    effective_source_language = resolve_effective_language(detected_language)
    target_key = (target_language or "").strip().lower() or None

    if target_key is None:
        # Preserve mode: keep the same language/script that was spoken.
        language_key = effective_source_language.strip().lower()

        if language_key == "urdu" and original_language_key == "hindi":
            # Whisper mislabeled Urdu speech as Hindi and wrote it
            # phonetically in Devanagari script. The spoken content is
            # correct — only the script is wrong — so this is a
            # mechanical script conversion (Devanagari -> Urdu Nastaliq/
            # Arabic script), NOT a translation.
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
                f"The detected spoken language is '{effective_source_language}'. Keep "
                "the output in that same language and in its native script — "
                "do NOT translate it into English or any other language."
            )
        forbid_roman = True
    elif target_key == "roman":
        # Explicit override: user's preferred_language is "roman" — force
        # Roman Urdu (Urdu spelled phonetically with Latin letters)
        # regardless of what language/script was actually spoken/detected.
        language_rule = (
            "Regardless of what language or script the input is written "
            "in, your output MUST be written in Roman Urdu — i.e. Urdu "
            "words spelled out phonetically using Latin/English letters "
            "(e.g. 'Aap kaisay hain?' instead of 'آپ کیسے ہیں؟'). If the "
            "input contains genuine English sentences/words, keep those "
            "as normal English — do not force English content into Urdu "
            "vocabulary — but any Urdu content must be transliterated "
            "into Roman Urdu, never left in Urdu Arabic script and never "
            "in Devanagari."
        )
        forbid_roman = False
    elif target_key == "urdu":
        # Explicit override: force Urdu Arabic script even if the speech
        # was detected as e.g. English — this is a deliberate translation
        # into Urdu, requested via the user's saved preference.
        language_rule = (
            "Regardless of what language was actually spoken, your output "
            "MUST be written in Urdu using the Urdu Arabic script (اردو) "
            "only. Translate the content into natural, fluent Urdu if it "
            "was not already in Urdu. NEVER use Roman Urdu (Latin "
            "letters) and NEVER use Devanagari."
        )
        forbid_roman = True
    else:
        # target_key == "english" (or any other override we might add
        # later) — deliberate translation into English.
        language_rule = (
            "Regardless of what language was actually spoken, your output "
            "MUST be written in natural, fluent English. Translate the "
            "content into English if it was not already in English."
        )
        forbid_roman = True

    roman_rule = (
        "- NEVER transliterate any language into Roman/Latin script under any\n"
        "  circumstances.\n"
        if forbid_roman
        else ""
    )

    return f"""\
You clean up raw voice-to-text transcripts for a WhatsApp voice notes app.

You are a transcript FORMATTER, not a summarizer. Your only job is to
take the raw spoken-word transcript and lightly polish it — you are NOT
writing a summary, a gist, or a shorter version of what was said.

Detected spoken language: {effective_source_language}

Rules:
- Remove filler words and verbal stumbles (um, uh, like, you know, etc.).
- Fix punctuation and capitalization, and break the text into clear
  paragraphs where natural pauses or topic changes occur.
- Preserve the original meaning and tone exactly as spoken.
- {language_rule}
- If the speech mixes languages (e.g. Urdu and English in the same
  recording) and you are not translating to a different target language,
  use Urdu Arabic script as the base, and keep any English words that
  naturally appeared in the speech exactly as-is — do not translate them
  into Urdu and do not transliterate them.
{roman_rule}- CRITICAL — NO DEVANAGARI SCRIPT, EVER: your output must NEVER contain
  any Devanagari (Hindi-script) characters, even a single word. Spoken
  Urdu and spoken Hindi sound identical, so input text sometimes contains
  a few scattered words written in Devanagari script by mistake, even
  inside an otherwise-Urdu transcript. Whenever you see Devanagari
  characters anywhere in the input, they are mis-scripted Urdu, NOT a
  foreign-language insertion to preserve — rewrite every such word or
  phrase as the natural Urdu word a native Urdu speaker would actually
  use for that same meaning (in whatever script/form your target output
  requires — Arabic script or Roman Urdu). This is NOT a letter-by-
  letter phonetic transliteration — Hindi and Urdu often use different
  vocabulary for the same everyday words (e.g. "उपलब्ध" is Hindi for
  "available", but a native Urdu speaker would say "دستیاب", not a
  phonetic spelling of "उपलब्ध"). Use your knowledge of natural,
  everyday spoken Urdu to pick the equivalent word or phrase, exactly
  as if the speaker had said that word in Urdu to begin with. Devanagari
  must always be fully converted, never preserved and never left mixed
  in — the final output must have zero Devanagari characters.
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


async def cleanup_transcript(
    raw_transcript: str,
    detected_language: str,
    target_language: str | None = None,
) -> str:
    """
    Send a raw transcript to a Groq-hosted LLM (llama-3.3-70b-versatile)
    for cleanup: removes filler words, fixes punctuation/paragraphs, and
    keeps the original meaning and full content intact (no summarizing —
    see `_build_cleanup_system_prompt` and the few-shot example below for
    how that's enforced).

    Args:
        raw_transcript: The raw Whisper transcript text.
        detected_language: Whisper's detected language (e.g. "english",
            "urdu") from `app.services.transcription.transcribe_audio`.
            Passed explicitly so the cleanup LLM knows exactly what was
            spoken, rather than guessing.
        target_language: If None (default), the output preserves
            whatever language/script was actually spoken (Phase 2
            behavior). If "english"/"urdu"/"roman", forces the output
            into that language/script instead — used when the user has a
            saved `preferred_language` that differs from what they
            actually spoke this time (Part 5 of Phase 3).

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
    system_prompt = _build_cleanup_system_prompt(detected_language, target_language)

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


_BUILTIN_TRANSLATE_TARGETS = {
    "english": "english",
    "urdu": "urdu",
    # Roman Urdu is handled by a dedicated translate path — reusing
    # cleanup_transcript pulls in English few-shot examples and a
    # "formatter" persona that often drifts into plain English.
}


def _normalize_language_key(language: str) -> str:
    return " ".join((language or "").strip().lower().replace("-", " ").replace("_", " ").split())


def _is_roman_urdu_target(target_language: str) -> bool:
    key = _normalize_language_key(target_language)
    return key in {"roman", "roman urdu", "urdu roman"}


def _build_translate_system_prompt(source_language: str, target_language: str) -> str:
    source = resolve_effective_language(source_language) or "the source language"
    target = (target_language or "").strip() or "English"
    return f"""\
You translate voice note transcripts for a WhatsApp voice notes app.

Translate the input text from {source} into {target}.

Rules:
- Preserve every idea, sentence, and detail — do NOT summarize.
- Keep the original meaning and tone.
- Write fluent, natural {target}.
- If the target is Urdu (not Roman), use Urdu Arabic script (اردو) only —
  never Devanagari and never Roman/Latin letters for Urdu words.
- Do not add information that wasn't in the original text.
- Reply with ONLY the translated text. No preamble, no explanations, no
  wrapping quotation marks.
"""


_ROMAN_URDU_SYSTEM_PROMPT = """\
You convert voice note transcripts into Roman Urdu for a WhatsApp voice notes app.

Roman Urdu = Urdu written phonetically with Latin/English letters
(e.g. "Aap kaise hain?" instead of "آپ کیسے ہیں؟" or "How are you?").

CRITICAL — DO NOT WRITE ENGLISH:
- Your output must read as spoken Urdu in Latin script, NOT as English.
- Wrong: "How are you? I have a meeting tomorrow."
- Right: "Aap kaise hain? Meri kal meeting hai."
- Wrong: "I wanted to talk about the project timeline."
- Right: "Main project timeline ke baare mein baat karna chahta tha."

Rules:
- If the input is Urdu Arabic script (or Devanagari), convert it into
  natural Roman Urdu — same words/meaning, Latin letters.
- If the input is English (or another language), translate the meaning
  into natural spoken Roman Urdu — do NOT leave it as English sentences.
- Keep common English loanwords that Urdu speakers actually say in
  English (meeting, project, deadline, OK, etc.) inside the Roman Urdu
  sentence — but the sentence frame must still be Roman Urdu.
- Preserve every idea, sentence, and detail — do NOT summarize.
- Never use Urdu Arabic script or Devanagari in the output.
- Reply with ONLY the Roman Urdu text. No preamble, no explanations, no
  wrapping quotation marks.
"""

_EXAMPLE_ROMAN_URDU_SRC_URDU = "آپ کیسے ہیں؟ کل میٹنگ ہے اور مجھے دفتر جانا ہے۔"
_EXAMPLE_ROMAN_URDU_OUT_URDU = "Aap kaise hain? Kal meeting hai aur mujhe daftar jana hai."

_EXAMPLE_ROMAN_URDU_SRC_EN = (
    "I wanted to talk to you about the project timeline. "
    "I think we need to push the deadline back by two weeks."
)
_EXAMPLE_ROMAN_URDU_OUT_EN = (
    "Main aap se project timeline ke baare mein baat karna chahta tha. "
    "Mujhe lagta hai hamein deadline ko do haftay peeche dhakelna hoga."
)


async def translate_transcript(
    cleaned_text: str,
    source_language: str,
    target_language: str = "english",
) -> str:
    """
    Translate an already-cleaned transcript (from the in-memory last-
    transcript cache) into `target_language` without re-running Whisper.

    Built-in targets english / urdu reuse the cleanup machinery (same
    "don't summarize" guarantees). Roman Urdu uses a dedicated prompt so
    the model does not drift into plain English. Any other language name
    gets a dedicated translation prompt (Arabic, French, Chinese, etc.).
    """
    if not cleaned_text.strip():
        return ""

    target_key = (target_language or "").strip().lower()
    if not target_key:
        raise ValueError("target_language must not be empty")

    if _is_roman_urdu_target(target_language):
        return await _translate_to_roman_urdu(cleaned_text)

    builtin = _BUILTIN_TRANSLATE_TARGETS.get(target_key)
    if builtin is not None:
        return await cleanup_transcript(
            cleaned_text, source_language, target_language=builtin
        )

    client = _get_groq_client()
    completion = await client.chat.completions.create(
        model=CLEANUP_MODEL,
        messages=[
            {
                "role": "system",
                "content": _build_translate_system_prompt(source_language, target_language),
            },
            {"role": "user", "content": cleaned_text},
        ],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()


async def _translate_to_roman_urdu(cleaned_text: str) -> str:
    """Dedicated Roman Urdu path with few-shot examples (Urdu + English inputs)."""
    client = _get_groq_client()
    completion = await client.chat.completions.create(
        model=CLEANUP_MODEL,
        messages=[
            {"role": "system", "content": _ROMAN_URDU_SYSTEM_PROMPT},
            {"role": "user", "content": _EXAMPLE_ROMAN_URDU_SRC_URDU},
            {"role": "assistant", "content": _EXAMPLE_ROMAN_URDU_OUT_URDU},
            {"role": "user", "content": _EXAMPLE_ROMAN_URDU_SRC_EN},
            {"role": "assistant", "content": _EXAMPLE_ROMAN_URDU_OUT_EN},
            {"role": "user", "content": cleaned_text},
        ],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()


_SUMMARIZE_LANGUAGE_LABELS = {
    "urdu": "Urdu, written in the Urdu Arabic script (اردو)",
    "english": "English",
    "roman": "Roman Urdu (Urdu spelled phonetically with Latin letters)",
}


def _build_summarize_system_prompt(target_language: str) -> str:
    language_key = (target_language or "").strip().lower()
    language_label = _SUMMARIZE_LANGUAGE_LABELS.get(language_key, target_language or "the same language as the input")

    return f"""\
You summarize voice note transcripts for a WhatsApp voice notes app.

Summarize the input text into a short list of concise bullet points
capturing only the key ideas — unlike transcript cleanup, brevity here IS
the goal.

Rules:
- Write the summary in {language_label}.
- Use "- " bullet points, one idea per line.
- Keep it concise: only the key points, not every detail.
- Do not add information that wasn't in the original text.
- Reply with ONLY the bullet points. No preamble, no closing remarks, no
  wrapping quotation marks.
"""


async def summarize_transcript(cleaned_text: str, target_language: str) -> str:
    """
    Re-run the LLM on an already-cleaned transcript to produce a concise
    bullet-point summary in `target_language` (the transcript's own
    language/script — callers should pass the cached detected language,
    not a translation preference). Used by Summarize / `/summarize`.
    """
    if not cleaned_text.strip():
        return ""

    client = _get_groq_client()
    completion = await client.chat.completions.create(
        model=CLEANUP_MODEL,
        messages=[
            {"role": "system", "content": _build_summarize_system_prompt(target_language)},
            {"role": "user", "content": cleaned_text},
        ],
        temperature=0.2,
    )
    return (completion.choices[0].message.content or "").strip()
