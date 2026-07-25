"""
WhatsApp Cloud API service layer.

This module wraps calls to Meta's WhatsApp Cloud API (Graph API):
sending text replies, and resolving/downloading incoming media (voice
notes, images, etc.). Parsing of *incoming* webhook payloads is handled
separately in `app/routers/webhook.py` (kept decoupled so "talk to Meta's
API" and "handle our own webhook" don't get tangled together).
"""

import logging
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Base URL for the Graph API. Meta periodically bumps the API version; keep
# this centralized so it's easy to update in one place later.
GRAPH_API_VERSION = "v20.0"
GRAPH_API_BASE_URL = f"https://graph.facebook.com/{GRAPH_API_VERSION}"

# A reasonable timeout so a slow/unresponsive Graph API call can't hang
# our webhook handler forever.
REQUEST_TIMEOUT_SECONDS = 10.0

# Media lookups/downloads get a longer timeout than plain API calls since
# they involve fetching an actual audio file, potentially over a slower
# connection than a small JSON request.
MEDIA_REQUEST_TIMEOUT_SECONDS = 30.0

# WhatsApp's hard limit on a single text message body is 4096 characters.
# We target a smaller budget (3996) to leave ~100 chars of headroom for a
# "Part X/Y" label prepended to each chunk when a message has to be split.
_WHATSAPP_MAX_MESSAGE_LENGTH = 4096
_CHUNK_TARGET_LENGTH = 3996
_MAX_CHUNKS = 3
_TRUNCATION_SUFFIX = " ... [truncated]"


async def send_text_message(to: str, message: str) -> bool:
    """
    Send a plain text WhatsApp message to a given phone number.

    Args:
        to: Recipient's phone number in international format, no "+" or
            leading zeros (e.g. "15551234567"). This is the format the
            Cloud API expects and is also how it reports sender numbers
            in incoming webhook payloads.
        message: The text body to send.

    Returns:
        True if the message was accepted by the Graph API (HTTP 2xx),
        False otherwise. Errors are logged, never raised, so a failed
        send doesn't crash the caller (e.g. the webhook handler).
    """
    if not settings.whatsapp_token or not settings.whatsapp_phone_number_id:
        logger.error(
            "Cannot send WhatsApp message: WHATSAPP_TOKEN or "
            "WHATSAPP_PHONE_NUMBER_ID is not configured. Check your .env file."
        )
        return False

    url = f"{GRAPH_API_BASE_URL}/{settings.whatsapp_phone_number_id}/messages"

    headers = {
        "Authorization": f"Bearer {settings.whatsapp_token}",
        "Content-Type": "application/json",
    }

    # Standard WhatsApp Cloud API payload shape for a plain text message.
    # https://developers.facebook.com/docs/whatsapp/cloud-api/reference/messages
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=payload)

        if response.is_success:
            logger.info("Sent WhatsApp message to %s: %r", to, message)
            return True

        # Meta returns a JSON error body with useful details on failure.
        logger.error(
            "Failed to send WhatsApp message to %s. Status=%s Response=%s",
            to,
            response.status_code,
            response.text,
        )
        return False

    except httpx.RequestError as exc:
        # Network-level failure (timeout, DNS, connection refused, etc.)
        logger.exception("Network error while sending WhatsApp message to %s: %s", to, exc)
        return False


async def get_media_info(media_id: str) -> tuple[str, int | None]:
    """
    Resolve a WhatsApp media ID into a temporary, authenticated download
    URL, plus the file size in bytes if Meta reports one.

    Incoming audio/voice/image/etc. messages in the webhook payload never
    contain the actual file — only this ID. This is step 1 of 2 for
    downloading media (step 2 is `download_media`, below). The URL
    returned here is short-lived (expires after a few minutes) and itself
    requires the same Bearer token to fetch, so it can't just be handed
    out or cached long-term.

    Returning `file_size` here (when Meta provides it) lets the caller
    reject an oversized voice note (Part 4's 25MB check) *before*
    spending time/bandwidth on `download_media` — Meta's media-lookup
    response usually already includes it, so this is effectively free.

    Raises on any failure (missing config, HTTP error, unexpected
    response shape) — callers are expected to catch and turn this into a
    user-facing error message, since this runs as part of the voice note
    processing pipeline.
    """
    if not settings.whatsapp_token:
        raise RuntimeError("WHATSAPP_TOKEN is not configured; cannot resolve media URL.")

    url = f"{GRAPH_API_BASE_URL}/{media_id}"
    headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}

    async with httpx.AsyncClient(timeout=MEDIA_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    media_url = data.get("url")
    if not media_url:
        raise RuntimeError(f"Graph API media lookup for {media_id!r} returned no 'url' field: {data}")

    file_size = data.get("file_size")
    return media_url, (int(file_size) if file_size is not None else None)


async def download_media(media_url: str) -> bytes:
    """
    Download the raw media bytes from the short-lived URL returned by
    `get_media_info`. Requires the same Bearer token used everywhere else —
    the media URL is not a public link.

    Raises on any failure; see `get_media_info` docstring for why.
    """
    if not settings.whatsapp_token:
        raise RuntimeError("WHATSAPP_TOKEN is not configured; cannot download media.")

    headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}

    async with httpx.AsyncClient(timeout=MEDIA_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(media_url, headers=headers)
    response.raise_for_status()

    return response.content


def split_into_chunks(
    text: str,
    chunk_size: int = _CHUNK_TARGET_LENGTH,
    max_chunks: int = _MAX_CHUNKS,
) -> list[str]:
    """
    Split a long message into at most `max_chunks` pieces, breaking on
    paragraph boundaries first, then sentence boundaries, and only
    falling back to a hard character slice if a single sentence is
    itself longer than `chunk_size` (pathological case).

    If the content still doesn't fit in `max_chunks` pieces, the last
    chunk is truncated and "... [truncated]" is appended, rather than
    silently dropping content across more messages than the caller wants.

    Does NOT add "Part X/Y" labels — that's the caller's job (see
    `send_long_message`), since a single-chunk message shouldn't get one.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def _add_piece(piece: str, joiner: str) -> None:
        """Greedily append `piece` to `current`, flushing when full."""
        nonlocal current
        candidate = f"{current}{joiner}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            return

        _flush()
        if len(piece) <= chunk_size:
            current = piece
            return

        # A single sentence/paragraph longer than chunk_size on its own —
        # split on sentence boundaries, and as a last resort hard-slice.
        sentences = re.split(r"(?<=[.!?\u06d4])\s+", piece)
        if len(sentences) > 1:
            for sentence in sentences:
                _add_piece(sentence, " ")
            return

        for i in range(0, len(piece), chunk_size):
            chunks.append(piece[i : i + chunk_size])

    for paragraph in paragraphs:
        _add_piece(paragraph, "\n\n")
    _flush()

    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
        last = chunks[-1]
        keep_length = max(0, chunk_size - len(_TRUNCATION_SUFFIX))
        chunks[-1] = last[:keep_length].rstrip() + _TRUNCATION_SUFFIX

    return chunks


async def send_long_message(to: str, message: str) -> bool:
    """
    Send a text message that may exceed WhatsApp's 4096-character limit,
    automatically splitting it into up to 3 chunks (labeled "Part X/Y")
    on paragraph/sentence boundaries. Messages that already fit in one
    chunk are sent exactly as `send_text_message` would send them — no
    "Part 1/1" label is added for single-chunk messages.

    Returns True only if every chunk was sent successfully.
    """
    chunks = split_into_chunks(message)

    if len(chunks) == 1:
        return await send_text_message(to, chunks[0])

    total = len(chunks)
    all_sent = True
    for index, chunk in enumerate(chunks, start=1):
        labeled_chunk = f"Part {index}/{total}\n\n{chunk}"
        sent = await send_text_message(to, labeled_chunk)
        all_sent = all_sent and sent

    return all_sent
