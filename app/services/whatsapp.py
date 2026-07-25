"""
WhatsApp Cloud API service layer.

This module wraps calls to Meta's WhatsApp Cloud API (Graph API):
sending text replies, and resolving/downloading incoming media (voice
notes, images, etc.). Parsing of *incoming* webhook payloads is handled
separately in `app/routers/webhook.py` (kept decoupled so "talk to Meta's
API" and "handle our own webhook" don't get tangled together).
"""

import logging

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


async def get_media_url(media_id: str) -> str:
    """
    Resolve a WhatsApp media ID into a temporary, authenticated download
    URL.

    Incoming audio/voice/image/etc. messages in the webhook payload never
    contain the actual file — only this ID. This is step 1 of 2 for
    downloading media (step 2 is `download_media`, below). The URL
    returned here is short-lived (expires after a few minutes) and itself
    requires the same Bearer token to fetch, so it can't just be handed
    out or cached long-term.

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

    return media_url


async def download_media(media_url: str) -> bytes:
    """
    Download the raw media bytes from the short-lived URL returned by
    `get_media_url`. Requires the same Bearer token used everywhere else —
    the media URL is not a public link.

    Raises on any failure; see `get_media_url` docstring for why.
    """
    if not settings.whatsapp_token:
        raise RuntimeError("WHATSAPP_TOKEN is not configured; cannot download media.")

    headers = {"Authorization": f"Bearer {settings.whatsapp_token}"}

    async with httpx.AsyncClient(timeout=MEDIA_REQUEST_TIMEOUT_SECONDS) as client:
        response = await client.get(media_url, headers=headers)
    response.raise_for_status()

    return response.content
