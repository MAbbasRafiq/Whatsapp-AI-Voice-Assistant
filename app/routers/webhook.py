"""
WhatsApp webhook routes.

Meta's WhatsApp Cloud API talks to our server via a single webhook URL that
supports two HTTP methods:

  - GET  /webhook  -> "verification handshake". Meta calls this once when
                       you register the webhook URL in the App Dashboard,
                       to prove you control the endpoint.
  - POST /webhook   -> "event delivery". Meta calls this every time
                       something happens (a new message, a status update
                       like "delivered"/"read", etc).

Reference: https://developers.facebook.com/docs/graph-api/webhooks/getting-started
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response, status

from app.config import settings
from app.services.voice_processing import process_voice_note
from app.services.whatsapp import send_text_message

logger = logging.getLogger(__name__)

# Shown to the user for any incoming message that isn't a voice/audio note
# (plain text, images, stickers, etc.) — this bot's whole purpose is
# turning voice notes into clean text, so we nudge users toward that.
_NOT_A_VOICE_NOTE_REPLY = "🎤 Please send a voice note and I'll convert it to clean text for you!"

router = APIRouter(tags=["webhook"])


@router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(default="", alias="hub.mode"),
    hub_verify_token: str = Query(default="", alias="hub.verify_token"),
    hub_challenge: str = Query(default="", alias="hub.challenge"),
) -> Response:
    """
    Webhook verification handshake, called once by Meta when you set/save
    the webhook URL + verify token in the Meta App Dashboard.

    Meta sends three query params: hub.mode, hub.verify_token, hub.challenge.
    If hub.mode == "subscribe" and hub.verify_token matches the secret we
    configured (WHATSAPP_VERIFY_TOKEN), we must echo back hub.challenge as
    plain text with a 200 status. Any mismatch should be rejected with 403
    so random/unauthenticated callers can't "verify" against our endpoint.
    """
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("Webhook verification succeeded.")
        # Must be returned as plain text, not JSON, or Meta will reject it.
        return Response(content=hub_challenge, media_type="text/plain", status_code=status.HTTP_200_OK)

    logger.warning(
        "Webhook verification failed. mode=%r token_matches=%s",
        hub_mode,
        hub_verify_token == settings.whatsapp_verify_token,
    )
    return Response(status_code=status.HTTP_403_FORBIDDEN)


@router.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks) -> Response:
    """
    Receive incoming WhatsApp events (messages, status updates, etc).

    IMPORTANT: WhatsApp expects a fast 200 OK response. If we don't
    acknowledge quickly, Meta will consider the delivery failed and retry,
    which can lead to duplicate processing later. So this handler stays
    lightweight: parse + log, then schedule all the real work (auto-reply
    for non-voice messages, or the full download/transcribe/cleanup/reply
    pipeline for voice notes) as BackgroundTasks so it runs *after* we've
    already responded 200 to Meta below. A future phase may swap
    BackgroundTasks for a proper task queue if reliability/retries become
    a concern (BackgroundTasks don't survive a server restart mid-task).

    We also must never let a malformed/unexpected payload shape crash this
    endpoint — WhatsApp sends other event types too (e.g. message status
    updates like "delivered"/"read"), which have a different shape and
    don't contain an actual user message. We defensively parse everything
    and always return 200, even when there's nothing interesting to log.
    """
    try:
        payload = await request.json()
    except Exception:
        logger.exception("Received webhook POST with invalid/non-JSON body.")
        return Response(status_code=status.HTTP_200_OK)

    logger.debug("Raw webhook payload: %s", payload)

    for message_info in _extract_messages(payload):
        sender = message_info["from"]
        message_type = message_info["type"]
        content = message_info["content"]

        logger.info(
            "Incoming WhatsApp message | from=%s | type=%s | content=%s",
            sender,
            message_type,
            content,
        )

        if message_type in ("audio", "voice"):
            # `content` is the media ID for audio/voice messages (see
            # `_parse_single_message` below). The full download -> convert
            # -> transcribe -> cleanup -> reply pipeline lives in
            # `app.services.voice_processing` and is scheduled as a
            # background task so it runs *after* we return 200 to Meta
            # below — required, since that pipeline involves several
            # slow network calls (Graph API, ffmpeg, Groq) that would
            # otherwise blow past WhatsApp's fast-ack requirement.
            background_tasks.add_task(process_voice_note, media_id=content, sender=sender)
        else:
            # Anything else (text, image, sticker, location, etc.) gets a
            # friendly nudge toward the bot's actual purpose. Also
            # scheduled as a background task purely for consistency/
            # symmetry with the voice note path above — a single text
            # send is fast, but this keeps the response path uniform.
            background_tasks.add_task(send_text_message, to=sender, message=_NOT_A_VOICE_NOTE_REPLY)

    # Always acknowledge with 200 quickly, even if there was nothing to
    # parse (e.g. a status update) or the payload had an unexpected shape.
    return Response(status_code=status.HTTP_200_OK)


def _extract_messages(payload: dict) -> list[dict]:
    """
    Walk the WhatsApp Cloud API webhook payload structure and pull out a
    simplified list of incoming messages.

    Expected shape (trimmed to the relevant path):

        {
          "entry": [
            {
              "changes": [
                {
                  "value": {
                    "messages": [
                      {
                        "from": "15551234567",
                        "type": "text" | "audio" | "voice" | ...,
                        "text": {"body": "hello"},                 # if type == "text"
                        "audio": {"id": "<media_id>", ...},          # if type == "audio"
                        "voice": {"id": "<media_id>", ...},          # if type == "voice"
                        ...
                      }
                    ]
                  }
                }
              ]
            }
          ]
        }

    Not every webhook call contains "messages" — e.g. delivery/read status
    updates use a "statuses" key instead with a completely different
    shape. We simply skip anything that doesn't match, rather than
    treating it as an error.

    Returns a list of dicts: {"from": str, "type": str, "content": str}
    where "content" is either the text body or a media ID (for
    audio/voice messages), depending on the message type.
    """
    results: list[dict] = []

    if not isinstance(payload, dict):
        return results

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}

            # No "messages" key means this change is something else, e.g.
            # a "statuses" update (sent/delivered/read/failed). Nothing to
            # extract for Phase 1 — just skip it quietly.
            messages = value.get("messages")
            if not messages:
                if value.get("statuses"):
                    logger.debug("Received a status update (not a message): %s", value.get("statuses"))
                continue

            for message in messages:
                results.append(_parse_single_message(message))

    return results


def _parse_single_message(message: dict) -> dict:
    """
    Extract sender, type, and content/media-id from a single WhatsApp
    message object, tolerating missing/unexpected fields.
    """
    sender = message.get("from", "unknown")
    message_type = message.get("type", "unknown")

    content: str
    if message_type == "text":
        content = message.get("text", {}).get("body", "")
    elif message_type in ("audio", "voice"):
        # Audio/voice messages don't include the actual audio bytes here —
        # only a media ID, which must be fetched separately via the Graph
        # API's media endpoint. That download/transcription flow is Phase 2.
        content = message.get(message_type, {}).get("id", "")
    else:
        # Covers other WhatsApp message types (image, video, document,
        # location, contacts, sticker, interactive/button replies, etc.)
        # which we don't handle yet in Phase 1.
        content = f"<unsupported message type: {message_type}>"

    return {"from": sender, "type": message_type, "content": content}
