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

Security + persistence pipeline, run in this exact order for every
incoming message:

    1. X-Hub-Signature-256 verification (whole request, before anything
       else is even parsed) — see app/services/security.py.
    2. Ensure a `users` row exists + update last_seen.
    3. Blocklist check (`users.is_blocked`) — silently ignored if true.
    4. Dedup check (`messages.wa_message_id` unique constraint) — silently
       ignored if this delivery was already processed (Meta retry).
    5. Dispatch: voice/audio -> rate limit -> full processing pipeline;
       interactive button/list replies -> interactive handler;
       plain text while waiting_for_language -> language input handler;
       "/command" text -> command handler; anything else -> default nudge.
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response, status

from app.config import settings
from app.services import db_ops
from app.services.commands import (
    DEFAULT_REPLY,
    handle_command,
    handle_interactive,
    handle_language_input,
    is_command,
)
from app.services.rate_limiter import check_and_record
from app.services.security import verify_signature
from app.services.user_state import is_waiting_for_language
from app.services.voice_processing import process_voice_note
from app.services.whatsapp import send_text_message
from app.utils import mask_phone

logger = logging.getLogger(__name__)

_RATE_LIMITED_REPLY = "⏳ Please wait a few seconds before sending another voice note."

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
    lightweight: verify + parse + log, then schedule all the real work
    (DB checks, dispatch, the full voice note pipeline, etc.) as a single
    BackgroundTask per message so it runs *after* we've already responded
    200 below.

    Signature verification is the one thing that runs *before* returning
    200 — an invalid signature means the request didn't come from Meta at
    all, so it gets a 403 and no processing of any kind.
    """
    raw_body = await request.body()

    if not verify_signature(raw_body, request.headers.get("X-Hub-Signature-256")):
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    try:
        payload = await request.json()
    except Exception:
        logger.exception("Received webhook POST with invalid/non-JSON body.")
        return Response(status_code=status.HTTP_200_OK)

    logger.debug("Raw webhook payload: %s", payload)

    for message_info in _extract_messages(payload):
        background_tasks.add_task(_handle_message, message_info)

    # Always acknowledge with 200 quickly, even if there was nothing to
    # parse (e.g. a status update) or the payload had an unexpected shape.
    return Response(status_code=status.HTTP_200_OK)


async def _handle_message(message_info: dict) -> None:
    """
    Runs as a BackgroundTask, once per incoming message, *after* the
    webhook has already returned 200 to Meta. Applies the user-record /
    blocklist / dedup checks, then dispatches to the voice note pipeline,
    interactive handler, language-input handler, command handler, or
    default reply.

    Never lets an exception escape — this runs detached from any request
    context, so an unhandled exception here would just vanish into
    BackgroundTasks' internals rather than surfacing anywhere useful.
    """
    sender = message_info["from"]
    wa_message_id = message_info["id"]
    message_type = message_info["type"]
    content = message_info["content"]
    interactive_id = message_info.get("interactive_id")
    interactive_title = message_info.get("interactive_title") or ""

    logger.info(
        "Incoming WhatsApp message | from=%s | id=%s | type=%s",
        mask_phone(sender),
        wa_message_id,
        message_type,
    )

    try:
        user, _is_new_user = await db_ops.touch_user(sender)

        if user.is_blocked:
            logger.info("Ignoring message from blocked user %s", mask_phone(sender))
            return

        is_new_delivery = await db_ops.record_message(wa_message_id, sender, status="received")
        if not is_new_delivery:
            # Duplicate webhook delivery (Meta retry) — already logged
            # inside record_message(); skip silently, no reply.
            return

        if message_type in ("audio", "voice"):
            if not check_and_record(sender):
                await send_text_message(sender, _RATE_LIMITED_REPLY)
                await db_ops.update_message_status(wa_message_id, "rate_limited")
                return

            await db_ops.update_message_status(wa_message_id, "queued")
            await process_voice_note(
                media_id=content,
                sender=sender,
                wa_message_id=wa_message_id,
            )
        elif message_type == "interactive" and interactive_id:
            await handle_interactive(sender, interactive_id, interactive_title)
            await db_ops.update_message_status(wa_message_id, "succeeded")
        elif message_type == "text" and is_waiting_for_language(sender):
            await handle_language_input(sender, content)
            await db_ops.update_message_status(wa_message_id, "succeeded")
        elif message_type == "text" and is_command(content):
            await handle_command(sender, content)
            await db_ops.update_message_status(wa_message_id, "succeeded")
        else:
            # Anything else (plain text, image, sticker, location, etc.)
            # gets a friendly nudge toward the bot's actual purpose.
            await send_text_message(sender, DEFAULT_REPLY)
            await db_ops.update_message_status(wa_message_id, "succeeded")

    except Exception:
        logger.exception(
            "Unhandled error while processing message | from=%s | id=%s",
            mask_phone(sender),
            wa_message_id,
        )


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
                        "id": "wamid.XXXX",
                        "type": "text" | "audio" | "voice" | "interactive" | ...,
                        "text": {"body": "hello"},
                        "audio": {"id": "<media_id>", ...},
                        "voice": {"id": "<media_id>", ...},
                        "interactive": {
                          "type": "button_reply" | "list_reply",
                          "button_reply": {"id": "...", "title": "..."},
                          "list_reply": {"id": "...", "title": "..."}
                        },
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

    Returns a list of dicts with keys: from, id, type, content,
    interactive_id (optional), interactive_title (optional).
    """
    results: list[dict] = []

    if not isinstance(payload, dict):
        return results

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value", {}) or {}

            # No "messages" key means this change is something else, e.g.
            # a "statuses" update (sent/delivered/read/failed). Nothing to
            # extract — just skip it quietly.
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
    Extract sender, message ID, type, and content/media-id / interactive
    reply from a single WhatsApp message object, tolerating missing/
    unexpected fields.
    """
    sender = message.get("from", "unknown")
    wa_message_id = message.get("id", "")
    message_type = message.get("type", "unknown")

    content = ""
    interactive_id: str | None = None
    interactive_title: str | None = None

    if message_type == "text":
        content = message.get("text", {}).get("body", "")
    elif message_type in ("audio", "voice"):
        # Audio/voice messages don't include the actual audio bytes here —
        # only a media ID, which must be fetched separately via the Graph
        # API's media endpoint (see app/services/whatsapp.py).
        content = message.get(message_type, {}).get("id", "")
    elif message_type == "interactive":
        interactive = message.get("interactive", {}) or {}
        interactive_type = interactive.get("type", "")
        if interactive_type == "button_reply":
            reply = interactive.get("button_reply", {}) or {}
        elif interactive_type == "list_reply":
            reply = interactive.get("list_reply", {}) or {}
        else:
            reply = {}
        interactive_id = reply.get("id") or None
        interactive_title = reply.get("title") or ""
        content = interactive_id or ""
    else:
        # Covers other WhatsApp message types (image, video, document,
        # location, contacts, sticker, etc.) which we don't handle beyond
        # the default nudge reply.
        content = f"<unsupported message type: {message_type}>"

    return {
        "from": sender,
        "id": wa_message_id,
        "type": message_type,
        "content": content,
        "interactive_id": interactive_id,
        "interactive_title": interactive_title,
    }
