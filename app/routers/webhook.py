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
       eligible plain text -> Convert to Voice offer; `/voice` (compat)
       -> TTS; anything else -> default nudge.
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
    handle_plain_text,
    is_command,
)
from app.services.rate_limiter import release_voice, try_acquire_voice
from app.services.security import verify_signature
from app.services.user_errors import ErrorType, classify_exception, send_user_error
from app.services.user_state import clear_waiting_for_language, is_waiting_for_language
from app.services.voice_processing import process_voice_note
from app.services.whatsapp import send_text_message
from app.utils import mask_phone

logger = logging.getLogger(__name__)

router = APIRouter(tags=["webhook"])

_VOICE_BUSY_REPLY = "⏳ Still processing your previous voice note. Please wait a moment."


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
        if not wa_message_id:
            logger.warning(
                "Skipping message with empty wa_message_id | from=%s | type=%s",
                mask_phone(sender),
                message_type,
            )
            return

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
            gate = try_acquire_voice(sender)
            if gate == "busy":
                await send_text_message(sender, _VOICE_BUSY_REPLY)
                await _safe_update_message_status(wa_message_id, "rate_limited")
                return
            if gate == "rate_limited":
                await send_user_error(sender, ErrorType.RATE_LIMITED)
                await _safe_update_message_status(wa_message_id, "rate_limited")
                return

            try:
                await _safe_update_message_status(wa_message_id, "queued")
                await process_voice_note(
                    media_id=content,
                    sender=sender,
                    wa_message_id=wa_message_id,
                )
            finally:
                release_voice(sender)
        elif message_type == "interactive" and interactive_id:
            await handle_interactive(sender, interactive_id, interactive_title)
            await _safe_update_message_status(wa_message_id, "succeeded")
        elif message_type == "interactive":
            # Interactive payload present but missing/empty reply id —
            # previously fell through to the generic nudge; keep that UX
            # via the centralized unexpected-interactive message (same text).
            logger.warning(
                "Interactive message with no reply id | from=%s | id=%s",
                mask_phone(sender),
                wa_message_id,
            )
            await send_user_error(sender, ErrorType.UNEXPECTED_INTERACTIVE)
            await _safe_update_message_status(wa_message_id, "succeeded")
        elif message_type == "text" and is_command(content):
            # /voice kept for backward compatibility — not advertised in UX.
            clear_waiting_for_language(sender)
            await handle_command(sender, content)
            await _safe_update_message_status(wa_message_id, "succeeded")
        elif message_type == "text" and is_waiting_for_language(sender):
            await handle_language_input(sender, content)
            await _safe_update_message_status(wa_message_id, "succeeded")
        elif message_type == "text":
            sent = await handle_plain_text(sender, content)
            if not sent:
                logger.error(
                    "Failed to handle plain text | from=%s | id=%s",
                    mask_phone(sender),
                    wa_message_id,
                )
                await _safe_update_message_status(wa_message_id, "failed")
            else:
                await _safe_update_message_status(wa_message_id, "succeeded")
        else:
            # Anything else (image, sticker, location, etc.) gets a friendly
            # nudge toward the bot's actual purpose.
            sent = await send_text_message(sender, DEFAULT_REPLY)
            if not sent:
                logger.error(
                    "Failed to send default nudge reply | from=%s | id=%s",
                    mask_phone(sender),
                    wa_message_id,
                )
                await _safe_update_message_status(wa_message_id, "failed")
            else:
                await _safe_update_message_status(wa_message_id, "succeeded")

    except Exception as exc:
        logger.exception(
            "Unhandled error while processing message | from=%s | id=%s",
            mask_phone(sender),
            wa_message_id,
        )
        # Previously this path logged only — the user got no WhatsApp reply.
        # Notify with a classified friendly message; if sending also fails,
        # log and move on (can't reach the user if Graph itself is down).
        try:
            await send_user_error(sender, classify_exception(exc))
        except Exception:
            logger.exception(
                "Failed to send error reply after unhandled error | from=%s",
                mask_phone(sender),
            )
        await _safe_update_message_status(wa_message_id, "failed")


async def _safe_update_message_status(wa_message_id: str, status: str) -> None:
    """
    Best-effort status write. Failures are logged but never raised so a
    DB blip after the user already got a reply cannot trigger a second
    (misleading) error message from the outer handler.
    """
    try:
        await db_ops.update_message_status(wa_message_id, status)
    except Exception:
        logger.exception(
            "Failed to update message status | id=%s | status=%s",
            wa_message_id,
            status,
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
