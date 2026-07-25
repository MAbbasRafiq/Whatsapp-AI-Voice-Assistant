"""
WhatsApp webhook request authentication.

Meta signs every webhook POST body with HMAC-SHA256, keyed with the app's
App Secret, and sends the result in the `X-Hub-Signature-256` header as
`sha256=<hex digest>`. Verifying this signature proves the request really
came from Meta (not e.g. a random POST to our public webhook URL trying
to trigger fake messages/spam users) and MUST happen before we do any
other processing of the request body.

Reference: https://developers.facebook.com/docs/graph-api/webhooks/getting-started#verification-requests
"""

import hashlib
import hmac
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_SIGNATURE_PREFIX = "sha256="


def verify_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """
    Verify the `X-Hub-Signature-256` header against the raw request body.

    Args:
        raw_body: The exact, unparsed request body bytes (signature is
            computed over the raw bytes — re-serializing parsed JSON
            would produce a different byte sequence and fail to match).
        signature_header: The raw header value, expected to look like
            "sha256=<hex digest>".

    Returns:
        True if the signature is valid. False on any mismatch, missing
        header, or missing APP_SECRET config.
    """
    if not settings.app_secret:
        # Fail closed would block all traffic if the operator forgot to
        # configure APP_SECRET — but fail *open* would silently disable a
        # security control without anyone noticing. We fail closed and
        # log loudly so misconfiguration is obvious immediately rather
        # than discovered later as a security gap.
        logger.error("APP_SECRET is not configured; rejecting webhook request.")
        return False

    if not signature_header or not signature_header.startswith(_SIGNATURE_PREFIX):
        logger.warning("Webhook request missing/malformed X-Hub-Signature-256 header.")
        return False

    provided_digest = signature_header[len(_SIGNATURE_PREFIX):]

    expected_digest = hmac.new(
        key=settings.app_secret.encode("utf-8"),
        msg=raw_body,
        digestmod=hashlib.sha256,
    ).hexdigest()

    # constant-time comparison to avoid leaking timing information about
    # how many leading characters matched.
    is_valid = hmac.compare_digest(provided_digest, expected_digest)
    if not is_valid:
        logger.warning("Webhook signature verification failed (digest mismatch).")
    return is_valid
