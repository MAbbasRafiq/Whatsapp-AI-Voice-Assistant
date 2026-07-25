"""Pydantic models / schemas package.

Phase 1 does not yet define dedicated request/response models for the
WhatsApp payloads (we parse the raw JSON dict directly in the webhook
router). This package is a placeholder for Phase 2+, where we'll likely
add typed models for incoming messages, transcription results, etc.
"""
