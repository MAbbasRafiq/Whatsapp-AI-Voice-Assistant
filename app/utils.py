"""Small shared helpers used across multiple service modules."""

import json
import logging

# Dedicated logger for structured analytics events (Phase 3, Part 10).
# Configured in app/main.py with its own handler/formatter (plain
# "%(message)s", no timestamp/level prefix, propagate=False) so every
# line emitted through this logger is a single, directly-parseable JSON
# object — useful for cost estimation / debugging without a real log
# aggregator or DB table.
analytics_logger = logging.getLogger("analytics")


def mask_phone(wa_phone: str) -> str:
    """
    Last 4 digits only, prefixed with "...", for privacy in logs (e.g.
    structured analytics logging — see app/services/voice_processing.py).
    """
    return f"...{wa_phone[-4:]}" if len(wa_phone) >= 4 else wa_phone


def log_voice_note_event(**fields: object) -> None:
    """
    Emit one structured JSON log line for a single voice note processing
    attempt. See the `event` key ("voice_note_processed") and field list
    in app/services/voice_processing.py for the exact schema.
    """
    analytics_logger.info(json.dumps(fields, ensure_ascii=False, default=str))
