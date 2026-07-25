"""
Application configuration.

All secrets and environment-specific values are loaded from environment
variables (or a local .env file when running locally) via pydantic-settings.

NEVER hardcode API keys/tokens in the source code — add them to a local
.env file (see .env.example) and access them through the `settings` object
exported from this module.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings, populated from environment variables."""

    # --- WhatsApp Cloud API credentials -----------------------------------
    # Access token for calling the Meta Graph API (WhatsApp Cloud API).
    # Generated in the Meta developer dashboard for your app/business account.
    whatsapp_token: str = ""

    # The "Phone Number ID" (not the actual phone number) assigned by Meta
    # to the WhatsApp Business number you're sending/receiving messages with.
    whatsapp_phone_number_id: str = ""

    # A secret string you choose yourself. You enter the same value in the
    # Meta dashboard when configuring the webhook, and Meta echoes it back
    # in the GET /webhook verification request so we can confirm the
    # request is legitimate.
    whatsapp_verify_token: str = ""

    # --- Groq API (Phase 2: voice note transcription + LLM cleanup) --------
    # API key from https://console.groq.com/keys. Used for both the
    # Whisper-compatible transcription endpoint and the chat completions
    # endpoint (transcript cleanup).
    groq_api_key: str = ""

    # --- General app settings ----------------------------------------------
    # "development", "staging", "production", etc. Defaults to "development".
    app_env: str = "development"

    # Reads variables from a `.env` file (if present) in addition to real
    # environment variables. Real environment variables always take
    # precedence over values in the .env file.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache means the .env file / environment is only read once per
    process, and the same Settings object is reused everywhere it's needed
    (e.g. via FastAPI dependency injection: `settings: Settings = Depends(get_settings)`).
    """
    return Settings()


# Convenience module-level instance for simple, non-DI usage (e.g. in
# services that just need `from app.config import settings`).
settings = get_settings()
