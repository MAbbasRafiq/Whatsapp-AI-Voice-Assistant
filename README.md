# VoiceNotes — WhatsApp Bot

VoiceNotes is a WhatsApp bot built with FastAPI that turns voice notes into
clean, well-formatted text.

- **Phase 1**: project skeleton + a working webhook that verifies itself
  with Meta, receives incoming WhatsApp messages, and sends plain text
  replies back.
- **Phase 2** : full voice note pipeline — download the audio
  from Meta, convert it with ffmpeg, transcribe it with Groq's Whisper
  API, clean up the transcript with a Groq LLM, and reply with the result.

Persistent storage / a database is not implemented yet (future phase).

## How it works

1. User sends a voice note on WhatsApp.
2. Meta POSTs the message to `/webhook` — the payload contains a `media_id`,
   **not** the actual audio file.
3. The webhook responds `200 OK` immediately (required by WhatsApp), and
   schedules the rest of the work as a background task.
4. Background task (`app/services/voice_processing.py`):
   a. Sends the user "⏳ Processing your voice note...".
   b. Resolves the `media_id` into a short-lived download URL via the
      Graph API, then downloads the raw audio (`app/services/whatsapp.py`).
   c. Converts the OGG/Opus audio to 16kHz mono WAV using ffmpeg, bundled
      via the `imageio-ffmpeg` pip package — no manual ffmpeg install
      needed (`app/services/transcription.py`).
   d. Transcribes the WAV file with Groq's Whisper-compatible API
      (`whisper-large-v3` — the full model, not the `-turbo` variant,
      which has noticeably worse multilingual accuracy/language
      detection), with automatic language detection — handles English,
      Urdu, and mixed/code-switched speech.
   e. Sends the raw transcript to a Groq-hosted LLM
      (`llama-3.3-70b-versatile`) to remove filler words, fix
      punctuation/paragraphs, and clean up formatting — without
      changing the meaning, language, or shortening/summarizing the
      content (`app/services/llm.py`).
   f. Sends the cleaned transcript back to the user on WhatsApp.
5. If the user sends anything other than a voice note (text, image,
   sticker, etc.), the bot replies asking them to send a voice note
   instead.
6. If any step in the pipeline fails, the user gets a friendly WhatsApp
   error message instead of silence.

## Project structure

```
voicenotes/
├── app/
│   ├── main.py                    # FastAPI app entrypoint (routes, CORS, logging, lifespan)
│   ├── config.py                  # Settings loaded from environment variables / .env
│   ├── routers/
│   │   └── webhook.py             # GET/POST /webhook — verification + receiving + dispatch
│   ├── services/
│   │   ├── whatsapp.py            # send_text_message(), get_media_url(), download_media()
│   │   ├── transcription.py       # ffmpeg conversion + Groq Whisper transcription
│   │   ├── llm.py                 # Groq LLM transcript cleanup
│   │   └── voice_processing.py    # Orchestrates the full voice-note pipeline
│   └── models/                    # (placeholder for typed schemas in later phases)
├── .env.example
├── requirements.txt
└── README.md
```

## 1. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate       # Windows PowerShell
# source venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

Note: `imageio-ffmpeg` bundles a static ffmpeg binary for your platform —
you do **not** need to install ffmpeg separately or add it to PATH.

## 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your real credentials:

```bash
copy .env.example .env      # Windows
# cp .env.example .env      # macOS/Linux
```

Then edit `.env`:

```
WHATSAPP_TOKEN=your_meta_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_VERIFY_TOKEN=choose_any_secret_string_here
GROQ_API_KEY=your_groq_api_key_here
APP_ENV=development
```

- `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` come from the
  [Meta App Dashboard](https://developers.facebook.com/apps/) under your
  app's WhatsApp > API Setup page.
- `WHATSAPP_VERIFY_TOKEN` is **any string you make up yourself** — you'll
  enter the exact same value in the Meta dashboard when configuring the
  webhook (step 4 below).
- `GROQ_API_KEY` comes from [console.groq.com/keys](https://console.groq.com/keys)
  — used for both voice transcription and transcript cleanup.

**Never commit your real `.env` file** — it's already excluded via
`.gitignore`. All code reads credentials exclusively from environment
variables via `app/config.py`; nothing is hardcoded.

## 3. Run the server locally

```bash
uvicorn app.main:app --reload --reload-dir app --port 8000
```

`--reload-dir app` limits the auto-reload file watcher to the `app/`
folder. Without it, `--reload` watches the *entire* project directory by
default — including `venv/` — so every `pip install` triggers a storm of
unnecessary server restarts as it detects hundreds of new files.

Verify it's up:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

## 4. Expose your local server to the internet (for Meta's webhook)

Meta's servers need to reach your webhook over the public internet — they
can't call `localhost` directly. During development, use a tunneling tool
like [ngrok](https://ngrok.com/) to expose your local port:

```bash
ngrok http 8000
```

This gives you a public HTTPS URL like `https://abcd1234.ngrok-free.dev`,
which forwards requests to your local `localhost:8000`. Note: this URL
changes every time you restart ngrok (on the free plan), so you'll need
to update the webhook config in the Meta dashboard each time.

Then, in the Meta App Dashboard, under **WhatsApp > Configuration**, set the
webhook URL to:

```
https://abcd1234.ngrok-free.dev/webhook
```

and set the **Verify Token** field to the same value you used for
`WHATSAPP_VERIFY_TOKEN` in your `.env`. Meta will call `GET /webhook` to
verify ownership before saving the configuration — the app handles that
automatically as long as the token matches.

Finally, click **Manage** next to webhook fields and subscribe to
`messages` — without this, Meta won't send you incoming message events at
all, even if the callback URL itself verified successfully.

## 5. Try it out

Send a **voice note** to your WhatsApp test number. You should:
1. Get an immediate "⏳ Processing your voice note..." reply.
2. A few seconds later, get the cleaned-up transcript back.

Send a **text message** instead, and you'll get a reply asking you to send
a voice note.

Watch the `uvicorn` terminal for structured log lines tracing each step of
the pipeline (media download, ffmpeg conversion, transcription, cleanup).
