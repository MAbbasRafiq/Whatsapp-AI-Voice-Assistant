# VoiceNotes — WhatsApp Bot

VoiceNotes is a WhatsApp bot built with FastAPI that turns voice notes into
clean, well-formatted text.

- **Phase 1**: project skeleton + a working webhook that verifies itself
  with Meta, receives incoming WhatsApp messages, and sends plain text
  replies back.
- **Phase 2**: full voice note pipeline — download the audio from Meta,
  convert it with ffmpeg, transcribe it with Groq's Whisper API, clean up
  the transcript with a Groq LLM, and reply with the result.
- **Phase 3** (current): production hardening — PostgreSQL persistence,
  webhook signature verification, deduplication, a blocklist, rate/daily
  usage limits, audio pre-checks, per-user language preferences, slash
  commands, long-message chunking, structured analytics logging, and
  Railway deployment config.

## How it works

1. User sends a voice note (or a text message/command) on WhatsApp.
2. Meta POSTs the message to `/webhook`. The handler:
   a. Verifies the `X-Hub-Signature-256` HMAC header against `APP_SECRET`
      — requests that don't verify get a `403` and are never processed.
   b. Responds `200 OK` immediately (required by WhatsApp) and schedules
      the rest of the work as a background task per message.
3. Background task (`app/routers/webhook.py::_handle_message`):
   a. Ensures a `users` row exists for the sender and updates `last_seen`.
   b. Skips silently if the user is blocked (`users.is_blocked`).
   c. Inserts into `messages` for dedup — if `wa_message_id` already
      exists (a Meta webhook retry), skips silently with no reply.
   d. Dispatches:
      - **Voice/audio** → rate limit (1 per 10s) → the full pipeline in
        `app/services/voice_processing.py` (daily quota check → download
        → file size / duration pre-checks → ffmpeg → Groq Whisper → Groq
        LLM cleanup, honoring any saved language preference → save
        transcript → chunked reply → structured analytics log line).
      - **`/command` text** → `app/services/commands.py`.
      - **Anything else** → a friendly nudge reply.
4. If any step fails, the user gets one of the specific friendly error
   messages below (never a raw/internal error), and the full exception
   is logged server-side.

### Slash commands

| Command      | Effect                                                                 |
|--------------|-------------------------------------------------------------------------|
| `/translate` | Re-runs the LLM (not Whisper) on your last transcript, in English      |
| `/summarize` | Re-runs the LLM on your last transcript as concise bullet points        |
| `/urdu`      | Saves your language preference — future transcripts reply in Urdu (اردو) |
| `/english`   | Saves your language preference — future transcripts reply in English   |
| `/roman`     | Saves your language preference — future transcripts reply in Roman Urdu |
| `/help`      | Shows the command list                                                 |

`/translate` and `/summarize` operate on an in-memory "last transcript"
cache (10-minute expiry, one slot per phone number, overwritten by each
new voice note) — they never re-run Whisper, only the cheaper LLM step.

> **Design note on `/roman`:** the original Phase 3 spec listed `/roman`
> with two conflicting meanings (a one-off "convert last transcript"
> action, and a persistent preference-setter, like `/urdu`/`/english`).
> Per your choice, `/roman` here is a **preference-setter only** — it
> does not re-render your last transcript. If you'd also like a one-off
> "convert to Roman Urdu right now" action, that would need a new command
> name (e.g. `/torroman`) since `/roman` is already taken by the
> preference-setter.

### First-time experience & footers

- **Brand-new user** (first message ever): after their first transcript,
  the bot appends an onboarding block explaining `/urdu`, `/english`,
  `/roman` so they can set a permanent preference — without blocking
  anything, they get their transcript immediately either way.
- **Every other successful transcript** (returning users, regardless of
  whether they've set a preference): a minimal footer is appended
  instead — `──────────────` + `💡 /translate • /summarize • /roman • /help`.
- If `users.preferred_language` is set and differs from what was actually
  spoken, the LLM cleanup step targets that language/script directly
  (one LLM call, not a cleanup-then-translate round trip).

### Limits & pre-checks

- **Rate limit**: 1 voice note per phone number per 10 seconds (in-memory,
  resets on restart).
- **Daily quota**: `DAILY_VOICE_LIMIT` voice notes per user per UTC day
  (default 20), tracked in `usage_daily`, atomic upsert via
  `INSERT ... ON CONFLICT ... DO UPDATE`.
- **File size**: rejected if over 25MB (checked via Meta's media metadata
  before downloading, and again on the downloaded bytes as a fallback).
- **Duration**: rejected if over 3 minutes, probed via `ffmpeg` (parsing
  its stderr `Duration:` line — `imageio-ffmpeg` doesn't bundle `ffprobe`).

## Project structure

```
voicenotes/
├── app/
│   ├── main.py                    # FastAPI entrypoint (routes, CORS, logging, DB startup check)
│   ├── config.py                  # Settings loaded from environment variables / .env
│   ├── database.py                # Async SQLAlchemy engine + session factory + connection test
│   ├── models.py                  # ORM models: users, messages, usage_daily, transcripts
│   ├── utils.py                   # mask_phone(), structured analytics logging helper
│   ├── routers/
│   │   └── webhook.py             # GET/POST /webhook — sig verification, dedup, blocklist, dispatch
│   └── services/
│       ├── whatsapp.py            # send_text_message(), send_long_message() (chunking), media download
│       ├── security.py            # X-Hub-Signature-256 HMAC verification
│       ├── rate_limiter.py        # in-memory 1-voice-note-per-10s limiter
│       ├── db_ops.py              # all async DB reads/writes (users, messages, usage_daily, transcripts)
│       ├── transcript_cache.py    # in-memory "last transcript" cache (10 min) for /translate, /summarize
│       ├── commands.py            # /translate /summarize /urdu /english /roman /help
│       ├── transcription.py       # ffmpeg conversion, duration probe, Groq Whisper transcription
│       ├── llm.py                 # Groq LLM cleanup/translate/summarize prompts
│       └── voice_processing.py    # orchestrates the full voice-note pipeline
├── migrations/                    # Alembic migration scripts (async)
├── alembic.ini
├── Procfile                       # Railway/Heroku-style process declaration
├── railway.toml                   # Railway build/deploy config
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
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/dbname
APP_SECRET=your_meta_app_secret_here
DAILY_VOICE_LIMIT=20
APP_ENV=development
```

- `WHATSAPP_TOKEN` and `WHATSAPP_PHONE_NUMBER_ID` come from the
  [Meta App Dashboard](https://developers.facebook.com/apps/) under your
  app's WhatsApp > API Setup page.
- `WHATSAPP_VERIFY_TOKEN` is **any string you make up yourself** — you'll
  enter the exact same value in the Meta dashboard when configuring the
  webhook (step 4 below).
- `GROQ_API_KEY` comes from [console.groq.com/keys](https://console.groq.com/keys)
  — used for voice transcription and all LLM cleanup/translate/summarize
  calls.
- `DATABASE_URL` is your PostgreSQL connection string, using the
  `+asyncpg` driver suffix (e.g. a Railway Postgres plugin gives you a
  `postgresql://...` URL — rewrite the scheme to `postgresql+asyncpg://`).
- `APP_SECRET` is from **Meta App Dashboard → App Settings → Basic → App
  Secret**. Used to verify that incoming webhook requests genuinely came
  from Meta (HMAC-SHA256 of the raw request body). Requests that fail
  this check get an immediate `403` before any other processing.
- `DAILY_VOICE_LIMIT` (default `20`) is the max voice notes a single user
  can process per UTC calendar day.

**Never commit your real `.env` file** — it's already excluded via
`.gitignore`. All code reads credentials exclusively from environment
variables via `app/config.py`; nothing is hardcoded.

## 3. Set up the database

This app uses PostgreSQL (via async SQLAlchemy + asyncpg) with 4 tables:
`users`, `messages`, `usage_daily`, `transcripts`. Table creation is
handled entirely by **Alembic migrations** — the app itself never creates
or alters tables at runtime (see `app/database.py` / `app/main.py`: startup
only tests connectivity and logs success/failure, it never runs
migrations automatically).

Once `DATABASE_URL` is set in `.env`, run the migration manually:

```bash
alembic upgrade head
```

**Run this manually before the first deploy, and again after every schema
change** (i.e. whenever a new migration file is added to
`migrations/versions/`). On Railway, run it via:

```bash
railway run alembic upgrade head
```

To create a new migration after changing `app/models.py`:

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

## 4. Run the server locally

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

Watch the startup logs for `Database connection established.` (or a
`Database connection failed` error if `DATABASE_URL` is wrong/unreachable
— the app still starts either way, just with DB-dependent features
degraded, per Phase 3's "never crash on DB failure" design).

## 5. Expose your local server to the internet (for Meta's webhook)

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

## 6. Try it out

Send a **voice note** to your WhatsApp test number. You should:
1. Get an immediate "⏳ Processing your voice note..." reply.
2. A few seconds later, get the cleaned-up transcript back, with a
   language header and (first time) an onboarding footer or (afterward) the
   minimal command footer.

Send a **text message** instead, and you'll get a reply asking you to send
a voice note (or use `/help`). Send `/help`, `/urdu`, `/english`, `/roman`,
`/translate`, or `/summarize` to try the commands.

Watch the `uvicorn` terminal for structured log lines tracing each step of
the pipeline, plus a single JSON analytics line (`"event":
"voice_note_processed"`) after each voice note attempt — see
`app/utils.py::log_voice_note_event` for the exact schema (phone numbers
are logged as last-4-digits only, for privacy).

## 7. Deploying to Railway

1. Create a new Railway project from this repo, and add a **PostgreSQL**
   plugin — Railway will inject `DATABASE_URL` (as `postgresql://...`;
   rewrite it to `postgresql+asyncpg://...` in your Railway service's
   environment variables, since that's the driver our async engine uses).
2. Set the remaining environment variables in the Railway service
   (`WHATSAPP_TOKEN`, `WHATSAPP_PHONE_NUMBER_ID`, `WHATSAPP_VERIFY_TOKEN`,
   `GROQ_API_KEY`, `APP_SECRET`, `DAILY_VOICE_LIMIT`, `APP_ENV=production`).
3. Deploy. Railway uses `railway.toml` (nixpacks builder, start command,
   `/health` healthcheck) — no `Procfile` changes needed for Railway
   specifically, but it's included for Heroku-style compatibility too.
4. **Before the app can actually work**, run the migration once against
   the deployed database:
   ```bash
   railway run alembic upgrade head
   ```
   Repeat this after every future schema change — it is **never** run
   automatically by the app or by `Procfile`/`railway.toml`, by design.
5. Update the webhook URL in the Meta App Dashboard to your Railway
   service's public URL (`https://<your-app>.up.railway.app/webhook`).

## Notes / what's NOT implemented yet

- No `/history` command yet, though `transcripts` already stores
  everything needed for one in a future phase.
- Rate limiting and the last-transcript cache are in-memory and
  per-process — they reset on restart and won't work correctly if this
  app is ever scaled to multiple concurrent instances (fine for a single
  Railway web process today; would need a shared store like Redis to scale
  horizontally).
- Blocking a user is manual, direct-SQL only (`UPDATE users SET
  is_blocked = true WHERE wa_phone = '...'`) — no admin command/API yet.
