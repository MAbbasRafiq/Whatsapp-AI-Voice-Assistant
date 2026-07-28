# VoiceNotes — WhatsApp Bot

VoiceNotes is a WhatsApp bot built with FastAPI that turns voice notes into
clean, well-formatted text — and plain text into speech via Reply Buttons.

- **Phase 1**: project skeleton + a working webhook that verifies itself
  with Meta, receives incoming WhatsApp messages, and sends plain text
  replies back.
- **Phase 2**: full voice note pipeline — download the audio from Meta,
  convert it with ffmpeg, transcribe it with Groq's Whisper API, clean up
  the transcript with a Groq LLM, and reply with the result.
- **Phase 3** (current): production hardening — PostgreSQL persistence,
  webhook signature verification, deduplication, a blocklist, rate/daily
  usage limits, audio pre-checks, button-based Translate / Summarize /
  Help / Text-to-Voice, long-message chunking, structured analytics
  logging, and Railway deployment config.

## How it works

1. User sends a voice note or text on WhatsApp.
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
      - **Voice/audio** → per-phone rate limit + in-flight lock → the full
        pipeline in `app/services/voice_processing.py` (daily quota
        reserve → download → file size / duration pre-checks → ffmpeg →
        Groq Whisper → Groq LLM cleanup **in the spoken language** →
        save transcript → chunked reply → Reply Buttons → analytics log).
      - **Interactive buttons / lists** → Translate, Summarize, Help,
        language picker, Listen, or Convert to Voice.
      - **Plain text (≥ 5 meaningful characters)** → cache text (15 min)
        → offer **🔊 Convert to Voice**.
      - **Short / other messages** → friendly guidance nudge.
4. If any step fails, the user gets one of the specific friendly error
   messages (never a raw/internal error), and the full exception is
   logged server-side.

### Button-based UX

All primary actions use WhatsApp Reply Buttons / List Messages:

| Action | When |
|--------|------|
| **🌍 Translate** | After a transcript — opens a language list (English, Urdu, Roman Urdu, Arabic, French, Chinese, or Other…) |
| **📝 Summarize** | After a transcript — bullet-point summary in the transcript's own language |
| **❓ Help** | After a transcript — short how-to |
| **Listen** | After a translation or summary — speaks that result aloud |
| **🔊 Convert to Voice** | After the user sends plain text (≥ 5 letters/numbers) |

Transcripts are always cleaned up in the **original spoken language**.
Users translate themselves by tapping Translate and picking a language —
there is no saved language preference.

Translate / Summarize / Listen operate on in-memory caches (15-minute
TTL, one slot per phone number). They never re-run Whisper.

> **Note:** `/voice <text>` still works for backward compatibility but is
> **not advertised** in help or nudges. Prefer sending text and tapping
> Convert to Voice.

### Limits & pre-checks

- **Voice rate limit**: 1 accepted voice note per phone per 10 seconds,
  plus an in-flight lock so a second note cannot start while the first
  is still processing (in-memory, resets on restart, single process).
- **TTS rate limit**: same pattern for Convert to Voice / Listen /
  `/voice` (busy lock + cooldown).
- **Daily quota**: `DAILY_VOICE_LIMIT` successful voice notes per user
  per UTC day (default 20). A slot is reserved up front and **refunded**
  if the pipeline fails before a usable transcript is saved. Tracked in
  `usage_daily` via conditional `INSERT ... ON CONFLICT ... DO UPDATE`.
- **File size**: rejected if over 25MB (Meta media metadata, then
  downloaded bytes as a fallback).
- **Duration**: rejected if over 3 minutes, probed via `ffmpeg` (parsing
  its stderr `Duration:` line — `imageio-ffmpeg` doesn't bundle `ffprobe`).

### Deployment constraint

In-memory caches (transcript, TTS Listen, pending text-to-voice, rate
limits, waiting-for-language) are **process-local**. Keep a **single**
uvicorn worker / Railway replica. Scaling to multiple instances requires
a shared store (e.g. Redis) first.

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
│       ├── whatsapp.py            # text / buttons / lists / media upload-download / chunking
│       ├── security.py            # X-Hub-Signature-256 HMAC verification
│       ├── rate_limiter.py        # voice + TTS cooldowns and in-flight locks
│       ├── db_ops.py              # async DB reads/writes (users, messages, usage, transcripts)
│       ├── transcript_cache.py    # last transcript (15 min) for Translate / Summarize
│       ├── tts_cache.py           # last summary/translation (15 min) for Listen
│       ├── text_to_voice_cache.py # pending plain text (15 min) for Convert to Voice
│       ├── user_state.py          # waiting_for_language after "Other language..."
│       ├── commands.py            # interactive handlers + plain-text /voice compat
│       ├── transcription.py       # ffmpeg conversion, duration probe, Groq Whisper
│       ├── llm.py                 # Groq LLM cleanup / translate / summarize
│       ├── tts.py                 # Edge TTS synthesis
│       ├── voice_processing.py    # orchestrates the full voice-note pipeline
│       ├── user_errors.py         # classified friendly WhatsApp error replies
│       └── voice_storage.py       # fire-and-forget Supabase Storage archival
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
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
SUPABASE_STORAGE_BUCKET=voice-recordings
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
- `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` come from **Supabase
  Dashboard → Project Settings → API**. Used only to archive raw voice
  notes to a private Storage bucket in the background (never blocks
  replies). Leave blank to skip archival.
- `SUPABASE_STORAGE_BUCKET` (default `voice-recordings`) is the private
  bucket name. Object keys are
  `<phone_number>/<whatsapp_message_id>.ogg` (or the original extension).

**Never commit your real `.env` file** — it's already excluded via
`.gitignore`. All code reads credentials exclusively from environment
variables via `app/config.py`; nothing is hardcoded.

### Supabase Storage bucket (voice archival)

Create a **private** bucket once in the Supabase dashboard (Storage → New
bucket):

1. Name: `voice-recordings` (must match `SUPABASE_STORAGE_BUCKET`).
2. Public bucket: **off** (keep private).
3. No extra Postgres tables or app-side RLS policies are required — the
   bot uploads with the **service role** key, which bypasses Storage RLS.

Uploads are scheduled with `asyncio.create_task` right after a successful
media download and are **never awaited** before transcription or WhatsApp
replies. Upload failures are logged only.

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

**Voice note → transcript**
1. Send a voice note to your WhatsApp test number.
2. Get an immediate "⏳ Processing your voice note..." reply.
3. A few seconds later, get the cleaned-up transcript (spoken language)
   plus Reply Buttons: Translate · Summarize · Help.

**Translate / Summarize**
- Tap **Translate**, pick a language (or Other… and type one).
- Tap **Summarize** for bullet points in the transcript's language.
- After either result, tap **Listen** to hear it as audio.

**Text → voice**
- Send any message with at least 5 letters/numbers.
- Tap **🔊 Convert to Voice**.

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
   `GROQ_API_KEY`, `APP_SECRET`, `DAILY_VOICE_LIMIT`, `SUPABASE_URL`,
   `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`,
   `APP_ENV=production`).
3. Deploy. Railway uses `railway.toml` (nixpacks builder, start command,
   `/health` healthcheck) — no `Procfile` changes needed for Railway
   specifically, but it's included for Heroku-style compatibility too.
   Keep **one** web replica (in-memory caches are not shared).
4. **Before the app can actually work**, run the migration once against
   the deployed database:
   ```bash
   railway run alembic upgrade head
   ```
   Repeat this after every future schema change — it is **never** run
   automatically by the app or by `Procfile`/`railway.toml`, by design.
5. Update the webhook URL in the Meta App Dashboard to your Railway
   service's public URL (`https://<your-app>.up.railway.app/webhook`).
