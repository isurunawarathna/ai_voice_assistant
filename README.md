# AI Voice Assistant (Sinhala / English)

Backend for the campus voice assistant. This release covers the **first minute of
a call**: the welcome message, language selection and the confirmation message.
Everything after that is handed to a pluggable agent, which is where the hybrid
agentic RAG goes.

```
caller ──WebSocket──► /api/v1/voice
                        │
                        ▼
   CONNECTING → WELCOME → LANGUAGE_SELECTION → ACTIVE ⇄ THINKING → CLOSED
                  │              │                │
          welcome_message   select_language    ConversationAgent  ← your hybrid RAG
              (WAV)       (speech / text / tap)   (EchoAgent today)
                            confirm prompt (WAV)
```

## Run it

```bash
uv sync --extra dev --extra ui      # or: pip install -r requirements-dev.txt -r requirements-ui.txt
cp .env.example .env

uvicorn app.main:app --port 8000     # terminal 1: the backend (loads Whisper on start)
streamlit run ui/streamlit_app.py    # terminal 2: the visual harness
```

Open http://localhost:8501 and press **Start call**. The page is a plain WebSocket
client, so it shows exactly what the Angular app will receive.

| In the UI | What it exercises |
| --- | --- |
| Speak (microphone) | Browser WAV → decode → Whisper → language detector → confirmation prompt |
| Type what they say | Transcript → language detector. Use the preset buttons (`ඉංග්ලිෂ්`, `two`, `Unclear`) to see retries and failure |
| Tap | `select_language` control message |
| Choose language again | `reset` |
| Call timeline / raw events | Every protocol event and prompt, with timings |

Production: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --ws-max-size 16777216`
with `ENVIRONMENT=production` (hides `/docs`, enforces the Origin check).

## What the first minute guarantees

**Language detection** (`app/services/language.py`) reports *why* it decided, and
refuses to guess when unsure.

- Exact keyword, a keyword Whisper split in two (`sin hala`), fuzzy keyword, short numeric choice
  (`one` / `two` / `එක` / `දෙක`), Sinhala script, Whisper language id, in that order.
- Sinhala-script spellings of "English" (`ඉංග්ලිෂ්`) are English, not Sinhala.
- Both languages in one answer is ambiguous. Look-alikes (`singles`) and sentences
  containing a number (`I have two questions`) are not a choice.

**State machine** (`app/conversation/orchestrator.py`)

- `select_language` is only accepted during language selection; `reset` is the way to change it later.
- Retry budget: `CONVERSATION__LANGUAGE_MAX_RETRIES` unrecognised answers, then `language.failed` and close (1008).
- Per-frame timeout *and* an overall deadline for the selection phase.
- Abuse budgets: consecutive unusable inputs (`MAX_INVALID_INPUTS`) and `reset`s (`MAX_RESETS`) are capped,
  because every reset replays a ~1.4 MB prompt.
- Startup fails fast if a prompt file is missing or corrupt (`AUDIO__STRICT_PROMPT_LOADING`, forced on in production).

**Admission control** (`app/api/ws/guard.py`) runs *before* the socket is accepted.

- Browser `Origin` must be in `SECURITY__ALLOWED_ORIGINS` (defaults to `CORS_ORIGINS`). Clients with no
  `Origin` header (scripts, Streamlit, telephony) are not browsers and pass.
- Per-IP concurrent connections and connections per minute.
- Refusals are HTTP 403 and never stream the welcome audio.

**Observability**

- Every log line carries `trace_id` and `session_id`; the trace id is passed on to the agent.
- `GET /metrics` (Prometheus text): `voice_connections_total{result}`, `voice_onboarding_outcomes_total{outcome}`
  (`selected`, `failed`, `timeout`, `abandoned`, `too_many_invalid_inputs`, `too_many_resets`),
  `voice_language_selected_total{language,method}`, `voice_language_attempts_total{reason}`,
  `voice_language_selection_seconds`, `voice_active_conversations`.
- `GET /health/live`, `GET /health/ready` (503 until prompts, STT, sessions and agent are ready).

The number to watch: **`abandoned / total` outcomes**. A rise means callers are hanging up before choosing a language.

## Protocol additions (still v1, additive)

`language.selected` now also carries `method` (`voice` | `text` | `ui`), `confidence` and `reason`.
New error codes: `language_timeout`, `too_many_invalid_inputs`, `too_many_resets`, `unexpected_input`.

## Plugging in the hybrid RAG agent

Nothing in the first minute changes. Implement `ConversationAgent` (`app/services/agent/base.py`), then:

```
AGENT__BACKEND=hybrid_rag
AGENT__IMPORT_PATH=my_rag.pipeline:build_agent
```

The agent receives `AgentRequest(query, language, history, trace_id, metadata)`.
`metadata["language_method"]` says how the caller chose their language. For spoken replies, implement
`TextToSpeechEngine` and register it in `app/container.py`.

## Tests

```bash
pytest            # 123 tests, ~1 s, no model download or microphone needed
ruff check app tests scripts ui
```

## Layout

```
app/
  api/            routes (voice, health, /metrics), WebSocket transport, connection guard
  audio/          codec + pre-recorded prompt library
  conversation/   orchestrator (state machine) and transport port
  core/           config, logging, trace context, metrics, errors
  domain/         enums, models, wire protocol
  services/       language detector, STT, TTS port, agent port
  session/        memory + Redis stores
ui/               Streamlit harness (session_client.py = WebSocket client, view_model.py = page logic)
assets/audio/     welcome, retry and confirmation recordings
```
