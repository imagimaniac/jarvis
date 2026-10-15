# Jarvis

> A local-first, memory-aware personal AI assistant with deterministic context retrieval.

Jarvis helps you build a private, searchable model of your own context — the things you write, decide, and receive — and brings the right piece of that context into a conversation at the right time. It runs entirely on your machine.

- **Local-first**: every word stays on your system. `data/` and `memory/` are gitignored on purpose: this repository is public *code*, never your *data*.
- **Deterministic**: retrieval is keyword/FTS5-first with transparent ranking and explicit scoring. The model renders deterministic context; it cannot silently edit your memories.
- **Source-honest**: self-captured thoughts and external records (email, cloud exports) are stored separately, ranked separately, and labeled by origin.
- **Model-replaceable**: any OpenAI-compatible local endpoint (Ollama by default). Search and recall work with the model offline.
- **Fork-first**: this project is intended to be forked and personalized. Instantiate it with your own profile seed and let it accumulate context over time.

> Not affiliated with Marvel or any other holder of the "Jarvis" name.

---

## Why "deterministic context retrieval"?

Most assistants stuff a prompt with whatever vaguely matches and hope the model is right. Jarvis instead keeps a **canonical memory layer**: durable, explicitly confirmed facts (preferences, goals, projects, learnings) with provenance that says whether each claim came from you directly, was inferred, or is superseded/forgotten. At query time, a deterministic context engine cross-rates candidates (importance, confidence, recency, type) under a hard token budget and assembles a bounded, auditable context prompt. Citations are checked afterward so the model cannot invent support.

The result is a system that *remembers deliberately* — and where a wrong memory can always be traced back to the exact raw words that produced it.

---

## Practical architecture

| Component | What it is | Status |
|---|---|---|
| **CLI** (`app/cli.py`) | `add`, `search`, `ask`, `show`, `recent`, `stats` | IMPLEMENTED |
| **Local HTTP API + PWA** | FastAPI at `127.0.0.1:8123`, SSE streaming, web UI in `static/` | IMPLEMENTED |
| **Storage** | SQLite + FTS5 full-text; append-only raw files in `memory/raw/` | IMPLEMENTED |
| **Search / recall** | FTS5 keywords + recency + self-vs-external boosts, snippet scoring | IMPLEMENTED |
| **Memory candidates** | Deterministic extraction of events, preferences, decisions, history | IMPLEMENTED |
| **Canonical memory** | Confirmed memories with confirm/update/supersede/forget lifecycle + provenance | IMPLEMENTED |
| **Context engine** | Cross-rates memory + external records under a token budget | IMPLEMENTED |
| **Chat** | Assembles evidence + personal context into a prompt; citation check | IMPLEMENTED |
| **Gmail sync** | Read-only, incremental, pull-only ingestion | IMPLEMENTED, DISABLED BY DEFAULT |
| **Takeout import** | Google Takeout archive ingest | IMPLEMENTED, OPT-IN |
| **LLM extraction / retrieval synthesis** | Model-assisted extraction and answer synthesis | EXPERIMENTAL, DISABLED BY DEFAULT |
| **Reminders / proactive surface** | Unseen important items, context-aware interjection | PLANNED |

**Stack**: Python · Typer CLI · FastAPI + PWA · SQLite with FTS5 · httpx · Ollama (OpenAI-compatible). No cloud dependency outside the optional–read-only Gmail/Takeout imports.

### Runtime request flow

```text
 You ──► CLI or Web UI
              │  (JARVIS_TOKEN bearer, loopback only)
              ▼
        FastAPI server 127.0.0.1:8123
              │
        ┌─────┴────────────────────────────┐
        ▼                                  ▼
   Deterministic recall            Context assembly
   (FTS5 + boosts, evidence)  +  (canonical memory, budget)
        │                                  │
        └─────────────┬────────────────────┘
                      ▼
                 Prompt (system = context + evidence, user = question)
                      │
                      ▼
              Ollama (or any local LLM)
                      │
                      ▼
        Streamed answer + citation check (cited / uncited / invented)
```

### Memory lifecycle

```text
raw capture ──► physical events ──► deterministic candidates ──► reviewed by Memory Judge (gated)
   (append only)     (SQLite/JSON)      (needs_review)               │
                                                                    ▼
                                       canonical memory: CONFIRM / UPDATE / SUPERSEDE / FORGET
                                                                    │
                            provenance: you-said | inferred | external│
                                                                    ▼
                                context engine assembles the daily prompt
```

Canonical memories carry an explicit lifecycle. A confirmed "I want to move to X" is superseded, never overwritten; the old record stays traceable. Sensitive candidates are never persisted, even when a model says they should be.

### Public code, private data

```text
   tracked by git                     never tracked (gitignored)
┌──────────────────────────┐   ┌──────────────────────────────────┐
│ app/  scripts/  tests/    │   │ data/jarvis.db  + *.bak          │
│ seeds/*.example.json      │   │ memory/raw/YYYY/MM/DD/*.txt      │
│ docs/  static/  config.* │   │ data/seed/ (your profile seed)    │
└──────────────────────────┘   │ data/gmail_oauth_token.json       │
                               │ config.yaml  .env                 │
                               └──────────────────────────────────┘
```

Fork this repository, copy the example profiles into `data/seed/`, and fill them with facts about yourself. The code is the product; your data never leaves the boundary.

### Local deployment

```text
                ┌────────── your laptop ──────────┐
                │  127.0.0.1:8123  FastAPI server  │
                │   ┌─────────┐    ┌────────────┐  │
   phone ◄────► │   │ SQLite  │    │  Ollama    │  │        ┌ (optional) Google ┐
(Tailscale/    │   │ + FTS5  │    │ llm: llama │  │//read──►│ Gmail / Takeout   │
  SSH tunnel)  │   │ + raw/  │    └────────────┘  │ pull    └───────────────────┘
                └───────────┴─────────────────────┘
```

The server binds loopback by default and refuses non-default `JARVIS_HOST`/`JARVIS_PORT` overrides. Remote access is intended to arrive through Tailscale Serve or an SSH tunnel — never by opening the port to the network.

---

## Quick start

Requires Python 3.10+ and (for chat) a local [Ollama](https://ollama.com) instance.

```sh
git clone <your fork URL>
cd jarvis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.example.yaml config.yaml          # only if you want local overrides
export JARVIS_TOKEN="$(openssl rand -hex 32)"   # required by the API
ollama pull llama3.2                        # or any model in config
```

Then capture and search:

```sh
python3 -m app.cli add "I decided to keep a paper planner and a digital one."
python3 -m app.cli search "planner"
python3 -m app.cli ask "which planner did I decide on?"
```

Run the server + web UI:

```sh
python3 -m uvicorn app.server:app --host 127.0.0.1 --port 8123
# open http://127.0.0.1:8123 (Basic/Bearer auth via JARVIS_TOKEN)
```

Or use the lifecycle scripts that enforce the loopback-only contract:

```sh
./scripts/start_jarvis.sh
./scripts/status_jarvis.sh
./scripts/stop_jarvis.sh
```

#### Run as a login service and the `jarvis` command

On macOS, Jarvis can run as a user LaunchAgent so it starts at login, recovers
from a crash, and keeps its API token in the login Keychain instead of the
environment:

```sh
cp scripts/com.example.jarvis.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.example.jarvis.plist
```

The agent runs `scripts/launchd_start_jarvis.sh`, which reads the `jarvis-token`
Keychain item, binds only to `127.0.0.1:8123`, and fails closed if the token is
missing. `KeepAlive` restarts it only after an *unsuccessful* exit, so an
intentional stop stays stopped (it starts again at the next login). Store the
token once with:

```sh
security add-generic-password -a "$USER" -s jarvis-token -w   # prompts, never argv
```

A single-word command wraps the lifecycle scripts (install `scripts/jarvis` to
`~/.local/bin/jarvis`, which is already on `PATH`):

```sh
jarvis            # ensure running, then show status
jarvis status     # running state and health
jarvis start      # start via the Keychain-backed LaunchAgent
jarvis stop       # stop and unload the agent (starts again at next login)
jarvis restart
jarvis logs -f    # follow the server log
```

Run the test suite (no model, no network needed):

```sh
pip install -r requirements-dev.txt
python3 -m pytest
```

### Personalize the seed

Copy the template manifests into the ignored `data/seed/` directory and fill them in with *your own*, non-sensitive facts:

```sh
cp config.seed.example.json data/seed/profile.json
cp seeds/canonical_memories.example.json data/seed/canonical_personal_memories.json
# edit, then:
./scripts/apply_canonical_memory_seed.sh
```

Every example template contains placeholder text (e.g. `EXAMPLE PERSON`, `EXAMPLE CITY`). The repository ships examples and templates only — never real profile data. See `docs/CANONICAL_MEMORY_SEED.md`.

### Configure every switch

Copy `config.example.yaml` to `config.yaml` (gitignored) and read it — every option is commented. Notable defaults:

- `features.context_engine: true` — deterministic cross-rating context assembly (falls back cleanly).
- `features.memory_candidates / memory_judge: false` — the deterministic candidate producer and gated Memory Judge are opt-in; enable once you have data you want to consent to.
- `features.gmail_sync: false` — read-only, pull-only, user-triggered. Never on by default.
- `privacy.dont_log_prompts: true`, `search.mode: fts5` — deterministic by default.
- `core.timezone` — set to your own locale; the example keeps a US default.

---

## Privacy & security model

- The API refuses every request without a valid `JARVIS_TOKEN`, and refuses non-loopback bindings. Fail closed, not open.
- Secrets (`JARVIS_TOKEN`, OAuth tokens) live in the environment or the OS credential store — never in YAML, never in the repo. OTPs/reset codes are redacted from ingested text.
- External ingestion is **pull-only**: Jarvis never writes to Gmail or any cloud service.
- Your raw memory is append-only and hashed (`content_hash`); `show <id>` returns the verbatim original.
- `data/`, `memory/`, `config.yaml`, and `.env` are gitignored. This repository is intentionally public code with no personal data.

---

## Development & testing

```sh
python3 -m pytest            # ~500 tests: storage, FTS, recall, context, chat, PWA, Gmail
```

Run from source (no install): tests import the package via `pythonpath = ["."]`. Runtime deps in `requirements.txt`, test deps in `requirements-dev.txt`. Messages and edits are byte-verified where integrity matters (`scripts/verify_hashes.py`, `scripts/rebuild_fts.py`).

Useful modules: `app/core/recall.py` (search/ranking), `app/core/context_engine.py` (assembler + budget), `app/core/canonical_memory.py` (lifecycle), `app/core/interaction_policy.py` (sender/date constraints), `app/server.py` (API + SSE).

---

## Roadmap

- **Retrieval synthesis (experimental)**: optional model-assisted summarization of a retrieved evidence set — already disabled by default, needs the evidence inspector hardened.
- **Semantic/hybrid search**: pluggable embeddings behind the existing `search.mode` switch (never the default).
- **Calendar read-only sync**: a sibling of the Gmail connector (disabled by default).
- **Reminders / proactive surface**: context-aware, non-spamming interruption layer.
- **Multi-step workflows**: calendar-aware scheduling suggestions grounded in the connected email + calendar context.

Planned work is intentionally gated: nothing here reaches the deterministic memory path without consent defaults that stay off.

## Documentation

- `ARCHITECTURE.md` — layers, data model, retrieval, source separation.
- `docs/` — trade-offs, external-source integrations, canonical memory seeding, changelog, phone access.
- `config.example.yaml`, `.env.example` — annotated, commented configuration.

## License

[MIT](LICENSE)