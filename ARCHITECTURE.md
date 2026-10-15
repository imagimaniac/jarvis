# Jarvis — Architecture

## Overview
Jarvis is a local-first personal memory + information layer with explicit source
separation (self vs. external) and read-only, pull-only external ingestion.
Core capture/search/retrieve works fully offline. Sync/ingestion is opt-in,
online-only where needed, and never writes back to external services.

## Layers

1. **CLI / TUI (primary)**
   - `app/cli.py`: `add | search | ask | show | recent | stats`
   - Deterministic Rich output; the CLI is the default capture surface.

2. **Local HTTP API + PWA**
   - `app/server.py`: FastAPI, binds `127.0.0.1:8123`, Bearer auth
     (`JARVIS_TOKEN`), SSE streaming for chat/evidence events.
   - `static/`: offline-first web app (index.html, manifest, service worker).
   - Lifecycle scripts (`scripts/start|stop|status|restart_jarvis.sh`) enforce
     the loopback-only binding contract.

3. **Core logic**
   - Capture (self input → raw).
   - Ingest (Gmail/Takeout, read-only) — user-triggered, opt-in.
   - Extract candidates (heuristic-first; LLM optional behind flags).
   - Classify importance/category (deterministic heuristics).
   - Search/Retrieve: FTS5 + recency/self boosts.
   - Context assembly: bounded, auditable prompt (see below).
   - Reminders / proactive surface: **Phase 5 — planned, not built.**

4. **Local memory layer**
   - **Raw (sacred)**: append-only files `memory/raw/YYYY/MM/DD/` with
     deterministic IDs, content_hash (SHA-256) integrity.
   - **Structured**: SQLite `data/jarvis.db` (schema below).
   - **Index**: SQLite FTS5 virtual tables; semantic search pluggable later.
   - **Config**: `config.yaml` (local-only overrides; gitignored).

5. **Providers (model-replaceable)**
   - `app/providers/llm.py`: OpenAI-compatible interface, Ollama default at
     `http://localhost:11434/v1`. No cloud fallback. Core has no hard
     dependency on the LLM for search/recall/context assembly.

## Data model (SQLite)

- **entries** — raw envelope: uuid, created_at (utc/local + tz), source_type,
  source (`cli|takeout|gmail|other`), `origin_kind` (`self|external_record`),
  origin_id (dedup), external_ref, raw_path, content_hash, text_raw, sizes.
- **structured** — extracted facets per entry: type, date, mood, people,
  places, projects, topics, events, questions, decisions, learnings, tags,
  category, importance, confidence, inferred_by, needs_review, is_sensitive.
- **relations** — cross-source links with confidence + rationale.
- **sources / meta / imports** — traceability, key/value, import audit.
- **syntheses** — model-produced summaries, labeled SYNTHESIZED, never written
  over facts.
- **memories / canonical memory** — confirmed durable memories with
  lifecycle state (CONFIRM/UPDATE/SUPERSEDE/FORGET), importance, confidence,
  sensitivity flag, provenance; context assembly reads this layer.
- **external_records** — normalized pull-only third-party records (gmail),
  with `author` for sender/date-constrained retrieval.
- **context_topics / threads** — conversation continuity state.
- **events_seen / event_state** — planned reminders bookkeeping.
- **FTS5**: `entries_fts` over text fields, kept in sync on insert/delete.

## Source separation & epistemics
- `origin_kind = self`: what you wrote. Memory/facts about your own thinking.
- `origin_kind = external_record`: Gmail/Takeout records; presented as
  third-party evidence, never as "you said" unless explicitly quoted.
- Default lens: `self_boost` ranks your own entries higher for memory-shaped
  queries; external records are labeled by source.
- Extraction is conservative: no invented feelings/decisions unless clearly
  stated; sensitive candidates (health, identity, political/religious) are
  never persisted even when a model proposes them.

## Retrieval & context assembly
- Search-first: FTS5 keyword retrieval with recency boost, type weighting, and
  self/external separation; top-k evidence with score, snippet, timestamp,
  source, origin_kind.
- **Recall** (`app/core/recall.py`) consolidates evidence, dedupes, renders.
- **Interaction policy** (`app/core/interaction_policy.py`) derives sender and
  date-window constraints from natural-language questions ("from X yesterday")
  and narrows external-record retrieval accordingly.
- **Context engine** (`app/core/context_engine.py`) cross-rates canonical
  memory + retrieved evidence under a token budget
  (MAX_PROMPT_CHARS/MAX_MEMORY_CHARS/MAX_MEMORY_ITEMS/MAX_EVIDENCE_ITEMS),
  separating memory context from citation evidence.
- **Citation check**: post-generation audit marks each citation
  cited/uncited/invented.
- `ask` may synthesize only via the disabled-by-default retrieval-synthesis
  flag; every synthesis is labeled and source-listed.

## Memory lifecycle
- Deterministic candidate extraction (events, preferences, decisions, history)
  → gated Memory Judge (`features.memory_judge`, off by default) → canonical
  memory write. Canonical entries are immutable points in a timeline: updates
  supersede, they never rewrite history. Seed manifests (`canonical_memories`)
  enter the same candidate→confirm path with explicit audit reasons.

## Sync architecture
- Pull-only, one-way, user-triggered; no two-way sync, no write-back.
- Gmail incremental connector (read-only OAuth via OS credential store) and
  Google Takeout import both land in `external_records`.
- Local DB + raw files are canonical for Jarvis; remote services are external
  read-only sources of truth.

## Security & privacy
- Loopback-only binding enforced; `JARVIS_TOKEN` fail-closed auth; tokens never
  in URLs.
- Read-only scopes; OTP/reset-token redaction; `is_sensitive` flag on ingestion.
- No silent cloud upload. `data/` and `memory/` are gitignored. This repository
  is public code, never personal data.