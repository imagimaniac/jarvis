# RESEARCH_GITHUB.md - Jarvis Architecture Research
(Plan Mode - Read-only notes)

## 1. vladimanaev/personal-memory
- Local-first RAG; Markdown source of truth; rebuildable index; local embeddings; skills/CLI+UI; plugin model.
- Borrow: rebuildable index (reinforces raw-as-sacred), skills modularity, readable exports alongside raw (non-destructive).
- Don't: Node stack/UI-first vs terminal-first.
- Relevance: strong alignment on local-first + source of truth.

## 2. snedea/aiassistant
- CalDAV/IMAP + Apple Notes; background scanners; ChromaDB+SQLite; FastAPI+React; facts/memory; triage; quiet hours; notifications.
- Borrow: ingestion isolated (scanners) but user-triggered default; triage (CRITICAL/HIGH/MEDIUM/LOW/IGNORE); quiet hours/thresholds; separate facts from raw.
- Don't: Apple Notes hardcoded; broad assistant scope.
- Relevance: Gmail/Calendar ingestion + classification + prioritization.

## 3. bmitioglov/personal-assistant
- ICS+Gmail IMAP/SMTP; reasoning separated from data/delivery; deterministic fallback; notes.md preferences; run non-interactively.
- Borrow: reasoning vs data+delivery separation; deterministic fallback; notes.md for standing prefs; payload boundary.
- Don't: email delivery as primary goal.
- Relevance: reliability + clean separation.

## 4. letta-ai/letta-code
- Stateful agent; memory blocks; MemFS (git-tracked); skills/subagents; crons/schedules; self-improvement; hooks/permissions; persistent identity.
- Borrow: memory blocks/evolving system instructions for "partly designed, partly discovered" personality; MemFS for tracking evolution (exclude raw private data); opt-in crons/schedules; persistent identity.
- Don't: aggressive self-rewriting/modding.
- Relevance: personality evolution from accumulated evidence.

## 5. open-webui/open-webui
- Extensible (Pipes/Filters/Actions/Tools/Skills, MCP); RBAC; artifacts; memory; calendar/automations; modular pipeline.
- Borrow: Pipes/Filters pipeline (ingest->classify->link->store); artifacts separate from raw (matches syntheses); RBAC/permissions; modular skills.
- Don't: web-heavy UI.
- Relevance: clean modular pipeline + artifact boundary.
