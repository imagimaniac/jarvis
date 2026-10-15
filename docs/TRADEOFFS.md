# TRADEOFFS.md (Plan Mode - Key tradeoffs)

- Terminal-first vs UI: keep CLI/TUI primary (matches spec). UI nice later, not MVP.
- User-triggered vs scheduled sync: user-triggered default (safer, offline-first). Add opt-in scheduled with quiet hours later.
- Heuristic vs LLM extraction: heuristic-first (fast/offline). LLM optional, conservative, inferred_by marked.
- Self-boost vs neutral: boost origin_kind=self for "thinking/deciding" queries to avoid external noise; keep explicit source filtering.
- Personality evolution vs stability: partly discovered but require evidence + reviewable memory blocks; don't inject assumptions from day1.
- Deterministic fallback vs always-LLM: prefer deterministic for reliability (esp. reminders/classification).
- Raw git-tracked? No: raw memory/private data should not be auto-committed. MemFS-like tracking only for summaries/config/personality blocks, never full raw corpus.
