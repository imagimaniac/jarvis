# CONTEXT_PERSISTENCE.md

## Project
Jarvis - local-first personal memory + information layer. Terminal-first, raw sacred, source separation (self vs external), pull-only read-only external ingestion, offline-first, model-replaceable.

## Quickstart
From the repository root:
```
PYTHONPATH=. python3 -m app.cli add "thought"
PYTHONPATH=. python3 -m app.cli search "query"
PYTHONPATH=. python3 -m app.cli ask "natural query"
PYTHONPATH=. python3 -m app.cli show <id>
PYTHONPATH=. python3 -m app.cli recent
PYTHONPATH=. python3 -m app.cli stats
PYTHONPATH=. python3 -m pytest -v
```

## Key points
- Raw: memory/raw/YYYY/MM/DD/*.txt, append-only, content_hash, show() returns verbatim.
- Storage: SQLite + FTS5 (rebuildable). External records tagged origin_kind=external_record.
- Privacy: data/ and memory/ are gitignored; the repository is public code, never personal data.