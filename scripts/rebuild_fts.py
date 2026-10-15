#!/usr/bin/env python3
"""Rebuild entries' text and the FTS5 index from memory/raw (the source of truth)."""
from pathlib import Path
import sys

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

from app.core import rebuild

dbp = base / 'data' / 'jarvis.db'
raw = base / 'memory' / 'raw'
r = rebuild.rebuild_fts(dbp, raw)
print(
    f"restored {r['restored']} from raw, created {r['created']} new, "
    f"{r['unbacked']} entries without a raw file, {r['fts_rows']} indexed"
)