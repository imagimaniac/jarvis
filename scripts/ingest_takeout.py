#!/usr/bin/env python3
"""Minimal Takeout-style importer (text files/dirs) - user-triggered, read-only."""
from pathlib import Path
import sys

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

from app.core import storage, utils

def ingest_dir(takeout_dir: Path, source: str = 'takeout', origin_kind: str = 'external_record'):
    if not takeout_dir.exists():
        print('not found', takeout_dir)
        return
    dbp = base / 'data' / 'jarvis.db'
    raw_dir = base / 'memory' / 'raw'
    db = storage.JarvisDB(dbp)
    count = 0
    for p in takeout_dir.rglob('*.txt'):
        try:
            text = p.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if not text.strip():
            continue
        dt = utils.utc_now()
        fpath, relpath, ch = utils.write_raw(raw_dir, text, dt)
        db.insert_entry(text=text, source=source, origin_kind=origin_kind, source_type='text', origin_id=str(p), external_ref=str(p), import_batch_id='takeout', raw_path=fpath, raw_relpath=relpath, content_hash=ch)
        count += 1
    db.close()
    print('ingested', count)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('usage: ingest_takeout.py <dir>')
        sys.exit(1)
    ingest_dir(Path(sys.argv[1]))
