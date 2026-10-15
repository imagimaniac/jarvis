#!/usr/bin/env python3
"""Recompute content_hash from raw files for pre-existing rows.

Rows written before the bytes-on-disk contract were hashed without the trailing
newline, so their stored hash disagrees with the file. Only the hash column is
updated; raw files are never modified.

Dry-run by default. Pass --apply to write. Back up data/jarvis.db first.
"""
from pathlib import Path
import shutil
import sys

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

from app.core import rebuild

dbp = base / 'data' / 'jarvis.db'

if '--apply' not in sys.argv:
    r = rebuild.rehash(dbp, apply=False)
    print(f"dry run: {r['count']} stale row(s); nothing written. Re-run with --apply to fix.")
    for s in r['stale']:
        print(' ', s)
    sys.exit(0)

backup = dbp.with_suffix('.db.pre-rehash.bak')
shutil.copy2(dbp, backup)
print('backup ->', backup)

r = rebuild.rehash(dbp, apply=True)
print(f"rehashed {r['count']} row(s)")
for s in r['stale']:
    print(' ', s)

remaining = rebuild.verify_hashes(dbp)
print(f"verify: {len(remaining)} mismatch(es) remain")
for m in remaining:
    print(' ', m)