"""Seed the canonical profile into memory.

Goes through the canonical writer path (utils.write_raw -> insert_entry ->
add_structured) so that content hashes stay correct and the database remains
fully rebuildable from raw files. It never touches an existing raw file.

Idempotent: entries are skipped when their content_hash is already present, so
re-running after editing data/seed/profile.json adds only what is new.

    PYTHONPATH=. python3 scripts/seed_memory.py --dry-run
    PYTHONPATH=. python3 scripts/seed_memory.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import models, storage, utils  # noqa: E402


def existing_hashes(db) -> set:
    return {r["content_hash"] for r in db.conn.execute(
        "SELECT content_hash FROM entries WHERE content_hash IS NOT NULL"
    )}


def seed(base: Path, seed_file: Path, *, dry_run: bool = False) -> tuple:
    cfg = {}
    for candidate in (base / "config.yaml", base / "config.example.yaml"):
        if candidate.exists():
            import yaml
            cfg = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            break
    raw_dir = base / (cfg.get("storage", {}).get("raw_dir") or "memory/raw")
    dbp = base / (cfg.get("storage", {}).get("db_path") or "data/jarvis.db")
    raw_dir.mkdir(parents=True, exist_ok=True)
    dbp.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(seed_file.read_text(encoding="utf-8"))
    facts = data.get("facts", [])
    db = storage.JarvisDB(dbp)
    seen = existing_hashes(db)
    added = skipped = 0

    for fact in facts:
        text = (fact.get("text") or "").strip()
        if not text:
            continue
        # Hash of the stored text, matching the writer contract exactly.
        ch = utils.content_hash_bytes(utils.raw_bytes(text))
        if ch in seen:
            skipped += 1
            continue
        if dry_run:
            print(f"  would add [{fact.get('category')}] {text}")
            added += 1
            continue
        dt = utils.utc_now()
        fpath, relpath, ch2 = utils.write_raw(raw_dir, text, dt)
        eid = db.insert_entry(
            text=text,
            source="seed_profile",
            origin_kind="self",          # the user asserted these about themselves
            source_type="text",
            raw_path=fpath,
            raw_relpath=relpath,
            content_hash=ch2,
        )
        db.add_structured(eid, models.Structured(
            entry_id=eid,
            category=fact.get("category"),
            importance=fact.get("importance", "MEDIUM"),
            # confidence 1.0 with inferred_by='user' keeps facets at MEMORY: these
            # were stated outright, not guessed by a heuristic.
            confidence=1.0,
            inferred_by="user",
            people=fact.get("people") or [],
            places=fact.get("places") or [],
            topics=fact.get("topics") or [],
            projects=fact.get("projects") or [],
            is_sensitive=1 if fact.get("sensitive") else 0,
            needs_review=0,
        ), inferred_by="user")
        seen.add(ch2)
        added += 1
        print(f"  #{eid} [{fact.get('category')}] {text[:64]}")

    if not dry_run:
        db.conn.commit()
    db.close()
    return added, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="show what would be added")
    ap.add_argument("--seed-file", default="data/seed/profile.json")
    ap.add_argument("--base", default=".")
    args = ap.parse_args()
    base = Path(args.base).resolve()
    seed_file = Path(args.seed_file)
    if not seed_file.is_absolute():
        seed_file = base / seed_file
    if not seed_file.exists():
        print(f"seed file not found: {seed_file}", file=sys.stderr)
        print("Copy config.seed.example.json to data/seed/profile.json and fill it in.", file=sys.stderr)
        return 1
    print(f"seeding from {seed_file}")
    added, skipped = seed(base, seed_file, dry_run=args.dry_run)
    print(f"{'would add' if args.dry_run else 'added'} {added}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())