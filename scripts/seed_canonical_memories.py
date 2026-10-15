"""Explicitly apply the curated, user-approved canonical personal-memory seed.

Safe by default: no database is opened or changed without ``--apply``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core import config, memory, memory_workflow, storage  # noqa: E402


SEED_ATTRIBUTION = "Curated user-approved personal seed (initial Jarvis profile)"
CONFIRM_REASON = "Explicitly confirmed curated personal seed"
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,79}$")


def _claim_key(value: str) -> str:
    return " ".join(unicodedata.normalize("NFC", value).split()).casefold()


def load_records(seed_file: Path) -> list[dict[str, Any]]:
    """Load and strictly validate the seed file before any database writes."""
    try:
        data = json.loads(seed_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("could not read canonical memory seed file") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("unsupported canonical memory seed schema")
    records = data.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("seed records must be a non-empty list")

    validated: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for item in records:
        if not isinstance(item, dict):
            raise ValueError("each seed record must be an object")
        required = {"key", "type", "claim", "importance", "confidence", "sensitive", "rationale"}
        if set(item) - (required | {"valid_from_utc"}) or not required <= set(item):
            raise ValueError("seed record fields do not match the supported schema")
        key = item["key"]
        if not isinstance(key, str) or not _KEY_RE.fullmatch(key) or key in seen_keys:
            raise ValueError("seed keys must be unique lowercase identifiers")
        if item["type"] not in memory.MEMORY_TYPES:
            raise ValueError("seed record has an unsupported memory type")
        if not isinstance(item["claim"], str) or not item["claim"].strip():
            raise ValueError("seed claim must be non-empty text")
        if isinstance(item["importance"], bool) or not isinstance(item["importance"], int) or not 1 <= item["importance"] <= 5:
            raise ValueError("seed importance must be an integer from 1 to 5")
        if isinstance(item["confidence"], bool) or not isinstance(item["confidence"], (int, float)) or not 0 <= item["confidence"] <= 1:
            raise ValueError("seed confidence must be between 0 and 1")
        if item["sensitive"] is not False:
            raise ValueError("initial canonical seed records must be non-sensitive")
        if not isinstance(item["rationale"], str) or not item["rationale"].strip():
            raise ValueError("seed rationale must be non-empty text")
        if "valid_from_utc" in item:
            try:
                valid_from = datetime.fromisoformat(str(item["valid_from_utc"]).replace("Z", "+00:00"))
                if valid_from.tzinfo is None or valid_from.utcoffset() is None:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("seed valid_from_utc must be a timezone-aware ISO-8601 timestamp") from exc
        seen_keys.add(key)
        validated.append(dict(item))
    return validated


def _all_memories(db) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    offset = 0
    while True:
        batch = memory.list_memories(db, limit=100, offset=offset)
        result.extend(batch)
        if len(batch) < 100:
            return result
        offset += len(batch)


def seed_records(db, records: list[dict[str, Any]], seed_file: Path) -> dict[str, int]:
    """Create and explicitly confirm missing seed records via memory services."""
    existing = _all_memories(db)
    by_seed_key = {
        str(item.get("source_attribution", "")).split("seed-key:", 1)[1].split("]", 1)[0]
        for item in existing if "seed-key:" in str(item.get("source_attribution", ""))
    }
    claims = {(_claim_key(item["claim"]), item["memory_type"]) for item in existing}
    added = skipped = 0
    source_path = seed_file.resolve()
    try:
        relative_source_path = source_path.relative_to(config.BASE).as_posix()
    except ValueError:
        relative_source_path = source_path.name

    for record in records:
        key = record["key"]
        claim_identity = (_claim_key(record["claim"]), record["type"])
        if key in by_seed_key or claim_identity in claims:
            skipped += 1
            continue

        origin_id = f"canonical-seed:{key}"
        source = db.conn.execute(
            "SELECT id FROM entries WHERE source_type='canonical_memory_seed' AND origin_id=? ORDER BY id LIMIT 1",
            (origin_id,),
        ).fetchone()
        source_id = source["id"] if source else db.insert_entry(
            f"Curated personal seed: {record['claim']}",
            source="canonical_memory_seed",
            origin_kind="self",
            source_type="canonical_memory_seed",
            origin_id=origin_id,
            raw_path=Path(relative_source_path),
            raw_relpath=relative_source_path,
        )

        attribution = f"{SEED_ATTRIBUTION} [seed-key:{key}]"
        candidate = memory.create_candidate(
            db,
            memory_type=record["type"],
            claim=record["claim"],
            source_entry_ids=[source_id],
            source_attribution=attribution,
            reason=f"Curated seed rationale: {record['rationale']}",
            candidate_reason=f"seed-key:{key}; {record['rationale']}",
            comparison_relation="NEW",
            confidence=float(record["confidence"]),
            importance=record["importance"],
            is_sensitive=False,
            valid_from_utc=record.get("valid_from_utc"),
        )
        memory_workflow.confirm_candidate(db, candidate["id"], reason=CONFIRM_REASON)
        by_seed_key.add(key)
        claims.add(claim_identity)
        added += 1
    return {"added": added, "skipped": skipped}


def apply_seed(seed_file: Path, *, db_path: Path | None = None) -> dict[str, int]:
    """Apply the manifest to the selected local Jarvis database."""
    records = load_records(seed_file)
    target = db_path or config.db_path()
    db = storage.JarvisDB(target)
    try:
        return seed_records(db, records, seed_file)
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-file", type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "seed" / "canonical_personal_memories.json",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="explicitly write missing records to the configured local Jarvis database",
    )
    args = parser.parse_args()
    if not args.apply:
        print("No database opened or changed. Review the seed file, then rerun with --apply to apply it.")
        return 0
    if not args.seed_file.is_file():
        print("canonical memory seed file not found", file=sys.stderr)
        return 1
    try:
        counts = apply_seed(args.seed_file)
    except Exception as exc:
        # Keep output concise and avoid printing input records or configuration.
        print(f"canonical memory seed failed: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Canonical memory seed complete: added {counts['added']}, skipped {counts['skipped']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
