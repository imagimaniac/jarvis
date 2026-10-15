"""Minimal models (dataclasses) for Jarvis."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json

@dataclass
class Entry:
    id: int | None = None
    uuid: str = ""
    created_at_utc: str = ""
    created_at_local: str = ""
    tz: str | None = None
    source_type: str = "text"
    source: str = "cli"
    origin_kind: str = "self"  # self|external_record
    origin_id: str | None = None
    import_batch_id: str | None = None
    external_ref: str | None = None
    raw_path: str = ""
    raw_relpath: str = ""
    content_hash: str = ""
    text_raw: str = ""
    size_bytes: int = 0

@dataclass
class Structured:
    entry_id: int
    date: str | None = None
    time_local: str | None = None
    type: str | None = None
    mood: str | None = None
    people: List[str] | None = None
    places: List[str] | None = None
    projects: List[str] | None = None
    topics: List[str] | None = None
    events: List[str] | None = None
    questions: List[str] | None = None
    decisions: List[str] | None = None
    learnings: List[str] | None = None
    tags: List[str] | None = None
    category: str | None = None
    importance: str | None = "MEDIUM"
    confidence: float | None = 0.5
    inferred_by: str | None = "heuristic"
    needs_review: int | None = 0
    reason: str | None = None
    is_sensitive: int | None = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("people","places","projects","topics","events","questions","decisions","learnings","tags"):
            if d.get(k) is None:
                d[k] = "[]"
            else:
                d[k] = json.dumps(d[k], ensure_ascii=False)
        return d
