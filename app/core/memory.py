"""Internal canonical personal-memory service.

Source entries remain immutable evidence. This module stores reviewed,
provenance-linked memory claims separately and never calls an LLM.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable


MEMORY_TYPES = (
    "FACT", "PREFERENCE", "GOAL", "DECISION", "PROJECT", "HABIT",
    "LEARNING", "HISTORY", "TEMPORARY",
)
LIFECYCLE_STATES = ("ACTIVE", "SUPERSEDED", "FORGOTTEN")
REVIEW_STATES = ("CANDIDATE", "CONFIRMED", "REJECTED")
CANDIDATE_RELATIONSHIPS = ("NEW", "POSSIBLE_UPDATE", "POSSIBLE_CONFLICT", "DUPLICATE", "TEMPORARY")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)
_UNSET = object()


class MemoryErrorBase(ValueError):
    """Base error for invalid or disallowed memory operations."""


class MemoryValidationError(MemoryErrorBase):
    """An input or provenance reference violates the memory data model."""


class MemoryTransitionError(MemoryErrorBase):
    """An operation is not allowed from the memory's current state."""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _timestamp(value: str | datetime | None, field: str, *, default: str | None = None) -> str:
    if value is None:
        if default is not None:
            return default
        raise MemoryValidationError(f"{field} is required")
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise MemoryValidationError(f"{field} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MemoryValidationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _claim(value: str) -> str:
    if not isinstance(value, str):
        raise MemoryValidationError("claim must be text")
    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise MemoryValidationError("claim must not be blank")
    return normalized


def _label(value: str, allowed: Iterable[str], field: str) -> str:
    if not isinstance(value, str):
        raise MemoryValidationError(f"{field} must be text")
    result = value.strip().upper()
    if result not in allowed:
        raise MemoryValidationError(f"unsupported {field}: {value!r}")
    return result


def _reason(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError("a non-empty reason is required")
    return " ".join(value.split())


def _confidence(value: float) -> float:
    if isinstance(value, bool):
        raise MemoryValidationError("confidence must be a number between 0 and 1")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise MemoryValidationError("confidence must be a number between 0 and 1") from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise MemoryValidationError("confidence must be a number between 0 and 1")
    return result


def _importance(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 5:
        raise MemoryValidationError("importance must be an integer from 1 to 5")
    return value


def _sensitive(value: bool) -> int:
    if not isinstance(value, bool):
        raise MemoryValidationError("is_sensitive must be a boolean")
    return int(value)


def _attribution(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError("source_attribution must not be blank")
    return " ".join(value.split())


def _source_ids(db, values: Iterable[int]) -> list[int]:
    if isinstance(values, (str, bytes)):
        raise MemoryValidationError("source_entry_ids must be a non-empty sequence of IDs")
    try:
        ids = list(values)
    except TypeError:
        raise MemoryValidationError("source_entry_ids must be a non-empty sequence of IDs") from None
    if not ids:
        raise MemoryValidationError("at least one source entry is required")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in ids):
        raise MemoryValidationError("source entry IDs must be positive integers")
    ids = list(dict.fromkeys(ids))
    placeholders = ",".join("?" for _ in ids)
    found = {row["id"] for row in db.conn.execute(f"SELECT id FROM entries WHERE id IN ({placeholders})", ids)}
    missing = [item for item in ids if item not in found]
    if missing:
        raise MemoryValidationError("all provenance source entries must exist")
    return ids


def _timestamp_pair(valid_from: str | datetime | None, valid_until: str | datetime | None,
                    *, default_from: str | None = None) -> tuple[str, str | None]:
    start = _timestamp(valid_from, "valid_from_utc", default=default_from)
    end = _timestamp(valid_until, "valid_until_utc") if valid_until is not None else None
    if end is not None and end < start:
        raise MemoryValidationError("valid_until_utc must not precede valid_from_utc")
    return start, end


def _base_row(db, memory_id: str) -> dict[str, Any]:
    row = db.conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
    if row is None:
        raise MemoryValidationError("memory not found")
    return dict(row)


def _source_rows(db, memory_id: str) -> list[dict[str, Any]]:
    return [dict(row) for row in db.conn.execute(
        "SELECT ms.entry_id, ms.relationship, ms.linked_at_utc, e.source, e.origin_kind, e.created_at_utc "
        "FROM memory_sources ms JOIN entries e ON e.id=ms.entry_id "
        "WHERE ms.memory_id=? ORDER BY ms.entry_id", (memory_id,)
    )]


def _snapshot(db, memory_id: str) -> dict[str, Any]:
    state = _base_row(db, memory_id)
    state["source_entry_ids"] = [item["entry_id"] for item in _source_rows(db, memory_id)]
    return state


def _record(db, memory_id: str, action: str, reason: str,
            previous: dict[str, Any] | None, new: dict[str, Any], changed_at: str,
            reviewer: str | None = None) -> None:
    db.conn.execute(
        "INSERT INTO memory_history (memory_id, action, changed_at_utc, reason, previous_state, new_state, reviewer) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (memory_id, action, changed_at, reason,
         json.dumps(previous, ensure_ascii=False, sort_keys=True) if previous is not None else None,
         json.dumps(new, ensure_ascii=False, sort_keys=True),
         reviewer),
    )


def _insert_sources(db, memory_id: str, source_ids: list[int], linked_at: str) -> None:
    db.conn.executemany(
        "INSERT INTO memory_sources (memory_id, entry_id, relationship, linked_at_utc) VALUES (?, ?, 'supports', ?)",
        [(memory_id, entry_id, linked_at) for entry_id in source_ids],
    )


def get_memory(db, memory_id: str) -> dict[str, Any]:
    state = _base_row(db, memory_id)
    state["sources"] = _source_rows(db, memory_id)
    return state


def get_history(db, memory_id: str) -> list[dict[str, Any]]:
    _base_row(db, memory_id)
    rows = db.conn.execute(
        "SELECT id, memory_id, action, changed_at_utc, reason, previous_state, new_state, reviewer "
        "FROM memory_history WHERE memory_id=? ORDER BY id", (memory_id,)
    )
    out = []
    for row in rows:
        item = dict(row)
        item["previous_state"] = json.loads(item["previous_state"]) if item["previous_state"] else None
        item["new_state"] = json.loads(item["new_state"])
        out.append(item)
    return out


def create_candidate(db, *, memory_type: str, claim: str, source_entry_ids: Iterable[int],
                     source_attribution: str, reason: str, confidence: float = 0.5,
                     importance: int = 3, is_sensitive: bool = False,
                     candidate_reason: str | None = None,
                     comparison_relation: str = "NEW",
                     related_memory_id: str | None = None,
                     valid_from_utc: str | datetime | None = None,
                     valid_until_utc: str | datetime | None = None) -> dict[str, Any]:
    """Create an unconfirmed, provenance-linked candidate; no inference occurs here."""
    kind = _label(memory_type, MEMORY_TYPES, "memory_type")
    text = _claim(claim)
    attribution = _attribution(source_attribution)
    rationale = _reason(reason)
    certainty = _confidence(confidence)
    priority = _importance(importance)
    sensitive = _sensitive(is_sensitive)
    candidate_reason = _reason(candidate_reason if candidate_reason is not None else reason)
    relation = _label(comparison_relation, CANDIDATE_RELATIONSHIPS, "comparison_relation")
    if relation in ("POSSIBLE_UPDATE", "POSSIBLE_CONFLICT", "DUPLICATE") and not related_memory_id:
        raise MemoryValidationError(f"{relation} requires a related memory")
    if relation in ("NEW", "TEMPORARY") and related_memory_id is not None:
        raise MemoryValidationError(f"{relation} must not reference an existing memory")
    if related_memory_id is not None:
        _base_row(db, related_memory_id)
    created = _now()
    valid_from, valid_until = _timestamp_pair(valid_from_utc, valid_until_utc, default_from=created)
    sources = _source_ids(db, source_entry_ids)
    memory_id = str(uuid.uuid4())
    with db.conn:
        db.conn.execute(
            "INSERT INTO memories (id, memory_type, claim, lifecycle_status, review_state, confidence, importance, "
            "is_sensitive, source_attribution, candidate_reason, comparison_relation, related_memory_id, "
            "created_at_utc, updated_at_utc, valid_from_utc, valid_until_utc) "
            "VALUES (?, ?, ?, 'ACTIVE', 'CANDIDATE', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (memory_id, kind, text, certainty, priority, sensitive, attribution, candidate_reason,
             relation, related_memory_id, created, created, valid_from, valid_until),
        )
        _insert_sources(db, memory_id, sources, created)
        db.conn.execute("INSERT INTO canonical_memories_fts (memory_id, claim) VALUES (?, ?)", (memory_id, text))
        _record(db, memory_id, "CANDIDATE_CREATED", rationale, None, _snapshot(db, memory_id), created)
    return get_memory(db, memory_id)


def _change_review(db, memory_id: str, target: str, action: str, reason: str,
                   reviewer: str | None = None) -> dict[str, Any]:
    rationale = _reason(reason)
    changed = _now()
    with db.conn:
        previous = _snapshot(db, memory_id)
        if previous["lifecycle_status"] != "ACTIVE" or previous["review_state"] != "CANDIDATE":
            raise MemoryTransitionError(f"only an active candidate can be {target.lower()}")
        db.conn.execute("UPDATE memories SET review_state=?, updated_at_utc=? WHERE id=?",
                         (target, changed, memory_id))
        _record(db, memory_id, action, rationale, previous, _snapshot(db, memory_id), changed,
                reviewer=reviewer)
    return get_memory(db, memory_id)


def confirm(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict[str, Any]:
    """Confirm an active candidate after an explicit human review."""
    return _change_review(db, memory_id, "CONFIRMED", "CONFIRMED", reason, reviewer=reviewer)


def reject(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict[str, Any]:
    """Reject an active candidate without deleting its audit record."""
    return _change_review(db, memory_id, "REJECTED", "REJECTED", reason, reviewer=reviewer)


def defer(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict[str, Any]:
    """Record a DEFER decision without changing an active candidate's state."""
    rationale = _reason(reason)
    changed = _now()
    with db.conn:
        previous = _snapshot(db, memory_id)
        if previous["lifecycle_status"] != "ACTIVE" or previous["review_state"] != "CANDIDATE":
            raise MemoryTransitionError("only an active candidate can be deferred")
        _record(db, memory_id, "DEFERRED", rationale, previous, _snapshot(db, memory_id), changed,
                reviewer=reviewer)
    return get_memory(db, memory_id)


def update_memory(db, memory_id: str, *, reason: str, reviewer: str | None = None,
                  claim: Any = _UNSET,
                  memory_type: Any = _UNSET, confidence: Any = _UNSET,
                  importance: Any = _UNSET, is_sensitive: Any = _UNSET,
                  source_attribution: Any = _UNSET,
                  source_entry_ids: Any = _UNSET,
                  valid_from_utc: Any = _UNSET,
                  valid_until_utc: Any = _UNSET) -> dict[str, Any]:
    """Update an active memory while recording full before/after snapshots."""
    rationale = _reason(reason)
    with db.conn:
        previous = _snapshot(db, memory_id)
        if previous["lifecycle_status"] != "ACTIVE" or previous["review_state"] == "REJECTED":
            raise MemoryTransitionError("only active, non-rejected memories can be updated")

        values = {key: previous[key] for key in (
            "memory_type", "claim", "confidence", "importance", "is_sensitive",
            "source_attribution", "valid_from_utc", "valid_until_utc",
        )}
        if memory_type is not _UNSET:
            values["memory_type"] = _label(memory_type, MEMORY_TYPES, "memory_type")
        if claim is not _UNSET:
            values["claim"] = _claim(claim)
        if confidence is not _UNSET:
            values["confidence"] = _confidence(confidence)
        if importance is not _UNSET:
            values["importance"] = _importance(importance)
        if is_sensitive is not _UNSET:
            values["is_sensitive"] = _sensitive(is_sensitive)
        if source_attribution is not _UNSET:
            values["source_attribution"] = _attribution(source_attribution)
        if valid_from_utc is not _UNSET:
            values["valid_from_utc"] = _timestamp(valid_from_utc, "valid_from_utc")
        if valid_until_utc is not _UNSET:
            values["valid_until_utc"] = (
                _timestamp(valid_until_utc, "valid_until_utc") if valid_until_utc is not None else None
            )
        if values["valid_until_utc"] is not None and values["valid_until_utc"] < values["valid_from_utc"]:
            raise MemoryValidationError("valid_until_utc must not precede valid_from_utc")

        sources = previous["source_entry_ids"]
        if source_entry_ids is not _UNSET:
            sources = _source_ids(db, source_entry_ids)
        changed_fields = any(values[key] != previous[key] for key in values) or sources != previous["source_entry_ids"]
        if not changed_fields:
            raise MemoryValidationError("update must change at least one field")

        changed = max(_now(), previous["updated_at_utc"])
        db.conn.execute(
            "UPDATE memories SET memory_type=?, claim=?, confidence=?, importance=?, is_sensitive=?, "
            "source_attribution=?, valid_from_utc=?, valid_until_utc=?, updated_at_utc=? WHERE id=?",
            (values["memory_type"], values["claim"], values["confidence"], values["importance"],
             values["is_sensitive"], values["source_attribution"], values["valid_from_utc"],
             values["valid_until_utc"], changed, memory_id),
        )
        if sources != previous["source_entry_ids"]:
            db.conn.execute("DELETE FROM memory_sources WHERE memory_id=?", (memory_id,))
            _insert_sources(db, memory_id, sources, changed)
        if values["claim"] != previous["claim"]:
            db.conn.execute("DELETE FROM canonical_memories_fts WHERE memory_id=?", (memory_id,))
            db.conn.execute("INSERT INTO canonical_memories_fts (memory_id, claim) VALUES (?, ?)",
                             (memory_id, values["claim"]))
        _record(db, memory_id, "UPDATED", rationale, previous, _snapshot(db, memory_id), changed,
                reviewer=reviewer)
    return get_memory(db, memory_id)


def supersede(db, memory_id: str, replacement_memory_id: str, *, reason: str,
              reviewer: str | None = None) -> dict[str, Any]:
    """Mark a confirmed active memory superseded by another confirmed memory."""
    rationale = _reason(reason)
    if memory_id == replacement_memory_id:
        raise MemoryValidationError("a memory cannot supersede itself")
    changed = _now()
    with db.conn:
        previous = _snapshot(db, memory_id)
        replacement = _base_row(db, replacement_memory_id)
        if previous["lifecycle_status"] != "ACTIVE" or previous["review_state"] != "CONFIRMED":
            raise MemoryTransitionError("only an active confirmed memory can be superseded")
        if replacement["lifecycle_status"] != "ACTIVE" or replacement["review_state"] != "CONFIRMED":
            raise MemoryTransitionError("replacement must be active and confirmed")
        db.conn.execute(
            "UPDATE memories SET lifecycle_status='SUPERSEDED', superseded_by_memory_id=?, updated_at_utc=? WHERE id=?",
            (replacement_memory_id, changed, memory_id),
        )
        _record(db, memory_id, "SUPERSEDED", rationale, previous, _snapshot(db, memory_id), changed,
                reviewer=reviewer)
    return get_memory(db, memory_id)


def forget(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict[str, Any]:
    """Hide an active memory from normal retrieval while preserving provenance/history."""
    rationale = _reason(reason)
    changed = _now()
    with db.conn:
        previous = _snapshot(db, memory_id)
        if previous["lifecycle_status"] != "ACTIVE":
            raise MemoryTransitionError("only an active memory can be forgotten")
        db.conn.execute("UPDATE memories SET lifecycle_status='FORGOTTEN', updated_at_utc=? WHERE id=?",
                         (changed, memory_id))
        _record(db, memory_id, "FORGOTTEN", rationale, previous, _snapshot(db, memory_id), changed,
                reviewer=reviewer)
    return get_memory(db, memory_id)


def _limit(value: int, maximum: int = 100) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise MemoryValidationError(f"limit must be between 1 and {maximum}")
    return value


def list_memories(db, *, memory_type: str | None = None, lifecycle_status: str | None = None,
                  review_state: str | None = None, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    """List canonical records with validated, optional type/state filters."""
    limit = _limit(limit)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise MemoryValidationError("offset must be a non-negative integer")
    clauses, params = [], []
    for value, allowed, column, field in (
        (memory_type, MEMORY_TYPES, "memory_type", "memory_type"),
        (lifecycle_status, LIFECYCLE_STATES, "lifecycle_status", "lifecycle_status"),
        (review_state, REVIEW_STATES, "review_state", "review_state"),
    ):
        if value is not None:
            clauses.append(f"{column}=?")
            params.append(_label(value, allowed, field))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    rows = db.conn.execute(
        "SELECT id FROM memories" + where + " ORDER BY created_at_utc DESC, id LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    return [get_memory(db, row["id"]) for row in rows]


def retrieve(db, query: str, *, limit: int = 5, now: str | datetime | None = None) -> list[dict[str, Any]]:
    """Lexically retrieve only active, confirmed, currently valid memories."""
    limit = _limit(limit)
    if not isinstance(query, str):
        raise MemoryValidationError("query must be text")
    words = list(dict.fromkeys(_WORD_RE.findall(query)))
    if not words:
        return []
    match = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
    at = _timestamp(now, "now") if now is not None else _now()
    rows = db.conn.execute(
        "SELECT m.id FROM canonical_memories_fts f JOIN memories m ON m.id=f.memory_id "
        "WHERE canonical_memories_fts MATCH ? AND m.lifecycle_status='ACTIVE' AND m.review_state='CONFIRMED' "
        "AND m.valid_from_utc<=? AND (m.valid_until_utc IS NULL OR m.valid_until_utc>=?) "
        "ORDER BY bm25(canonical_memories_fts), m.importance DESC, m.updated_at_utc DESC LIMIT ?",
        (match, at, at, limit),
    )
    return [get_memory(db, row["id"]) for row in rows]


def overview(db, *, limit: int = 30, now: str | datetime | None = None) -> list[dict[str, Any]]:
    """List active, confirmed, currently valid memories by importance then recency.

    This is the deterministic broad-view listing used when a message explicitly
    asks about the user rather than about one narrow topic. It performs no
    lexical match, so highly important memories are surfaced even when the
    question's keywords do not overlap their wording. Still read-only and fully
    validity-gated."""
    limit = _limit(limit)
    at = _timestamp(now, "now") if now is not None else _now()
    rows = db.conn.execute(
        "SELECT id FROM memories WHERE lifecycle_status='ACTIVE' AND review_state='CONFIRMED' "
        "AND valid_from_utc<=? AND (valid_until_utc IS NULL OR valid_until_utc>=?) "
        "ORDER BY importance DESC, updated_at_utc DESC, id LIMIT ?",
        (at, at, limit),
    )
    return [get_memory(db, row["id"]) for row in rows]
