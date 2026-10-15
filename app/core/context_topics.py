"""Small SQLite topic groups for associating continuity threads.

Topics are only created explicitly. Automatic matching is exact, conservative,
and never creates a topic; this module is not wired into chat retrieval.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any


MAX_DISPLAY_NAME = 160
RELATIONS = ("PRIMARY", "RELATED")
PROVENANCE = ("USER_EXPLICIT", "CONTINUITY_EXACT_MATCH")
_TOPIC_NAMESPACE = uuid.UUID("58c5b423-41bc-5a28-8676-fec4cbfeb5e1")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_GENERIC_WORDS = {"project", "topic", "conversation", "discussion", "work", "working"}
MAX_TOPIC_SIBLINGS = 2
MIN_TOPIC_CONFIDENCE = 0.8


class ContextTopicError(ValueError):
    """Invalid topic or association operation."""


class TopicAssociationError(ContextTopicError):
    """The requested thread-topic association is not allowed."""


def normalize_key(display_name: str) -> str:
    if not isinstance(display_name, str):
        raise ContextTopicError("display_name must be text")
    normalized = unicodedata.normalize("NFKC", display_name).casefold()
    return " ".join(_WORD_RE.findall(normalized))


def _display_name(value: str) -> tuple[str, str]:
    if not isinstance(value, str):
        raise ContextTopicError("display_name must be text")
    display = " ".join(unicodedata.normalize("NFC", value).split())
    key = normalize_key(display)
    if not display or not key:
        raise ContextTopicError("display_name must not be blank")
    if len(display) > MAX_DISPLAY_NAME:
        raise ContextTopicError(f"display_name exceeds {MAX_DISPLAY_NAME} characters")
    return display, key


def _stamp(now: datetime | None = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ContextTopicError("timestamps must include a timezone")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _topic_row(db, topic_id: str) -> dict[str, Any] | None:
    row = db.conn.execute("SELECT * FROM context_topics WHERE topic_id=?", (topic_id,)).fetchone()
    return dict(row) if row else None


def _strong_exact_match(db, thread_id: str, topic: dict[str, Any]) -> bool:
    from .conversation_continuity import ContinuityRepository

    thread = ContinuityRepository(db).get_thread(thread_id)
    if not thread:
        return False
    key = normalize_key(thread["topic"])
    meaningful = [word for word in key.split() if word not in _GENERIC_WORDS]
    return len(meaningful) >= 2 and key == topic["normalized_key"]


def get_topic(db, topic_id: str) -> dict[str, Any] | None:
    """Retrieve one topic by stable ID."""
    return _topic_row(db, topic_id)


def get_topic_by_key(db, display_name_or_key: str) -> dict[str, Any] | None:
    """Retrieve a topic by its normalized display name/key."""
    key = normalize_key(display_name_or_key)
    if not key:
        return None
    row = db.conn.execute("SELECT * FROM context_topics WHERE normalized_key=?", (key,)).fetchone()
    return dict(row) if row else None


def create_topic(db, display_name: str, *, now: datetime | None = None) -> dict[str, Any]:
    """Create an explicitly curated topic, idempotently by normalized key."""
    display, key = _display_name(display_name)
    stamp = _stamp(now)
    topic_id = str(uuid.uuid5(_TOPIC_NAMESPACE, key))
    db.conn.execute(
        "INSERT OR IGNORE INTO context_topics(topic_id,display_name,normalized_key,created_at,updated_at) "
        "VALUES(?,?,?,?,?)",
        (topic_id, display, key, stamp, stamp),
    )
    db.save()
    # A deterministic ID plus UNIQUE(normalized_key) makes duplicate creation
    # converge on the original row, including concurrent retries.
    topic = _topic_row(db, topic_id)
    if topic is None:
        topic = get_topic_by_key(db, key)
    if topic is None:
        raise ContextTopicError("could not create topic")
    return topic


def list_topics(db, *, limit: int = 200) -> list[dict[str, Any]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ContextTopicError("limit must be between 1 and 1000")
    rows = db.conn.execute(
        "SELECT * FROM context_topics ORDER BY updated_at DESC, normalized_key LIMIT ?", (limit,)
    )
    return [dict(row) for row in rows]


def associate_thread_topic(
    db,
    thread_id: str,
    topic_id: str,
    *,
    relation: str,
    confidence: float,
    provenance: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Idempotently link a thread and topic with explicit provenance."""
    if relation not in RELATIONS:
        raise ContextTopicError("relation must be PRIMARY or RELATED")
    if provenance not in PROVENANCE:
        raise ContextTopicError("unsupported association provenance")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ContextTopicError("confidence must be between 0 and 1")
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ContextTopicError("thread_id is required")
    topic = _topic_row(db, topic_id) if isinstance(topic_id, str) else None
    if topic is None:
        raise ContextTopicError("topic not found")
    if not db.conn.execute("SELECT 1 FROM continuity_threads WHERE thread_id=?", (thread_id,)).fetchone():
        raise ContextTopicError("continuity thread not found")
    if provenance == "CONTINUITY_EXACT_MATCH" and not _strong_exact_match(db, thread_id, topic):
        raise TopicAssociationError("topic association is not a strong exact match")
    if provenance == "USER_EXPLICIT" and float(confidence) != 1.0:
        raise ContextTopicError("explicit associations must use confidence 1.0")

    stamp = _stamp(now)
    current = db.conn.execute(
        "SELECT * FROM continuity_thread_topics WHERE thread_id=? AND topic_id=?", (thread_id, topic_id)
    ).fetchone()
    confidence = float(confidence)
    if current and (current["relation"], current["confidence"], current["provenance"]) == (
        relation, confidence, provenance
    ):
        return dict(current)
    try:
        if current:
            db.conn.execute(
                "UPDATE continuity_thread_topics SET relation=?,confidence=?,provenance=?,updated_at=? "
                "WHERE thread_id=? AND topic_id=?",
                (relation, confidence, provenance, stamp, thread_id, topic_id),
            )
        else:
            db.conn.execute(
                "INSERT INTO continuity_thread_topics(thread_id,topic_id,relation,confidence,provenance,created_at,updated_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (thread_id, topic_id, relation, confidence, provenance, stamp, stamp),
            )
        db.conn.execute("UPDATE context_topics SET updated_at=? WHERE topic_id=?", (stamp, topic_id))
        db.save()
    except sqlite3.IntegrityError:
        db.conn.rollback()
        # Keep sqlite-specific error text/IDs out of a future API surface.
        raise TopicAssociationError("association conflicts with existing topic links") from None
    row = db.conn.execute(
        "SELECT * FROM continuity_thread_topics WHERE thread_id=? AND topic_id=?", (thread_id, topic_id)
    ).fetchone()
    return dict(row)


def associate_explicitly(db, thread_id: str, topic_id: str, *, relation: str = "PRIMARY",
                         now: datetime | None = None) -> dict[str, Any]:
    """Associate after an explicit user-established topic relationship."""
    return associate_thread_topic(
        db, thread_id, topic_id, relation=relation, confidence=1.0,
        provenance="USER_EXPLICIT", now=now,
    )


def remove_thread_topic(db, thread_id: str, topic_id: str) -> bool:
    cur = db.conn.execute(
        "DELETE FROM continuity_thread_topics WHERE thread_id=? AND topic_id=?", (thread_id, topic_id)
    )
    if cur.rowcount:
        db.conn.execute(
            "UPDATE context_topics SET updated_at=? WHERE topic_id=?",
            (_stamp(), topic_id),
        )
        db.save()
        return True
    db.conn.rollback()
    return False


def list_topics_for_thread(db, thread_id: str) -> list[dict[str, Any]]:
    rows = db.conn.execute(
        "SELECT t.*, a.relation, a.confidence, a.provenance, a.created_at AS association_created_at, "
        "a.updated_at AS association_updated_at FROM context_topics t "
        "JOIN continuity_thread_topics a ON a.topic_id=t.topic_id WHERE a.thread_id=? "
        "ORDER BY CASE a.relation WHEN 'PRIMARY' THEN 0 ELSE 1 END, t.normalized_key",
        (thread_id,),
    )
    return [dict(row) for row in rows]


def list_threads_for_topic(db, topic_id: str) -> list[dict[str, Any]]:
    from .conversation_continuity import ContinuityRepository

    rows = db.conn.execute(
        "SELECT thread_id,relation,confidence,provenance,created_at,updated_at "
        "FROM continuity_thread_topics WHERE topic_id=? "
        "ORDER BY CASE relation WHEN 'PRIMARY' THEN 0 ELSE 1 END, updated_at DESC, thread_id",
        (topic_id,),
    ).fetchall()
    repo = ContinuityRepository(db)
    result = []
    for row in rows:
        item = dict(row)
        item["thread"] = repo.get_thread(item["thread_id"])
        result.append(item)
    return result


def find_strong_topic_match(db, thread_id: str) -> dict[str, Any] | None:
    """Find an existing exact topic match; never creates or fuzzy-links topics.

    A generic/single-word thread label is too weak for automatic association.
    Callers can use :func:`associate_explicitly` when the user establishes it.
    """
    from .conversation_continuity import ContinuityRepository

    thread = ContinuityRepository(db).get_thread(thread_id)
    if not thread:
        return None
    topic = get_topic_by_key(db, thread["topic"])
    return topic if topic and _strong_exact_match(db, thread_id, topic) else None


def associate_strong_match(db, thread_id: str, *, relation: str = "RELATED",
                           now: datetime | None = None) -> dict[str, Any] | None:
    """Associate only when the thread topic exactly matches an existing topic."""
    topic = find_strong_topic_match(db, thread_id)
    if topic is None:
        return None
    return associate_thread_topic(
        db, thread_id, topic["topic_id"], relation=relation, confidence=0.95,
        provenance="CONTINUITY_EXACT_MATCH", now=now,
    )


def relevant_sibling_threads(
    db,
    primary_thread: dict[str, Any] | None,
    message: str,
    *,
    primary_match_score: float,
    explicit_restore: bool = False,
    now: datetime | None = None,
    limit: int = MAX_TOPIC_SIBLINGS,
) -> list[dict[str, Any]]:
    """Return a few relevant direct siblings of a strongly matched thread.

    This performs exactly one lookup through the selected thread's PRIMARY
    topic. It neither follows other topics on sibling threads nor creates
    associations. The lexical score and threshold are shared with continuity
    retrieval.
    """
    if not primary_thread or limit <= 0:
        return []
    limit = min(limit, MAX_TOPIC_SIBLINGS)
    if not explicit_restore and (not isinstance(primary_match_score, (int, float))
                                 or primary_match_score < 0.22):
        return []

    primary_link = db.conn.execute(
        "SELECT topic_id FROM continuity_thread_topics "
        "WHERE thread_id=? AND relation='PRIMARY' AND confidence>=?",
        (primary_thread["thread_id"], MIN_TOPIC_CONFIDENCE),
    ).fetchone()
    if not primary_link:
        return []

    from . import conversation_continuity as continuity

    current = now or continuity._now()
    # A "where were we?" request has no useful lexical tokens. Use the selected
    # thread's compact snapshot as the query anchor in that explicit case.
    query = message if not explicit_restore else " ".join((
        primary_thread.get("current_problem_or_goal", ""),
        *primary_thread.get("decisions", []),
        *primary_thread.get("open_questions", []),
    ))
    candidates = []
    for item in list_threads_for_topic(db, primary_link["topic_id"]):
        sibling = item.get("thread")
        if not sibling or sibling["thread_id"] == primary_thread["thread_id"]:
            continue
        if item["confidence"] < MIN_TOPIC_CONFIDENCE or sibling["status"] != "ACTIVE":
            continue
        last = continuity._parse_time(sibling.get("last_activity_at"))
        expiry = continuity._parse_time(sibling.get("expires_at"))
        if expiry and expiry <= current:
            continue
        if not last or current - last > continuity.EXPIRE_AFTER:
            continue
        if not explicit_restore and current - last > continuity.DORMANT_AFTER:
            continue

        snapshot_text = " ".join((
            sibling.get("current_problem_or_goal", ""),
            *sibling.get("decisions", []), *sibling.get("open_questions", []),
        ))
        score, hits = continuity._similarity(query, snapshot_text)
        # Reuse the same acceptance gate as find_relevant; a shared topic alone
        # is never sufficient to inject a sibling.
        if score < 0.22 or not (hits >= 2 or score >= 0.55):
            continue
        candidates.append((score, last.timestamp(), sibling))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    return [item[2] for item in candidates[:limit]]
