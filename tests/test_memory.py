import shutil
import tempfile
import uuid
from pathlib import Path

import pytest

from app.core import memory, storage


def _db():
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    db = storage.JarvisDB(root / "data/jarvis.db")
    return root, db


def _source(db, text="Original source capture"):
    return db.insert_entry(text, source="cli", origin_kind="self",
                           raw_path="memory/raw/source.txt", raw_relpath="source.txt")


def _candidate(db, source_id, claim="User prefers concise answers", **kwargs):
    values = {
        "memory_type": "PREFERENCE",
        "claim": claim,
        "source_entry_ids": [source_id],
        "source_attribution": "user-confirmed capture",
        "reason": "explicitly captured for review",
    }
    values.update(kwargs)
    return memory.create_candidate(db, **values)


def test_create_candidate_normalizes_claim_and_keeps_source_link_and_history():
    root, db = _db()
    try:
        source_id = _source(db, "Do not copy this full text into the memory row.")
        created = _candidate(db, source_id, "  User  prefers\n concise answers  ")
        assert str(uuid.UUID(created["id"])) == created["id"]
        assert created["memory_type"] == "PREFERENCE"
        assert created["claim"] == "User prefers concise answers"
        assert created["lifecycle_status"] == "ACTIVE"
        assert created["review_state"] == "CANDIDATE"
        assert created["created_at_utc"] == created["updated_at_utc"]
        assert created["valid_from_utc"] == created["created_at_utc"]
        assert created["candidate_reason"] == "explicitly captured for review"
        assert created["comparison_relation"] == "NEW"
        assert created["sources"][0]["entry_id"] == source_id
        assert created["sources"][0]["source"] == "cli"
        assert "text_raw" not in created["sources"][0]
        assert len(db.conn.execute("SELECT * FROM entries").fetchall()) == 1
        assert len(db.conn.execute("SELECT * FROM memory_history WHERE memory_id=?", (created["id"],)).fetchall()) == 1
        event = memory.get_history(db, created["id"])[0]
        assert event["action"] == "CANDIDATE_CREATED"
        assert event["previous_state"] is None
        assert event["new_state"]["source_entry_ids"] == [source_id]
    finally:
        db.close()
        shutil.rmtree(root)


@pytest.mark.parametrize("changes", [
    {"memory_type": "UNKNOWN"},
    {"claim": "  \n "},
    {"confidence": -0.01},
    {"confidence": 1.01},
    {"confidence": float("nan")},
    {"importance": 0},
    {"importance": 6},
    {"importance": True},
    {"is_sensitive": 1},
    {"source_attribution": " "},
    {"valid_from_utc": "not-a-date"},
    {"valid_from_utc": "2026-04-01T00:00:00"},
    {"valid_from_utc": "2026-04-02T00:00:00Z", "valid_until_utc": "2026-04-01T00:00:00Z"},
    {"source_entry_ids": []},
    {"source_entry_ids": [999999]},
    {"source_entry_ids": [True]},
    {"reason": " "},
])
def test_candidate_validation_rejects_invalid_values(changes):
    root, db = _db()
    try:
        source_id = _source(db)
        values = {
            "memory_type": "FACT", "claim": "A claim", "source_entry_ids": [source_id],
            "source_attribution": "user statement", "reason": "review candidate",
        }
        values.update(changes)
        with pytest.raises(memory.MemoryValidationError):
            memory.create_candidate(db, **values)
        assert db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()["COUNT(*)"] == 0
    finally:
        db.close()
        shutil.rmtree(root)


def test_invalid_memory_state_is_rejected_by_sqlite_constraints():
    root, db = _db()
    try:
        with pytest.raises(Exception):
            db.conn.execute(
                "INSERT INTO memories (id,memory_type,claim,lifecycle_status,review_state,confidence,importance,"
                "is_sensitive,source_attribution,created_at_utc,updated_at_utc,valid_from_utc) "
                "VALUES ('bad','NOPE','claim','ACTIVE','CANDIDATE',0.5,3,0,'user','2026-01-01T00:00:00Z',"
                "'2026-01-01T00:00:00Z','2026-01-01T00:00:00Z')"
            )
    finally:
        db.close()
        shutil.rmtree(root)


def test_confirm_reject_and_review_transitions_are_audited():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id)
        confirmed = memory.confirm(db, candidate["id"], reason="user reviewed and approved")
        assert confirmed["review_state"] == "CONFIRMED"
        with pytest.raises(memory.MemoryTransitionError):
            memory.confirm(db, candidate["id"], reason="cannot confirm twice")

        rejected = _candidate(db, source_id, "A different proposed claim")
        assert memory.reject(db, rejected["id"], reason="not a durable preference")["review_state"] == "REJECTED"
        with pytest.raises(memory.MemoryTransitionError):
            memory.confirm(db, rejected["id"], reason="rejected candidates are terminal")
        events = memory.get_history(db, candidate["id"])
        assert [e["action"] for e in events] == ["CANDIDATE_CREATED", "CONFIRMED"]
        assert events[1]["previous_state"]["review_state"] == "CANDIDATE"
        assert events[1]["new_state"]["review_state"] == "CONFIRMED"
    finally:
        db.close()
        shutil.rmtree(root)


def test_update_records_before_after_values_updates_fts_and_provenance():
    root, db = _db()
    try:
        first_source, second_source = _source(db, "Original assertion"), _source(db, "Clarifying assertion")
        candidate = _candidate(db, first_source, "User likes concise replies")
        memory.confirm(db, candidate["id"], reason="verified by user")
        updated = memory.update_memory(
            db, candidate["id"], reason="user corrected the preference",
            claim="User prefers short, direct replies", confidence=0.95, importance=4,
            source_entry_ids=[first_source, second_source],
        )
        assert updated["claim"] == "User prefers short, direct replies"
        assert updated["confidence"] == 0.95 and updated["importance"] == 4
        assert {s["entry_id"] for s in updated["sources"]} == {first_source, second_source}
        assert memory.retrieve(db, "concise") == []
        assert memory.retrieve(db, "short direct replies")[0]["id"] == candidate["id"]
        history = memory.get_history(db, candidate["id"])
        change = history[-1]
        assert change["action"] == "UPDATED"
        assert change["reason"] == "user corrected the preference"
        assert change["previous_state"]["claim"] == "User likes concise replies"
        assert change["new_state"]["claim"] == "User prefers short, direct replies"
        assert change["previous_state"]["source_entry_ids"] == [first_source]
        assert change["new_state"]["source_entry_ids"] == [first_source, second_source]
    finally:
        db.close()
        shutil.rmtree(root)


def test_update_rejects_noop_invalid_fields_and_terminal_records():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id)
        with pytest.raises(memory.MemoryValidationError):
            memory.update_memory(db, candidate["id"], reason="no-op")
        with pytest.raises(memory.MemoryValidationError):
            memory.update_memory(db, candidate["id"], reason="bad confidence", confidence=2)
        memory.forget(db, candidate["id"], reason="remove candidate")
        with pytest.raises(memory.MemoryTransitionError):
            memory.update_memory(db, candidate["id"], reason="cannot edit forgotten memory", claim="new")
    finally:
        db.close()
        shutil.rmtree(root)


def test_supersede_requires_active_confirmed_memories_and_preserves_both_versions():
    root, db = _db()
    try:
        source_id = _source(db)
        old = _candidate(db, source_id, "User lives in one city", memory_type="HISTORY")
        new = _candidate(db, source_id, "User currently lives in another city", memory_type="HISTORY")
        with pytest.raises(memory.MemoryTransitionError):
            memory.supersede(db, old["id"], new["id"], reason="replacement is not approved")
        memory.confirm(db, old["id"], reason="old claim approved")
        memory.confirm(db, new["id"], reason="new claim approved")
        result = memory.supersede(db, old["id"], new["id"], reason="user corrected current location")
        assert result["lifecycle_status"] == "SUPERSEDED"
        assert result["superseded_by_memory_id"] == new["id"]
        assert memory.get_memory(db, new["id"])["lifecycle_status"] == "ACTIVE"
        assert memory.retrieve(db, "lives city") == [memory.get_memory(db, new["id"])]
        event = memory.get_history(db, old["id"])[-1]
        assert event["action"] == "SUPERSEDED"
        assert event["previous_state"]["lifecycle_status"] == "ACTIVE"
        assert event["new_state"]["lifecycle_status"] == "SUPERSEDED"
        with pytest.raises(memory.MemoryTransitionError):
            memory.forget(db, old["id"], reason="superseded is terminal")
    finally:
        db.close()
        shutil.rmtree(root)


def test_forget_hides_memory_but_keeps_history_and_source_entry():
    root, db = _db()
    try:
        source_id = _source(db, "Original source is preserved after forget")
        item = _candidate(db, source_id, "User enjoys a particular hobby", memory_type="HABIT")
        memory.confirm(db, item["id"], reason="confirmed")
        assert memory.retrieve(db, "particular hobby")
        forgotten = memory.forget(db, item["id"], reason="user asked Jarvis to forget this")
        assert forgotten["lifecycle_status"] == "FORGOTTEN"
        assert memory.retrieve(db, "particular hobby") == []
        assert memory.get_memory(db, item["id"])["sources"][0]["entry_id"] == source_id
        assert db.get(source_id)["entry"]["text_raw"] == "Original source is preserved after forget"
        assert memory.get_history(db, item["id"])[-1]["action"] == "FORGOTTEN"
        with pytest.raises(memory.MemoryTransitionError):
            memory.forget(db, item["id"], reason="cannot forget twice")
    finally:
        db.close()
        shutil.rmtree(root)


def test_retrieve_filters_candidate_rejected_forgotten_superseded_and_invalid_dates():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id, "candidate zebra protocol")
        rejected = _candidate(db, source_id, "rejected zebra protocol")
        memory.reject(db, rejected["id"], reason="not accurate")
        forgotten = _candidate(db, source_id, "forgotten zebra protocol")
        memory.forget(db, forgotten["id"], reason="no longer wanted")
        expired = _candidate(db, source_id, "expired zebra protocol",
                             valid_from_utc="2025-01-01T00:00:00Z",
                             valid_until_utc="2025-12-31T23:59:59Z")
        memory.confirm(db, expired["id"], reason="was true temporarily")
        future = _candidate(db, source_id, "future zebra protocol",
                            valid_from_utc="2030-01-01T00:00:00Z")
        memory.confirm(db, future["id"], reason="starts later")
        eligible = _candidate(db, source_id, "confirmed zebra protocol",
                              valid_from_utc="2025-01-01T00:00:00Z")
        memory.confirm(db, eligible["id"], reason="approved")

        results = memory.retrieve(db, "zebra protocol", now="2026-01-01T00:00:00Z")
        assert [item["id"] for item in results] == [eligible["id"]]
        assert all(item["lifecycle_status"] == "ACTIVE" and item["review_state"] == "CONFIRMED"
                   for item in results)
        assert memory.retrieve(db, "   ") == []
    finally:
        db.close()
        shutil.rmtree(root)


def test_list_and_filters_include_nonretrievable_review_states():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id, "candidate preference")
        confirmed = _candidate(db, source_id, "confirmed goal", memory_type="GOAL")
        memory.confirm(db, confirmed["id"], reason="approved")
        assert [m["id"] for m in memory.list_memories(db, review_state="candidate")] == [candidate["id"]]
        assert [m["id"] for m in memory.list_memories(db, memory_type="GOAL", lifecycle_status="active")] == [confirmed["id"]]
        with pytest.raises(memory.MemoryValidationError):
            memory.list_memories(db, lifecycle_status="DELETED")
        with pytest.raises(memory.MemoryValidationError):
            memory.list_memories(db, limit=101)
    finally:
        db.close()
        shutil.rmtree(root)


def test_v2_database_migrates_additively_without_converting_entries():
    root, db = _db()
    try:
        source_id = _source(db, "Pre-existing source stays intact")
        db.conn.execute("DROP TABLE memory_history")
        db.conn.execute("DROP TABLE memory_sources")
        db.conn.execute("DROP TABLE memories")
        db.conn.execute("DROP TABLE canonical_memories_fts")
        db.conn.execute("UPDATE meta SET value='2' WHERE key='schema_version'")
        db.conn.commit()
        db.close()

        migrated = storage.JarvisDB(root / "data/jarvis.db")
        assert migrated.schema_version == 9
        assert migrated.get(source_id)["entry"]["text_raw"] == "Pre-existing source stays intact"
        assert migrated.search_fts("pre-existing source", limit=5)
        assert migrated.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
        assert migrated.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"] == "9"
        migrated.close()
    finally:
        try:
            db.close()
        except Exception:
            pass
        shutil.rmtree(root)


def test_canonical_memory_tables_and_foreign_keys_are_separate_from_entries():
    root, db = _db()
    try:
        names = {row["name"] for row in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"entries", "entries_fts", "memories", "memory_sources", "memory_history",
                "canonical_memories_fts"} <= names
        source_id = _source(db)
        item = _candidate(db, source_id)
        with pytest.raises(Exception):
            db.conn.execute("DELETE FROM entries WHERE id=?", (source_id,))
        assert db.get(source_id) is not None
        assert db.conn.execute("SELECT claim FROM memories WHERE id=?", (item["id"],)).fetchone()["claim"] == item["claim"]
    finally:
        db.close()
        shutil.rmtree(root)
