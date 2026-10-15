import json
import shutil
import tempfile
from pathlib import Path

import pytest

from app.core import memory, personal_context, storage
from scripts.seed_canonical_memories import load_records, seed_records


ROOT = Path(__file__).resolve().parents[1]
SEED_FILE = ROOT / "seeds" / "canonical_memories.example.json"


def _db():
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    return root, storage.JarvisDB(root / "data/jarvis.db")


def test_seed_manifest_contains_personal_memory_seed_v1_and_supported_types():
    records = load_records(SEED_FILE)
    assert len(records) == 12
    assert {item["type"] for item in records} == {"PREFERENCE", "LEARNING", "PROJECT", "GOAL"}
    assert {kind: sum(item["type"] == kind for item in records)
            for kind in {item["type"] for item in records}} == {
                "PREFERENCE": 5, "LEARNING": 3, "PROJECT": 2, "GOAL": 2,
            }
    assert len({item["key"] for item in records}) == len(records)
    assert all(item["sensitive"] is False for item in records)
    assert all(item["confidence"] == 1.0 for item in records)


def test_seed_uses_canonical_service_confirmation_and_preserves_source_history():
    root, db = _db()
    try:
        existing_id = db.insert_entry("Existing ordinary capture remains intact", source="cli")
        before_count = db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        records = load_records(SEED_FILE)
        summary = seed_records(db, records, SEED_FILE)
        assert summary == {"added": len(records), "skipped": 0}
        assert db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"] == before_count + len(records)
        assert db.get(existing_id)["entry"]["text_raw"] == "Existing ordinary capture remains intact"

        seeded = [item for item in memory.list_memories(db, limit=100)
                  if item["source_attribution"].startswith("Curated user-approved personal seed")]
        assert len(seeded) == len(records)
        for item in seeded:
            assert item["review_state"] == "CONFIRMED"
            assert item["lifecycle_status"] == "ACTIVE"
            assert item["is_sensitive"] == 0
            assert item["source_attribution"].startswith("Curated user-approved personal seed")
            assert len(item["sources"]) == 1
            source = item["sources"][0]
            assert source["source"] == "canonical_memory_seed"
            assert source["origin_kind"] == "self"
            history = memory.get_history(db, item["id"])
            assert [event["action"] for event in history] == ["CANDIDATE_CREATED", "CONFIRMED"]
            assert history[-1]["reason"] == "Explicitly confirmed curated personal seed"
            assert history[0]["new_state"]["source_entry_ids"] == [source["entry_id"]]
    finally:
        db.close()
        shutil.rmtree(root)


def test_seed_is_idempotent_and_seed_sources_do_not_enter_ordinary_evidence_search():
    root, db = _db()
    try:
        records = load_records(SEED_FILE)
        assert seed_records(db, records, SEED_FILE) == {"added": len(records), "skipped": 0}
        entry_count = db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"]
        memory_count = db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        assert seed_records(db, records, SEED_FILE) == {"added": 0, "skipped": len(records)}
        assert db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"] == entry_count
        assert db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == memory_count
        assert db.search_fts("private personal assistant", limit=10) == []
        assert all(item["source"] != "canonical_memory_seed" for item in db.recent(limit=20))
        assert personal_context.build_personal_context(db, "Tell me about my private Jarvis assistant")
    finally:
        db.close()
        shutil.rmtree(root)


def test_seed_context_selects_relevant_records_and_omits_unrelated_records_and_provenance():
    root, db = _db()
    try:
        seed_records(db, load_records(SEED_FILE), SEED_FILE)
        context = personal_context.build_personal_context(db, "How is my private Jarvis assistant project going?")
        assert "private personal AI assistant called Jarvis" in context
        unrelated = personal_context.build_personal_context(db, "How do I bake sourdough bread?")
        assert unrelated == ""

        seeded = memory.list_memories(db, limit=100)
        hidden = [str(item["id"]) for item in seeded]
        hidden.extend(item["source_attribution"] for item in seeded)
        hidden.extend(item["created_at_utc"] for item in seeded)
        hidden.extend("canonical_memory_seed" for _ in seeded)
        for value in hidden:
            assert value not in context
        assert "ACTIVE" not in context and "CONFIRMED" not in context
        assert str(SEED_FILE) not in context
    finally:
        db.close()
        shutil.rmtree(root)


def test_seeded_memories_rank_by_importance_when_relevance_matches(tmp_path):
    root, db = _db()
    seed_file = tmp_path / "ranked.json"
    records = [
        {"key": "lower_priority", "type": "PROJECT", "claim": "User develops blue orbit planning tools",
         "importance": 1, "confidence": 1.0, "sensitive": False, "rationale": "test ranking"},
        {"key": "higher_priority", "type": "PROJECT", "claim": "User develops blue orbit research tools",
         "importance": 5, "confidence": 1.0, "sensitive": False, "rationale": "test ranking"},
    ]
    seed_file.write_text(json.dumps({"schema_version": 1, "records": records}), encoding="utf-8")
    try:
        seed_records(db, load_records(seed_file), seed_file)
        context = personal_context.build_personal_context(db, "Tell me about blue orbit", max_memories=1)
        assert "research tools" in context
        assert "planning tools" not in context
    finally:
        db.close()
        shutil.rmtree(root)


@pytest.mark.parametrize("mutate", [
    lambda record: record.update(sensitive=True),
    lambda record: record.update(type="UNKNOWN"),
    lambda record: record.update(confidence=2),
    lambda record: record.update(valid_from_utc="2026-01-01T00:00:00"),
    lambda record: record.update(unexpected="value"),
])
def test_seed_rejects_unsafe_or_invalid_manifest_before_database_changes(tmp_path, mutate):
    source = load_records(SEED_FILE)[0]
    mutate(source)
    seed_file = tmp_path / "bad.json"
    seed_file.write_text(json.dumps({"schema_version": 1, "records": [source]}), encoding="utf-8")
    root, db = _db()
    try:
        with pytest.raises(ValueError):
            load_records(seed_file)
        assert db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"] == 0
        assert db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == 0
    finally:
        db.close()
        shutil.rmtree(root)
