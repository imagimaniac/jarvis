"""Evolving User Model projection (Phase 3): structure, qualities, provenance,
determinism, bounds, and strict read-only safety."""
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.core import memory, storage, user_model
from app.core.user_model import ALL_SECTIONS, DURABLE_SECTIONS, UserModelError


def _db():
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    db = storage.JarvisDB(root / "data/jarvis.db")
    return root, db


def _source(db, text="Source"):
    return db.insert_entry(text, source="cli", origin_kind="self",
                           raw_path="memory/raw/source.txt", raw_relpath="source.txt")


def _candidate(db, source_id, claim, **kwargs):
    values = {
        "memory_type": "FACT",
        "claim": claim,
        "source_entry_ids": [source_id],
        "source_attribution": "user statement",
        "reason": "review",
        "confidence": 0.8,
        "importance": 3,
    }
    values.update(kwargs)
    return memory.create_candidate(db, **values)


def _confirmed(db, source_id, claim, **kwargs):
    item = _candidate(db, source_id, claim, **kwargs)
    return memory.confirm(db, item["id"], reason="user approved")


def _build(db, **kwargs):
    return user_model.build_user_model(db, **kwargs)


# ---------------------------------------------------------------------------
# EMPTY MODEL
# ---------------------------------------------------------------------------

def test_empty_model_is_bounded_and_undefined():
    root, db = _db()
    try:
        model = _build(db)
        assert not model.defined
        assert model.stage == "EMERGING"
        assert model.confirmed_count == 0
        assert model.pending_count == 0
        for section in ALL_SECTIONS:
            assert model.sections[section] == []
        assert model.quality_counts == {"CONFIRMED": 0, "INFERRED": 0, "OBSERVED_PATTERN": 0, "TEMPORARY": 0}
        assert set(model.as_dict()["sections"].keys()) == set(ALL_SECTIONS)
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# SINGLE CONFIRMED MEMORY
# ---------------------------------------------------------------------------

def test_single_confirmed_memory_builds_facet_with_provenance():
    root, db = _db()
    try:
        source_id = _source(db)
        goal = _confirmed(db, source_id, "User wants to run a half marathon", memory_type="GOAL")
        model = _build(db)
        assert model.defined
        assert model.stage == "EMERGING"
        assert model.confirmed_count == 1
        assert [f.claim for f in model.sections["goals"]] == [goal["claim"]]
        facet = model.sections["goals"][0]
        assert facet.quality == "CONFIRMED"
        assert facet.memory_id == goal["id"]
        assert list(facet.source_entry_ids) == [source_id]
        assert facet.section == "goals"
        assert all(not model.sections[s] for s in ALL_SECTIONS if s != "goals")
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# MULTIPLE MEMORY TYPES
# ---------------------------------------------------------------------------

def test_each_memory_type_lands_in_its_durable_section():
    root, db = _db()
    try:
        source_id = _source(db)
        by_type = {
            "FACT": "User is fluent in German",
            "GOAL": "User aims to write a book",
            "PROJECT": "User is building a home lab",
            "PREFERENCE": "User prefers early espresso calls",
            "DECISION": "User decided to switch to a standing desk",
            "LEARNING": "User learned sqlite window functions",
            "HABIT": "User usually walks after dinner",
            "HISTORY": "User moved to Lisbon in June",
        }
        for kind, claim in by_type.items():
            _confirmed(db, source_id, claim, memory_type=kind)
        model = _build(db)
        expected_sections = {
            "FACT": "facts", "GOAL": "goals", "PROJECT": "projects",
            "PREFERENCE": "preferences", "DECISION": "decisions",
            "LEARNING": "learning", "HABIT": "habits", "HISTORY": "history",
        }
        for kind, section in expected_sections.items():
            assert [f.claim for f in model.sections[section]] == [by_type[kind]], kind
        assert model.sections["temporary"] == []
        assert model.confirmed_count == 8
        assert model.stage == "ESTABLISHED"
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# INACTIVE / SUPERSEDED (changed preferences & goals)
# ---------------------------------------------------------------------------

def test_changed_preference_superseded_is_development_not_current():
    root, db = _db()
    try:
        source_id = _source(db)
        old = _confirmed(db, source_id, "User prefers dark mode", memory_type="PREFERENCE")
        new = _confirmed(db, source_id, "User prefers light mode", memory_type="PREFERENCE")
        memory.supersede(db, old["id"], new["id"], reason="preference changed")
        model = _build(db)
        assert [f.claim for f in model.sections["preferences"]] == [new["claim"]]
        assert all(f.memory_id != old["id"] for f in model.sections["preferences"])
        assert model.confirmed_count == 1
        changes = model.sections["development"]
        assert len(changes) == 1
        assert old["claim"] in changes[0].claim and new["claim"] in changes[0].claim
        assert changes[0].quality == "CONFIRMED"
        assert changes[0].related_memory_ids == (old["id"], new["id"])
        assert all(not model.sections[s] for s in DURABLE_SECTIONS if s != "preferences")
    finally:
        db.close()
        shutil.rmtree(root)


def test_changed_goal_superseded_surfaces_only_current():
    root, db = _db()
    try:
        source_id = _source(db)
        old = _confirmed(db, source_id, "User wants to learn Spanish", memory_type="GOAL")
        new = _confirmed(db, source_id, "User wants to learn German", memory_type="GOAL")
        memory.supersede(db, old["id"], new["id"], reason="goal updated")
        model = _build(db)
        assert [f.claim for f in model.sections["goals"]] == [new["claim"]]
        assert [c.related_memory_ids for c in model.sections["development"]] == [(old["id"], new["id"])]
    finally:
        db.close()
        shutil.rmtree(root)


def test_changed_preference_via_update_is_in_place_without_duplication():
    root, db = _db()
    try:
        source_id = _source(db)
        pref = _confirmed(db, source_id, "User prefers dark mode", memory_type="PREFERENCE")
        memory.update_memory(db, pref["id"], reason="preference changed",
                             claim="User prefers light mode")
        model = _build(db)
        assert [f.claim for f in model.sections["preferences"]] == ["User prefers light mode"]
        assert all(f.memory_id == pref["id"] for f in model.sections["preferences"])
        assert len(model.sections["preferences"]) == 1
        assert model.sections["development"] == []
    finally:
        db.close()
        shutil.rmtree(root)


def test_inactive_superseded_and_forgotten_memories_not_in_current_model():
    root, db = _db()
    try:
        source_id = _source(db)
        pref = _confirmed(db, source_id, "User likes macOS", memory_type="PREFERENCE")
        driver = _confirmed(db, source_id, "User wants an EV", memory_type="GOAL")
        repl = _confirmed(db, source_id, "User wants a hybrid", memory_type="GOAL")
        memory.supersede(db, driver["id"], repl["id"], reason="goal updated")
        memory.forget(db, pref["id"], reason="obsolete")
        model = _build(db)
        assert model.sections["preferences"] == []
        assert [f.claim for f in model.sections["goals"]] == [repl["claim"]]
        assert all(f.memory_id != pref["id"] for f in model.sections["goals"])
        assert any(c.related_memory_ids[0] == driver["id"] for c in model.sections["development"])
        assert model.confirmed_count == 1  # only the active replacement
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# INFERRED vs CONFIRMED
# ---------------------------------------------------------------------------

def test_candidate_is_inferred_and_never_enters_the_confirmed_model():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id, "User is a professional pilot", memory_type="GOAL")
        model = _build(db)
        assert model.sections["goals"] == []
        assert model.sections["pending"] == []
        assert model.pending_count == 1
        assert model.confirmed_count == 0

        with_pending = _build(db, include_pending=True)
        assert with_pending.sections["goals"] == []
        assert len(with_pending.sections["pending"]) == 1
        facet = with_pending.sections["pending"][0]
        assert facet.quality == "INFERRED"
        assert facet.memory_id == candidate["id"]
        assert facet.claim == candidate["claim"]
    finally:
        db.close()
        shutil.rmtree(root)


def test_single_conversation_cannot_redefine_the_user():
    root, db = _db()
    try:
        source_id = _source(db)
        _candidate(db, source_id, "User is a professional pilot", memory_type="GOAL")
        _candidate(db, source_id, "User is a barista", memory_type="FACT")
        model = _build(db, include_pending=True)
        assert model.sections["goals"] == []
        assert model.sections["facts"] == []
        assert model.pending_count == 2
        assert {f.quality for f in model.sections["pending"]} == {"INFERRED"}
    finally:
        db.close()
        shutil.rmtree(root)


def test_rejected_memory_is_absent_from_model_and_pending():
    root, db = _db()
    try:
        source_id = _source(db)
        item = _candidate(db, source_id, "User loves pineapple pizza", memory_type="PREFERENCE")
        memory.reject(db, item["id"], reason="not for the model")
        model = _build(db, include_pending=True)
        assert model.sections["preferences"] == []
        assert model.sections["pending"] == []
        assert model.pending_count == 0
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# CONFLICTING CONFIRMED MEMORIES (no silent resolution)
# ---------------------------------------------------------------------------

def test_conflicting_confirmed_memories_both_visible_with_provenance():
    root, db = _db()
    try:
        source_id = _source(db)
        a = _confirmed(db, source_id, "User loves winter", memory_type="PREFERENCE")
        b = _confirmed(db, source_id, "User hates winter", memory_type="PREFERENCE")
        model = _build(db)
        claims = {f.claim for f in model.sections["preferences"]}
        assert claims == {"User loves winter", "User hates winter"}
        assert {f.memory_id for f in model.sections["preferences"]} == {a["id"], b["id"]}
        assert model.quality_counts["CONFIRMED"] == 2
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# PATTERNS (OBSERVED_PATTERN requires repetition)
# ---------------------------------------------------------------------------

def test_pattern_requires_two_consistent_observations():
    root, db = _db()
    try:
        source_id = _source(db)
        first = _confirmed(db, source_id, "I usually drink espresso in the morning", memory_type="HABIT")
        model = _build(db)
        assert len(model.sections["habits"]) == 1
        assert model.sections["patterns"] == []
        assert model.quality_counts["OBSERVED_PATTERN"] == 0

        second = _confirmed(db, source_id, "I often drink espresso after lunch", memory_type="HABIT")
        model = _build(db)
        assert len(model.sections["habits"]) == 2
        assert len(model.sections["patterns"]) == 1
        pattern = model.sections["patterns"][0]
        assert pattern.quality == "OBSERVED_PATTERN"
        assert pattern.related_memory_ids == tuple(sorted((first["id"], second["id"])))
        assert "Observed 2 times" in pattern.claim
        assert model.quality_counts["OBSERVED_PATTERN"] == 1
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# TEMPORARY
# ---------------------------------------------------------------------------

def test_temporary_memories_are_flagged_and_separated():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "Working from home this week", memory_type="TEMPORARY")
        model = _build(db)
        assert len(model.sections["temporary"]) == 1
        facet = model.sections["temporary"][0]
        assert facet.quality == "TEMPORARY"
        assert all(not model.sections[s] for s in DURABLE_SECTIONS)
        assert model.quality_counts["TEMPORARY"] == 1
    finally:
        db.close()
        shutil.rmtree(root)


def test_expired_confirmed_memory_is_not_current():
    root, db = _db()
    try:
        source_id = _source(db)
        now = datetime.now(timezone.utc)
        past = (now - timedelta(days=10)).isoformat().replace("+00:00", "Z")
        expired = (now - timedelta(days=5)).isoformat().replace("+00:00", "Z")
        _confirmed(db, source_id, "User attended a conference in March",
                   memory_type="FACT", valid_from_utc=past, valid_until_utc=expired)
        model = _build(db, now=(now + timedelta(days=1)).isoformat().replace("+00:00", "Z"))
        assert model.sections["facts"] == []
        assert model.confirmed_count == 0
        assert model.quality_counts["CONFIRMED"] == 0
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# SENSITIVE
# ---------------------------------------------------------------------------

def test_sensitive_confirmed_excluded_by_default():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User suffers from a private condition",
                   memory_type="FACT", is_sensitive=True)
        assert _build(db).sections["facts"] == []
        assert _build(db, include_sensitive=True).sections["facts"]
        assert _build(db).confirmed_count == 1  # counted, but not surfaced
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------------

def test_facet_carries_provenance_ids():
    root, db = _db()
    try:
        source_id = _source(db)
        item = _confirmed(db, source_id, "User is a morning person", memory_type="FACT")
        facet = _build(db).sections["facts"][0]
        assert facet.memory_id == item["id"]
        assert list(facet.source_entry_ids) == [source_id]
        assert facet.updated_at_utc == item["updated_at_utc"]
        assert facet.confidence == item["confidence"]
        assert facet.importance == item["importance"]
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# DETERMINISM & REBUILD
# ---------------------------------------------------------------------------

def test_rebuilding_is_exactly_deterministic():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User wants to learn Rust", memory_type="GOAL")
        _confirmed(db, source_id, "User prefers tea", memory_type="PREFERENCE")
        _confirmed(db, source_id, "I usually drink espresso in the morning", memory_type="HABIT")
        _confirmed(db, source_id, "I often drink espresso after lunch", memory_type="HABIT")
        _candidate(db, source_id, "User may relocate to Berlin", memory_type="GOAL")
        first = _build(db, include_pending=True).as_dict()
        second = _build(db, include_pending=True).as_dict()
        assert first == second
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# NO DUPLICATION / NO MUTATION
# ---------------------------------------------------------------------------

def test_each_current_memory_appears_exactly_once_across_durable_sections():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User wants a home brewery", memory_type="GOAL")
        _confirmed(db, source_id, "User keeps a sourdough starter", memory_type="HABIT")
        _confirmed(db, source_id, "Working abroad next month", memory_type="TEMPORARY")
        old = _confirmed(db, source_id, "User lived in Madrid", memory_type="FACT")
        new = _confirmed(db, source_id, "User lives in Berlin", memory_type="FACT")
        memory.supersede(db, old["id"], new["id"], reason="moved")
        _candidate(db, source_id, "User might resurface a startup", memory_type="GOAL")
        model = _build(db, include_pending=True)
        seen = []
        for section in DURABLE_SECTIONS + ("temporary",):
            seen.extend(f.memory_id for f in model.sections[section])
        assert len(seen) == len(set(seen)), "a current memory must not be duplicated"
        assert new["id"] in seen and old["id"] not in seen
        for facet in model.sections["development"]:
            assert old["id"] in facet.related_memory_ids
            assert old["id"] not in seen
    finally:
        db.close()
        shutil.rmtree(root)


def test_build_never_mutates_canonical_state():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User wants a home lab", memory_type="GOAL")
        _confirmed(db, source_id, "User prefers dark mode", memory_type="PREFERENCE")
        _candidate(db, source_id, "User might adopt a dog", memory_type="GOAL")

        def fingerprint():
            return {
                "memories": db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"],
                "history": db.conn.execute("SELECT COUNT(*) AS n FROM memory_history").fetchone()["n"],
                "entries": db.conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()["n"],
                "sources": db.conn.execute("SELECT COUNT(*) AS n FROM memory_sources").fetchone()["n"],
                "updated": db.conn.execute(
                    "SELECT MAX(updated_at_utc) AS m FROM memories").fetchone()["m"],
                "fts": db.conn.execute(
                    "SELECT COUNT(*) AS n FROM canonical_memories_fts").fetchone()["n"],
            }

        before = fingerprint()
        _build(db, include_pending=True)
        _build(db, include_pending=True)
        assert fingerprint() == before
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# BOUNDS & VALIDATION
# ---------------------------------------------------------------------------

def test_section_bounds_and_invalid_limits_rejected():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User is curious", memory_type="FACT")
        assert _build(db, limits={"facts": 0}).sections["facts"] == []
        assert _build(db, limits={"facts": 1}).sections["facts"]
        with pytest.raises(UserModelError):
            _build(db, limits={"bogus": 5})
        with pytest.raises(UserModelError):
            _build(db, limits={"facts": -1})
    finally:
        db.close()
        shutil.rmtree(root)


def test_invalid_now_rejected_and_timezone_required():
    root, db = _db()
    try:
        _confirmed(db, _source(db), "User is here", memory_type="FACT")
        with pytest.raises(UserModelError):
            _build(db, now=datetime(2026, 1, 1))  # naive
        with pytest.raises(UserModelError):
            _build(db, now="not-a-timestamp")
    finally:
        db.close()
        shutil.rmtree(root)


# ---------------------------------------------------------------------------
# STAGE
# ---------------------------------------------------------------------------

def test_stage_reflects_maturity_boundary():
    root, db = _db()
    try:
        source_id = _source(db)
        for i in range(2):
            _confirmed(db, source_id, f"User preference {i}", memory_type="FACT")
        assert _build(db).stage == "EMERGING"
        _confirmed(db, source_id, "User preference third", memory_type="FACT")
        assert _build(db).stage == "ESTABLISHED"
    finally:
        db.close()
        shutil.rmtree(root)

# ---------------------------------------------------------------------------
# PROMPT-SAFE RENDERING
# ---------------------------------------------------------------------------

def test_render_projection_groups_bounds_and_marks_uncertain():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User is a manager in business strategy and analytics",
                   memory_type="FACT", importance=5)
        _confirmed(db, source_id, "User prefers concise tables and structured comparisons",
                   memory_type="PREFERENCE", confidence=0.5)
        model = _build(db)
        text = user_model.render_projection(model)
        assert "Facts:" in text and "Preferences:" in text
        assert "manager in business strategy and analytics" in text
        assert "Possibly: User prefers concise tables and structured comparisons" in text
        # Prompt-safe: no provenance, ids, lifecycle labels, or timestamps.
        assert str(source_id) not in text
        assert "CONFIRMED" not in text and "ACTIVE" not in text and "2026" not in text
        # Deterministic.
        assert text == user_model.render_projection(_build(db))
    finally:
        db.close()
        shutil.rmtree(root)


def test_render_projection_empty_and_bounded():
    root, db = _db()
    try:
        assert user_model.render_projection(_build(db)) == ""
        source_id = _source(db)
        for i in range(6):
            _confirmed(db, source_id, f"User confirmed durable fact number {i}", importance=4)
        text = user_model.render_projection(_build(db), max_chars=120)
        assert 0 < len(text) <= 120
        assert not text.endswith("\n")
    finally:
        db.close()
        shutil.rmtree(root)
