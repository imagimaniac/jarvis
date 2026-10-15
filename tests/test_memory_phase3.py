import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core import memory, memory_workflow, personal_context, storage


def _db():
    root = Path(tempfile.mkdtemp())
    (root / "data").mkdir()
    return root, storage.JarvisDB(root / "data/jarvis.db")


def _source(db, text="User statement"):
    return db.insert_entry(text, source="test", origin_kind="self", raw_path="private/source.txt",
                           raw_relpath="private/source.txt")


def _candidate(db, source_id, claim, *, kind="FACT", **kwargs):
    args = {
        "memory_type": kind, "claim": claim, "source_entry_ids": [source_id],
        "source_attribution": "user statement", "reason": "explicitly stated",
    }
    args.update(kwargs)
    return memory.create_candidate(db, **args)


def _confirmed(db, source_id, claim, *, kind="FACT", **kwargs):
    item = _candidate(db, source_id, claim, kind=kind, **kwargs)
    return memory_workflow.confirm_candidate(db, item["id"], reason="user approved")


def test_workflow_confirms_candidate_in_place_and_preserves_rationale_provenance_history():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id, "User prefers concise answers", candidate_reason="explicit preference")
        before_count = db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"]
        confirmed = memory_workflow.confirm_candidate(db, candidate["id"], reason="confirmed by user")
        assert confirmed["id"] == candidate["id"]
        assert confirmed["review_state"] == "CONFIRMED"
        assert confirmed["candidate_reason"] == "explicit preference"
        assert confirmed["sources"][0]["entry_id"] == source_id
        assert db.conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()["n"] == before_count
        assert [event["action"] for event in memory.get_history(db, candidate["id"])] == ["CANDIDATE_CREATED", "CONFIRMED"]
    finally:
        db.close()
        shutil.rmtree(root)


def test_workflow_rejects_candidate_and_keeps_audited_record():
    root, db = _db()
    try:
        source_id = _source(db)
        item = _candidate(db, source_id, "User may like cycling", comparison_relation="NEW")
        result = memory_workflow.reject_candidate(db, item["id"], reason="not durable")
        assert result["review_state"] == "REJECTED"
        assert memory.get_history(db, item["id"])[-1]["reason"] == "not durable"
        assert result["sources"][0]["entry_id"] == source_id
    finally:
        db.close()
        shutil.rmtree(root)


def test_workflow_updates_confirmed_memory_with_history_and_original_provenance():
    root, db = _db()
    try:
        source_id, correction_id = _source(db), _source(db, "User clarified preference")
        item = _confirmed(db, source_id, "User prefers concise replies", kind="PREFERENCE")
        result = memory_workflow.update_confirmed_memory(
            db, item["id"], reason="user clarified", claim="User prefers direct concise replies",
            source_entry_ids=[source_id, correction_id],
        )
        assert result["id"] == item["id"]
        assert {source["entry_id"] for source in result["sources"]} == {source_id, correction_id}
        event = memory.get_history(db, item["id"])[-1]
        assert event["action"] == "UPDATED"
        assert event["previous_state"]["claim"] == "User prefers concise replies"
        assert event["new_state"]["claim"] == "User prefers direct concise replies"
    finally:
        db.close()
        shutil.rmtree(root)


def test_workflow_supersedes_and_forgets_through_existing_audited_service():
    root, db = _db()
    try:
        source_id = _source(db)
        old = _confirmed(db, source_id, "User lives in Oslo", kind="HISTORY")
        new = _confirmed(db, source_id, "User lives in Delhi", kind="HISTORY")
        result = memory_workflow.supersede_memory(db, old["id"], new["id"], reason="user moved")
        assert result["lifecycle_status"] == "SUPERSEDED"
        assert memory.get_history(db, old["id"])[-1]["reason"] == "user moved"
        forgotten = memory_workflow.forget_memory(db, new["id"], reason="user requested forget")
        assert forgotten["lifecycle_status"] == "FORGOTTEN"
        assert memory.retrieve(db, "Delhi") == []
        assert db.get(source_id) is not None
    finally:
        db.close()
        shutil.rmtree(root)


@pytest.mark.parametrize("operation", ["update", "supersede", "forget"])
def test_decision_workflow_rejects_operations_on_unconfirmed_candidate(operation):
    root, db = _db()
    try:
        source_id = _source(db)
        item = _candidate(db, source_id, "User prefers tea", kind="PREFERENCE")
        with pytest.raises(memory.MemoryTransitionError):
            if operation == "update":
                memory_workflow.update_confirmed_memory(db, item["id"], reason="edit", claim="User prefers coffee")
            elif operation == "supersede":
                replacement = _candidate(db, source_id, "Replacement preference", kind="PREFERENCE")
                memory_workflow.supersede_memory(db, item["id"], replacement["id"], reason="replace")
            else:
                memory_workflow.forget_memory(db, item["id"], reason="forget")
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_uses_relevant_confirmed_memory_and_hides_internal_metadata():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User is building a local Jarvis assistant", kind="PROJECT")
        context = personal_context.build_personal_context(db, "How is my Jarvis assistant project going?")
        assert "building a local Jarvis assistant" in context
        assert "background information" in context
        assert "take precedence" in context
        assert "memory system" in context
        assert "according to your memory" not in context.casefold()
        assert "private/source.txt" not in context
        assert "ACTIVE" not in context and "CONFIRMED" not in context and "CANDIDATE" not in context
        assert str(source_id) not in context
        assert "2026-" not in context
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_excludes_candidates_rejected_forgotten_and_superseded_memories():
    root, db = _db()
    try:
        source_id = _source(db)
        candidate = _candidate(db, source_id, "Candidate memory about orchid planning")
        rejected = _candidate(db, source_id, "Rejected memory about orchid planning")
        memory_workflow.reject_candidate(db, rejected["id"], reason="not approved")
        forgotten = _confirmed(db, source_id, "Forgotten memory about orchid planning")
        memory_workflow.forget_memory(db, forgotten["id"], reason="forgotten")
        old = _confirmed(db, source_id, "Superseded memory about orchid planning")
        replacement = _confirmed(db, source_id, "Replacement memory about orchid planning")
        memory_workflow.supersede_memory(db, old["id"], replacement["id"], reason="replaced")
        context = personal_context.build_personal_context(db, "Tell me about orchid planning")
        assert "Candidate memory" not in context
        assert "Rejected memory" not in context
        assert "Forgotten memory" not in context
        assert "Superseded memory" not in context
        assert "Replacement memory" in context
        assert candidate["id"] not in context
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_respects_validity_window_and_returns_empty_for_no_match():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User has temporary cobalt project", kind="TEMPORARY",
                   valid_from_utc="2026-01-01T00:00:00Z", valid_until_utc="2026-02-01T00:00:00Z")
        assert personal_context.build_personal_context(
            db, "What is the cobalt project?", now="2026-03-01T00:00:00Z"
        ) == ""
        assert personal_context.build_personal_context(db, "Tell me about unrelated topic") == ""
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_is_bounded_by_memory_count_and_character_size():
    root, db = _db()
    try:
        source_id = _source(db)
        for i in range(6):
            _confirmed(db, source_id, f"User is building project atlas item {i} with details", kind="PROJECT", importance=5)
        context = personal_context.build_personal_context(
            db, "Tell me about project atlas", max_memories=2, max_chars=800
        )
        assert len(context) <= 800
        assert context.count("\n- ") <= 2
    finally:
        db.close()
        shutil.rmtree(root)


def test_importance_is_an_explainable_ranking_tiebreaker():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User works with cobalt migration", importance=1)
        _confirmed(db, source_id, "User works with cobalt reporting", importance=5)
        context = personal_context.build_personal_context(db, "What about cobalt work?", max_memories=1)
        assert "cobalt reporting" in context
    finally:
        db.close()
        shutil.rmtree(root)


def test_current_message_correction_overrides_conflicting_older_preference():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User prefers moving to Oslo", kind="PREFERENCE")
        context = personal_context.build_personal_context(
            db, "I've decided I don't want to move to Oslo anymore. What should I consider?"
        )
        assert "Oslo" not in context
        assert context == ""  # the conflicting older preference is withheld entirely
        assert memory.retrieve(db, "Oslo")  # context construction does not mutate canonical memory
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_marks_low_confidence_and_has_provider_neutral_message_seam():
    root, db = _db()
    try:
        source_id = _source(db)
        _confirmed(db, source_id, "User may be learning ceramics", kind="LEARNING", confidence=0.5)
        context = personal_context.build_personal_context(db, "What about ceramics learning?")
        assert "Possibly:" in context
        message = personal_context.as_system_message(context)
        assert message == {"role": "system", "content": context}
        assert personal_context.as_system_message("") is None
    finally:
        db.close()
        shutil.rmtree(root)


def test_context_argument_bounds_are_validated():
    root, db = _db()
    try:
        with pytest.raises(memory.MemoryValidationError):
            personal_context.build_personal_context(db, "project", max_memories=0)
        with pytest.raises(memory.MemoryValidationError):
            personal_context.build_personal_context(db, "project", max_chars=100)
    finally:
        db.close()
        shutil.rmtree(root)
