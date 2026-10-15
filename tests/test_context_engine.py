"""Deterministic Context Engine (Phase 4): cross-rating memory + evidence lanes."""
import itertools
import json
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import context_engine, memory, memory_workflow, personal_context, recall, storage
from app.server import app

TOKEN = "context-engine-test-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FIXED = datetime(2026, 10, 8, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fixed_memory_clock(monkeypatch):
    """Pin the in-memory clock to FIXED so confirmed memories are valid at the
    engine's frozen 'now', while preserving creation order (monotonic steps
    strictly before FIXED keep retrieve's bm25/importance/newest tie-breaks deterministic)."""
    step = timedelta(minutes=1)
    base = FIXED - timedelta(hours=3)
    counter = itertools.count()

    def frozen_now():
        return (base + step * next(counter)).isoformat().replace("+00:00", "Z")

    monkeypatch.setattr(memory, "_now", frozen_now)


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TOKEN", TOKEN)
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    return TestClient(app, raise_server_exceptions=False)


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


def _confirmed(db, claim, *, kind="FACT", importance=4, confidence=0.90, is_sensitive=False,
               valid_until=None):
    kwargs = {"importance": importance, "confidence": confidence, "is_sensitive": is_sensitive}
    if valid_until is not None:
        kwargs["valid_until_utc"] = valid_until
    item = _candidate(db, _source(db), claim, kind=kind, **kwargs)
    return memory_workflow.confirm_candidate(db, item["id"], reason="user approved")


def _age(db, memory_id, days_before: float):
    stamped = (FIXED - timedelta(days=days_before)).isoformat().replace("+00:00", "Z")
    db.conn.execute(
        "UPDATE memories SET updated_at_utc=?, created_at_utc=? WHERE id=?",
        (stamped, stamped, memory_id),
    )


def _evidence_entry(item_id, text, importance="LOW", facet_status="MEMORY"):
    return {
        "id": item_id, "date": "2026-10-04T12:00:00", "status": "MEMORY",
        "facet_status": facet_status, "attribution": "you said this",
        "source": "cli", "category": None, "importance": importance,
        "text": text, "raw_relpath": "private/source.txt",
    }


def _events(response):
    events = []
    for frame in response.text.strip().split("\n\n"):
        if not frame.strip():
            continue
        name = next(line[6:].strip() for line in frame.splitlines() if line.startswith("event:"))
        data = next(line[5:].strip() for line in frame.splitlines() if line.startswith("data:"))
        events.append((name, json.loads(data)))
    return events


def test_empty_database_builds_empty_bounded_package():
    root, db = _db()
    try:
        package = context_engine.assemble(db, "Where do I work?", now=FIXED)
        assert package.evidence == []
        assert package.memories == []
        assert package.render_memory_lane() == ""
        assert package.as_dict()["evidence_filtered"] == 0
        assert "work" in package.keywords
        assert package.now.endswith("Z")
    finally:
        db.close()
        shutil.rmtree(root)


def test_offtopic_confirmed_memory_stays_below_the_relevance_floor():
    root, db = _db()
    try:
        _confirmed(db, "User enjoys alpine gardening in the mountains")
        package = context_engine.assemble(db, "Where do I work?", now=FIXED)
        assert package.memories == []
        assert package.render_memory_lane() == ""
    finally:
        db.close()
        shutil.rmtree(root)


def test_higher_confidence_ranks_above_equal_coverage_lower_confidence():
    root, db = _db()
    try:
        low = _confirmed(db, "User commutes to work by train", confidence=0.45)
        high = _confirmed(db, "User commutes to work by train", confidence=0.95)
        package = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert len(package.memories) == 2
        assert [item.memory_id for item in package.memories] == [high["id"], low["id"]]
        assert package.memories[0].score > package.memories[1].score
    finally:
        db.close()
        shutil.rmtree(root)


def test_ineligible_lifecycle_states_never_enter_the_memory_lane():
    root, db = _db()
    try:
        control = _confirmed(db, "User commutes to work by train")
        _candidate(db, _source(db), "User commutes to work by train",
                   confidence=0.9, importance=4)  # CANDIDATE
        rejected = _candidate(db, _source(db), "User commutes to work by train",
                              confidence=0.9, importance=4)
        memory_workflow.reject_candidate(db, rejected["id"], reason="not durable")
        forgotten = _confirmed(db, "User commutes to work by train")
        memory_workflow.forget_memory(db, forgotten["id"], reason="no longer true")
        old = _confirmed(db, "User commutes to work by train")
        replacement = _confirmed(db, "Alpine gardening is a mountain hobby")
        memory_workflow.supersede_memory(db, old["id"], replacement["id"], reason="restated")
        expired = _confirmed(
            db, "User commutes to work by train",
            valid_until=FIXED - timedelta(hours=1),
        )
        package = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert [item.memory_id for item in package.memories] == [control["id"]]
        assert all(item.quality == "CONFIRMED" for item in package.memories)
        assert expired["id"] not in {item.memory_id for item in package.memories}
    finally:
        db.close()
        shutil.rmtree(root)


def test_recency_tiers_rank_newer_memories_first_deterministically():
    root, db = _db()
    try:
        oldest = _confirmed(db, "User tends the gazebo at noon")
        middle = _confirmed(db, "User tends the gazebo at noon")
        newest = _confirmed(db, "User tends the gazebo at noon")
        _age(db, oldest["id"], 120)
        _age(db, middle["id"], 20)
        _age(db, newest["id"], 3)
        package = context_engine.assemble(db, "Tell me about the gazebo at noon", now=FIXED)
        assert [item.memory_id for item in package.memories] == [newest["id"], middle["id"], oldest["id"]]
        assert package.memories[0].score > package.memories[1].score > package.memories[2].score
    finally:
        db.close()
        shutil.rmtree(root)


def test_section_fit_boosts_a_goal_memory_for_a_goal_cued_query():
    root, db = _db()
    try:
        goal = _confirmed(db, "User wants to learn Mandarin fluently", kind="GOAL")
        pref = _confirmed(db, "User enjoys studying Mandarin grammar", kind="PREFERENCE")
        package = context_engine.assemble(db, "What is the user's goal for learning Mandarin?", now=FIXED)
        assert len(package.memories) == 2
        assert package.memories[0].memory_id == goal["id"]
        assert package.memories[0].score > package.memories[1].score
    finally:
        db.close()
        shutil.rmtree(root)


def test_emerging_stage_shifts_scores_and_established_stabilizes():
    root, db = _db()
    try:
        one = _confirmed(db, "User commutes to work by train")
        two = _confirmed(db, "User commutes to work by bus")
        emerging = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert emerging.memories[0].score < 1.0
        _confirmed(db, "User enjoys alpine gardening in the mountains")  # raises stage to ESTABLISHED
        established = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert established.memories[0].memory_id == emerging.memories[0].memory_id
        assert abs(established.memories[0].score - emerging.memories[0].score - context_engine.EMERGING_SHIFT) < 1e-9
    finally:
        db.close()
        shutil.rmtree(root)


def test_replacement_for_superseded_memory_ranks_higher():
    root, db = _db()
    try:
        old = _confirmed(db, "User wants to learn Rust", kind="GOAL")
        replacement = _confirmed(db, "User wants to learn Rust", kind="GOAL")
        memory_workflow.supersede_memory(db, old["id"], replacement["id"], reason="restated it")
        plain = _confirmed(db, "User wants to learn Rust", kind="GOAL")
        package = context_engine.assemble(db, "What does the user want to learn?", now=FIXED)
        assert len(package.memories) == 2
        assert package.memories[0].memory_id == replacement["id"]
        assert abs(package.memories[0].score - package.memories[1].score - context_engine.DEVELOPMENT_BUMP) < 1e-9
    finally:
        db.close()
        shutil.rmtree(root)


def test_items_carry_provenance_and_the_prompt_hides_ids_and_dates():
    root, db = _db()
    try:
        source_id = _source(db)
        item = _candidate(db, source_id, "User prefers concise explanations about local Jarvis projects", kind="PREFERENCE")
        confirmed = memory_workflow.confirm_candidate(db, item["id"], reason="user approved")
        package = context_engine.assemble(db, "Can you explain my local Jarvis project briefly?", now=FIXED)
        assert len(package.memories) == 1
        memory_item = package.memories[0]
        assert memory_item.source_entry_ids == (str(source_id),)
        assert memory_item.memory_type == "PREFERENCE"
        rendered = package.render_memory_lane()
        assert "background only" in rendered
        assert "concise explanations about local Jarvis projects" in rendered
        assert confirmed["id"] not in rendered
        assert "2026-" not in rendered
        as_dict = package.as_dict()["memories"][0]
        assert as_dict["memory_id"] == confirmed["id"]
        assert as_dict["qcover"] >= context_engine.MIN_MEMORY_COVERAGE
    finally:
        db.close()
        shutil.rmtree(root)


def test_rebuilding_is_exactly_deterministic():
    root, db = _db()
    try:
        _confirmed(db, "User commutes to work by train", confidence=0.95)
        _confirmed(db, "User prefers concise explanations", kind="PREFERENCE", confidence=0.8)
        first = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        second = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert first.as_dict() == second.as_dict()
    finally:
        db.close()
        shutil.rmtree(root)


def test_memory_budgets_enforce_max_memories_and_max_chars():
    root, db = _db()
    try:
        for _ in range(6):
            _confirmed(db, "User commutes to work by train")
        package = context_engine.assemble(db, "How does the user commute to work?", now=FIXED, max_memories=2)
        assert len(package.memories) == 2
        assert len(package.render_memory_lane(1000)) <= 1000
        assert len(package.render_memory_lane()) <= context_engine.MAX_MEMORY_CHARS
    finally:
        db.close()
        shutil.rmtree(root)


def test_assemble_never_mutates_configurable_state():
    root, db = _db()
    try:
        _confirmed(db, "User commutes to work by train")
        _confirmed(db, "User prefers concise explanations", kind="PREFERENCE")

        def counts():
            rows = {}
            for table in ("memories", "memory_history", "memory_sources"):
                rows[table] = db.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
            return rows

        before = counts()
        sample_before = db.conn.execute(
            "SELECT claim, review_state, lifecycle_status FROM memories ORDER BY created_at_utc, id"
        ).fetchall()
        context_engine.assemble(db, "How does the user commute to work?", now=FIXED).as_dict()
        context_engine.assemble(db, "How does the user commute to work?", now=FIXED).render_memory_lane()
        assert counts() == before
        sample_after = db.conn.execute(
            "SELECT claim, review_state, lifecycle_status FROM memories ORDER BY created_at_utc, id"
        ).fetchall()
        assert sample_after == sample_before
    finally:
        db.close()
        shutil.rmtree(root)


def test_evidence_pass_filter_keeps_relevant_synonym_and_high_importance():
    root, db = _db()
    try:
        evidence = [
            _evidence_entry(1, "I work at Acme.", importance="MEDIUM"),
            _evidence_entry(2, "whales navigate oceans silently", importance="LOW"),
            _evidence_entry(3, "planetary rings shimmer brightly", importance="HIGH"),
        ]
        kept = context_engine.apply_evidence_filter(evidence, "Where do I work?")
        assert [entry["id"] for entry in kept] == [1, 3]
        synonym = context_engine.apply_evidence_filter(
            [_evidence_entry(4, "progress in my professional role", importance="LOW")],
            "How is my work going?",
        )
        assert [entry["id"] for entry in synonym] == [4]
        assert context_engine.apply_evidence_filter([], "Where do I work?") == []
        package = context_engine.assemble(db, "Where do I work?", evidence=evidence, now=FIXED)
        assert [entry["id"] for entry in package.evidence] == [1, 3]
        assert package.evidence_filtered == 1
    finally:
        db.close()
        shutil.rmtree(root)


def test_external_only_evidence_skips_the_pass_filter():
    root, db = _db()
    try:
        evidence = [_evidence_entry(9, "whales navigate oceans silently", importance="LOW")]
        kept = context_engine.apply_evidence_filter(evidence, "When is my interview?", external_only=True)
        assert [entry["id"] for entry in kept] == [9]
        package = context_engine.assemble(db, "When is my interview?", evidence=evidence,
                                          external_only=True, now=FIXED)
        assert package.evidence_filtered == 0
        assert [entry["id"] for entry in package.evidence] == [9]
    finally:
        db.close()
        shutil.rmtree(root)


def test_evidence_citation_contract_preserved_through_engine():
    root, db = _db()
    try:
        evidence = [
            _evidence_entry(7, "I work at Acme.", importance="HIGH"),
            _evidence_entry(9, "whales navigate oceans silently", importance="LOW"),
        ]
        package = context_engine.assemble(db, "Where do I work?", evidence=evidence, now=FIXED)
        assert [entry["id"] for entry in package.evidence] == [7]
        formatted = recall.format_context(package.evidence)
        assert "I work at Acme." in formatted
        assert "whales navigate oceans" not in formatted
        check = recall.check_citations("You said [7].", package.evidence)
        assert check["cited"] == ["7"]
        assert check["uncited"] == []
        assert check["invented"] == []
    finally:
        db.close()
        shutil.rmtree(root)


def test_sensitive_confirmed_memory_excluded_by_default_and_included_when_requested():
    root, db = _db()
    try:
        _confirmed(db, "User commutes to work by train", is_sensitive=True)
        package = context_engine.assemble(db, "How does the user commute to work?", now=FIXED)
        assert package.memories == []
        package = context_engine.assemble(db, "How does the user commute to work?", now=FIXED,
                                          include_sensitive=True)
        assert len(package.memories) == 1
    finally:
        db.close()
        shutil.rmtree(root)


def test_conflicting_confirmed_preferences_both_ranked_without_resolution():
    root, db = _db()
    try:
        _confirmed(db, "User prefers concise explanations", kind="PREFERENCE", confidence=0.95)
        _confirmed(db, "User prefers elaborate explanations", kind="PREFERENCE", confidence=0.95)
        package = context_engine.assemble(db, "Should I keep my explanations concise or elaborate?", now=FIXED)
        claims = [item.claim for item in package.memories]
        assert len(claims) == 2
        assert any("concise explanations" in claim for claim in claims)
        assert any("elaborate explanations" in claim for claim in claims)
    finally:
        db.close()
        shutil.rmtree(root)


def test_invalid_arguments_are_rejected():
    root, db = _db()
    try:
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, 123)
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", max_memories=True)
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", max_memories=0)
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", max_memory_chars=100)
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", evidence="not-a-list")
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", now="2026-10-08")
        with pytest.raises(context_engine.ContextEngineError):
            context_engine.assemble(db, "hi", now="not-a-date")
    finally:
        db.close()
        shutil.rmtree(root)


def _server_db_path():
    import os
    return Path(os.environ["JARVIS_DB_PATH"])


def test_promoted_primitives_keep_exact_behavior_and_are_public():
    for private_name in ("_tokens", "_overridden", "_recency_bonus", "_TYPE_WEIGHT", "_CONTEXT_GUIDANCE", "_format_claim"):
        assert not hasattr(personal_context, private_name), private_name + " should be promoted, not duplicated"
    assert personal_context.tokens("How does the user commute to work?") == {"user", "commute", "work"}
    assert personal_context.tokens("") == set()
    assert personal_context.is_overridden(
        "User plans to hike this weekend", "I changed my mind: cancel the plans"
    )
    assert not personal_context.is_overridden("User plans to hike this weekend", "What are this weekend's plans?")
    assert personal_context.recency_bonus("2026-10-05T00:00:00Z", FIXED) == 0.08
    assert personal_context.recency_bonus("2026-09-01T00:00:00Z", FIXED) == 0.02
    assert personal_context.recency_bonus("not-a-date", FIXED) == 0.0
    assert personal_context.format_claim(
        {"claim": "User prefers   concise answers", "confidence": 0.5}
    ) == "Possibly: User prefers concise answers"
    assert personal_context.format_claim(
        {"claim": "User prefers   concise answers", "confidence": 0.9}
    ) == "User prefers concise answers"
    assert personal_context.TYPE_WEIGHT["PREFERENCE"] == 0.10
    assert personal_context.TYPE_WEIGHT["TEMPORARY"] == -0.08
    assert isinstance(personal_context.CONTEXT_GUIDANCE, str)
    assert "must not be cited" in personal_context.CONTEXT_GUIDANCE


def test_engine_and_personal_context_render_identical_wording_for_same_memory():
    root, db = _db()
    try:
        _confirmed(db, "User prefers concise explanations about local Jarvis projects",
                   kind="PREFERENCE", confidence=0.9)
        message = "Can you explain my local Jarvis project briefly?"
        lane = context_engine.assemble(db, message, now=FIXED).render_memory_lane()
        personal = personal_context.build_personal_context(db, message, now=FIXED)
        assert lane == personal
        assert lane.startswith("Personal context is passive background")
    finally:
        db.close()
        shutil.rmtree(root)


def test_engine_memory_lane_default_on_and_recall_contract(client, monkeypatch):
    from app.server import recall as server_recall
    db = storage.JarvisDB(_server_db_path())
    try:
        _confirmed(db, "User prefers concise explanations about local Jarvis projects", kind="PREFERENCE")
    finally:
        db.close()
    calls = {}

    def fake_recall(passed_db, query, k, **kwargs):
        calls["recall"] = (query, k, kwargs)
        return []

    real_prompt = server_recall.answer_prompt

    def prompt_spy(question, items, history, **kwargs):
        calls["prompt"] = (question, items, history)
        return real_prompt(question, items, history, **kwargs)

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            calls["messages"] = messages
            yield "Context noted. "
            yield "[done]"

    monkeypatch.setattr(server_recall, "recall", fake_recall)
    monkeypatch.setattr(server_recall, "answer_prompt", prompt_spy)
    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)

    response = client.post("/api/chat", json={"message": "Can you explain my local Jarvis project briefly?"},
                           headers=AUTH)
    events = _events(response)
    assert response.status_code == 200
    assert calls["recall"] == ("Can you explain my local Jarvis project briefly?", 5, {"include_external": False})
    texts = "\n".join(message.get("content", "") for message in calls["messages"])
    assert "concise explanations about local Jarvis projects" in texts
    assert "background only" in texts
    assert [(name, data.get("citations", {}).get("ok")) for name, data in events if name == "done"] == [("done", True)]


def test_server_falls_back_to_personal_context_when_context_engine_disabled(client, monkeypatch):
    monkeypatch.setenv("JARVIS_CONTEXT_ENGINE", "0")
    from app.server import personal_context as server_personal
    db = storage.JarvisDB(_server_db_path())
    try:
        _confirmed(db, "User prefers concise explanations about local Jarvis projects", kind="PREFERENCE")
    finally:
        db.close()
    captured = {}
    real = server_personal.build_personal_context

    def spy(passed_db, message, **kwargs):
        captured["context"] = real(passed_db, message, **kwargs)
        return captured["context"]

    monkeypatch.setattr(server_personal, "build_personal_context", spy)

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Context noted. [done]"

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)

    response = client.post("/api/chat", json={"message": "Can you explain my local Jarvis project briefly?"},
                           headers=AUTH)
    assert response.status_code == 200
    assert "concise explanations about local Jarvis projects" in captured["context"]
    assert _events(response)[-1][0] == "done"


def test_broad_mode_surfaces_important_memory_without_keyword_overlap():
    root, db = _db()
    try:
        _confirmed(db, "User is a manager in business strategy and analytics",
                   kind="FACT", importance=5, confidence=0.95)
        message = "What kind of person am I and what am I trying to become?"
        narrow = context_engine.assemble(db, message, now=FIXED, broad=False)
        broad = context_engine.assemble(db, message, now=FIXED, broad=True)
        assert narrow.memories == []  # no lexical overlap -> below the coverage floor
        assert any("manager in business strategy" in item.claim for item in broad.memories)
        assert broad.overview and "manager in business strategy" in broad.overview
    finally:
        db.close()
        shutil.rmtree(root)


def test_broad_and_narrow_rendering_contracts():
    root, db = _db()
    try:
        _confirmed(db, "User is building a private personal AI assistant called Jarvis",
                   kind="PROJECT", importance=5)
        broad = context_engine.assemble(db, "What do you know about me?", now=FIXED, broad=True)
        narrow = context_engine.assemble(db, "What do you know about me?", now=FIXED, broad=False)
        assert broad.broad is True and narrow.broad is False
        lane = broad.render_personal_lane()
        assert "Personal profile mode" in lane
        assert "Personal overview" in lane and "Jarvis" in lane
        # Narrow rendering is byte-identical across both entry points.
        assert narrow.render_personal_lane() == narrow.render_memory_lane()
        assert "Personal profile mode" not in narrow.render_memory_lane()
    finally:
        db.close()
        shutil.rmtree(root)


def test_personal_query_chat_injects_overview_and_never_denies(client, monkeypatch):
    from app.server import recall as server_recall
    db = storage.JarvisDB(_server_db_path())
    try:
        _confirmed(db, "User is a manager in business strategy and analytics",
                   kind="FACT", importance=5, confidence=0.95)
    finally:
        db.close()
    calls = {}

    def fake_recall(passed_db, query, k, **kwargs):
        calls["recall"] = (query, k, kwargs)
        return []

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            calls["messages"] = messages
            yield "You build Jarvis. [done]"

    monkeypatch.setattr(server_recall, "recall", fake_recall)
    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)

    response = client.post("/api/chat", json={"message": "What do you know about me?"}, headers=AUTH)
    assert response.status_code == 200
    text = "\n".join(message.get("content", "") for message in calls["messages"])
    assert "manager in business strategy and analytics" in text
    assert "Personal profile mode" in text
    assert "Do not say you have no information about the user" in text


def test_normal_question_does_not_receive_personal_overview(client, monkeypatch):
    from app.server import recall as server_recall
    db = storage.JarvisDB(_server_db_path())
    try:
        _confirmed(db, "User is a manager in business strategy and analytics",
                   kind="FACT", importance=5, confidence=0.95)
    finally:
        db.close()
    calls = {}

    def fake_recall(passed_db, query, k, **kwargs):
        return []

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            calls["messages"] = messages
            yield "A heat pump moves heat. [done]"

    monkeypatch.setattr(server_recall, "recall", fake_recall)
    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)

    response = client.post("/api/chat", json={"message": "How does a heat pump work?"}, headers=AUTH)
    assert response.status_code == 200
    text = "\n".join(message.get("content", "") for message in calls["messages"])
    assert "Personal profile mode" not in text
    assert "Personal overview" not in text