"""Context topics group continuity threads without changing chat retrieval."""
from datetime import datetime, timedelta, timezone

import pytest

from app.core import (
    context_topics as topics,
    conversation_continuity as continuity,
    external_sources,
    memory,
    storage,
)


def _snapshot(topic="Jarvis topic grouping", problem="Continue the discussion about this subject."):
    return {
        "topic": topic,
        "current_problem_or_goal": problem,
        "decisions": [],
        "open_questions": [],
        "temporary_commitments": [],
        "expires_at": None,
    }


@pytest.fixture
def db(tmp_path):
    database = storage.JarvisDB(tmp_path / "topics.db")
    yield database
    database.close()


def test_topic_creation_uses_stable_normalized_key_and_prevents_duplicates(db):
    first = topics.create_topic(db, "Jarvis: Conversation Continuity")
    second = topics.create_topic(db, "  JARVIS conversation continuity!  ")

    assert first["topic_id"] == second["topic_id"]
    assert first["normalized_key"] == "jarvis conversation continuity"
    assert len(topics.list_topics(db)) == 1
    assert topics.get_topic(db, first["topic_id"])["display_name"] == "Jarvis: Conversation Continuity"
    assert topics.get_topic_by_key(db, "JARVIS conversation continuity") == first


def test_topic_names_are_validated_and_timestamp_is_timezone_aware(db):
    with pytest.raises(topics.ContextTopicError):
        topics.create_topic(db, "   ")
    with pytest.raises(topics.ContextTopicError):
        topics.create_topic(db, "x" * (topics.MAX_DISPLAY_NAME + 1))
    with pytest.raises(topics.ContextTopicError):
        topics.create_topic(db, "Project Alpha", now=datetime(2026, 1, 1))
    assert topics.list_topics(db) == []


def test_association_supports_primary_related_and_multiple_topics_per_thread(db):
    thread_repo = continuity.ContinuityRepository(db)
    first_thread = thread_repo.create_thread(_snapshot("Jarvis personal assistant"))
    second_thread = thread_repo.create_thread(_snapshot("Jarvis local memory"))
    primary = topics.create_topic(db, "Jarvis personal assistant")
    related = topics.create_topic(db, "Local memory architecture")

    association = topics.associate_explicitly(db, first_thread["thread_id"], primary["topic_id"])
    topics.associate_explicitly(
        db, first_thread["thread_id"], related["topic_id"], relation="RELATED"
    )
    topics.associate_explicitly(db, second_thread["thread_id"], primary["topic_id"], relation="RELATED")

    assert association["relation"] == "PRIMARY"
    assert association["confidence"] == 1.0
    assert association["provenance"] == "USER_EXPLICIT"
    assert {item["topic_id"] for item in topics.list_topics_for_thread(db, first_thread["thread_id"])} == {
        primary["topic_id"], related["topic_id"]
    }
    assert {item["thread_id"] for item in topics.list_threads_for_topic(db, primary["topic_id"])} == {
        first_thread["thread_id"], second_thread["thread_id"]
    }


def test_association_is_idempotent_and_primary_is_unique_per_thread(db):
    moment = datetime(2026, 4, 5, tzinfo=timezone.utc)
    thread = continuity.ContinuityRepository(db).create_thread(_snapshot("Jarvis continuity design"), now=moment)
    first_topic = topics.create_topic(db, "Jarvis continuity design", now=moment)
    second_topic = topics.create_topic(db, "Jarvis cross conversation links", now=moment)

    first = topics.associate_explicitly(db, thread["thread_id"], first_topic["topic_id"], now=moment)
    repeated = topics.associate_explicitly(db, thread["thread_id"], first_topic["topic_id"], now=moment)
    assert repeated == first
    assert db.conn.execute("SELECT count(*) AS n FROM continuity_thread_topics").fetchone()["n"] == 1

    with pytest.raises(topics.TopicAssociationError):
        topics.associate_explicitly(db, thread["thread_id"], second_topic["topic_id"])
    assert len(topics.list_topics_for_thread(db, thread["thread_id"])) == 1


def test_remove_association_and_thread_forget_cascade(db):
    repo = continuity.ContinuityRepository(db)
    thread = repo.create_thread(_snapshot("Jarvis memory architecture"))
    topic = topics.create_topic(db, "Jarvis memory architecture")
    topics.associate_explicitly(db, thread["thread_id"], topic["topic_id"])
    assert topics.remove_thread_topic(db, thread["thread_id"], topic["topic_id"])
    assert not topics.remove_thread_topic(db, thread["thread_id"], topic["topic_id"])

    topics.associate_explicitly(db, thread["thread_id"], topic["topic_id"])
    assert repo.forget(thread["thread_id"], thread["revision"])
    assert topics.list_threads_for_topic(db, topic["topic_id"]) == []
    assert topics.get_topic(db, topic["topic_id"]) is not None


def test_strong_match_is_exact_and_never_creates_or_uses_weak_topic_labels(db):
    repo = continuity.ContinuityRepository(db)
    exact = repo.create_thread(_snapshot("Jarvis conversation continuity"))
    weak = repo.create_thread(_snapshot("Gmail"))
    fuzzy = repo.create_thread(_snapshot("Gmail connector implementation"))
    topic = topics.create_topic(db, "Jarvis conversation continuity")

    linked = topics.associate_strong_match(db, exact["thread_id"])
    assert linked["topic_id"] == topic["topic_id"]
    assert linked["relation"] == "RELATED"
    assert linked["confidence"] == 0.95
    assert linked["provenance"] == "CONTINUITY_EXACT_MATCH"
    assert topics.associate_strong_match(db, weak["thread_id"]) is None
    assert topics.associate_strong_match(db, fuzzy["thread_id"]) is None
    assert len(topics.list_topics(db)) == 1
    assert topics.list_topics_for_thread(db, weak["thread_id"]) == []
    assert topics.list_topics_for_thread(db, fuzzy["thread_id"]) == []


def test_relevant_siblings_are_primary_topic_only_relevant_ranked_and_bounded(db):
    repo = continuity.ContinuityRepository(db)
    now = datetime.now(timezone.utc)
    current = repo.create_thread(_snapshot(
        "Jarvis continuity retrieval",
        "Select relevant snapshots for cross-conversation continuity retrieval.",
    ), now=now)
    best = repo.create_thread(_snapshot(
        "Jarvis continuity retrieval decisions",
        "Choose relevant snapshots and bounded retrieval for Jarvis continuity.",
    ), now=now)
    next_best = repo.create_thread(_snapshot(
        "Jarvis continuity history",
        "Persist older snapshots for continuity retrieval across conversations.",
    ), now=now - timedelta(days=1))
    third = repo.create_thread(_snapshot(
        "Jarvis continuity notes",
        "Review continuity retrieval notes from prior conversations.",
    ), now=now - timedelta(days=2))
    unrelated_same_topic = repo.create_thread(_snapshot(
        "Jarvis finance project",
        "Compare household budget categories and annual expenses.",
    ), now=now)
    other_topic = repo.create_thread(_snapshot(
        "Continuity retrieval project",
        "Jarvis continuity retrieval relevant snapshots across conversations.",
    ), now=now)

    topic = topics.create_topic(db, "Jarvis ongoing assistant project", now=now)
    other = topics.create_topic(db, "Separate conversation project", now=now)
    topics.associate_explicitly(db, current["thread_id"], topic["topic_id"])
    for thread in (best, next_best, third, unrelated_same_topic):
        topics.associate_explicitly(db, thread["thread_id"], topic["topic_id"], relation="RELATED")
    # This candidate is highly similar but reachable only through a sibling's
    # second topic; retrieval must not recurse through that relationship.
    topics.associate_explicitly(db, best["thread_id"], other["topic_id"], relation="RELATED")
    topics.associate_explicitly(db, other_topic["thread_id"], other["topic_id"])

    message = "How should Jarvis continuity retrieval select relevant snapshots?"
    selected = topics.relevant_sibling_threads(
        db, current, message, primary_match_score=0.7, now=now,
    )
    ids = [thread["thread_id"] for thread in selected]
    assert len(selected) == 2 == topics.MAX_TOPIC_SIBLINGS
    assert ids[0] == best["thread_id"]
    assert next_best["thread_id"] in ids
    assert third["thread_id"] not in ids
    assert unrelated_same_topic["thread_id"] not in ids
    assert other_topic["thread_id"] not in ids


def test_weak_match_low_confidence_and_missing_primary_do_not_expand_context(db):
    repo = continuity.ContinuityRepository(db)
    current = repo.create_thread(_snapshot("Jarvis continuity retrieval"))
    sibling = repo.create_thread(_snapshot("Jarvis continuity retrieval history"))
    topic = topics.create_topic(db, "Jarvis ongoing assistant project")
    topics.associate_explicitly(db, current["thread_id"], topic["topic_id"])
    topics.associate_explicitly(db, sibling["thread_id"], topic["topic_id"], relation="RELATED")

    assert topics.relevant_sibling_threads(
        db, current, "How should this work?", primary_match_score=0.1,
    ) == []
    assert topics.relevant_sibling_threads(
        db, current, "How should Jarvis continuity retrieval work?", primary_match_score=0.7,
        limit=0,
    ) == []

    db.conn.execute(
        "UPDATE continuity_thread_topics SET confidence=0.4 WHERE thread_id=? AND relation='PRIMARY'",
        (current["thread_id"],),
    )
    db.save()
    assert topics.relevant_sibling_threads(
        db, current, "How should Jarvis continuity retrieval work?", primary_match_score=0.7,
    ) == []


def test_thread_without_topic_keeps_existing_single_thread_context_exactly(db):
    thread = continuity.ContinuityRepository(db).create_thread(_snapshot(topic=""))
    old_context = continuity.format_context(thread)
    siblings = topics.relevant_sibling_threads(
        db, thread, "How should continuity work?", primary_match_score=0.7,
    )
    assert siblings == []
    assert continuity.format_context(thread, related_threads=siblings) == old_context


def test_related_context_uses_one_shared_budget_and_keeps_primary_first(db):
    primary = _snapshot("Jarvis continuity", "Primary focus " + "primary " * 100)
    sibling = _snapshot("Jarvis continuity", "Sibling focus " + "sibling " * 60)
    rendered = continuity.format_context(primary, related_threads=[sibling])
    assert len(rendered) <= continuity.MAX_CONTEXT_CHARS
    assert rendered.index("Primary focus") < rendered.index("Related discussion")
    assert "Conversation background for natural continuity" in rendered


def test_invalid_association_and_nonexistent_records_are_rejected(db):
    thread = continuity.ContinuityRepository(db).create_thread(_snapshot("Jarvis project design"))
    topic = topics.create_topic(db, "Jarvis architecture review")
    for kwargs in (
        {"relation": "OTHER", "confidence": 1.0, "provenance": "USER_EXPLICIT"},
        {"relation": "PRIMARY", "confidence": 1.1, "provenance": "USER_EXPLICIT"},
        {"relation": "PRIMARY", "confidence": 0.5, "provenance": "MODEL_GUESS"},
    ):
        with pytest.raises(topics.ContextTopicError):
            topics.associate_thread_topic(db, thread["thread_id"], topic["topic_id"], **kwargs)
    with pytest.raises(topics.ContextTopicError):
        topics.associate_explicitly(db, "missing-thread", topic["topic_id"])
    with pytest.raises(topics.ContextTopicError):
        topics.associate_explicitly(db, thread["thread_id"], "missing-topic")
    with pytest.raises(topics.TopicAssociationError):
        topics.associate_thread_topic(
            db, thread["thread_id"], topic["topic_id"], relation="RELATED",
            confidence=0.95, provenance="CONTINUITY_EXACT_MATCH",
        )
    assert topics.list_topics_for_thread(db, thread["thread_id"]) == []


def test_v7_migration_preserves_existing_continuity_memory_and_external_records(tmp_path):
    path = tmp_path / "migration.db"
    db = storage.JarvisDB(path)
    entry_id = db.insert_entry("Existing local evidence remains untouched", source="cli")
    candidate = memory.create_candidate(
        db, memory_type="FACT", claim="The existing confirmed memory remains intact.",
        source_entry_ids=[entry_id], source_attribution="user capture", reason="migration fixture",
    )
    confirmed = memory.confirm(db, candidate["id"], reason="migration fixture confirmation")
    external = external_sources.ExternalSourceRepository(db).upsert(external_sources.NormalizedExternalRecord(
        provider_type="gmail", external_id="existing-message", title="Existing external evidence",
        normalized_text="This Gmail record remains separate from topics and memory.",
    ))
    thread = continuity.ContinuityRepository(db).create_thread(_snapshot("Jarvis project history"))
    db.conn.execute("DROP TABLE continuity_thread_topics")
    db.conn.execute("DROP TABLE context_topics")
    db.conn.execute("UPDATE meta SET value='6' WHERE key='schema_version'")
    db.conn.commit()
    db.close()

    migrated = storage.JarvisDB(path)
    try:
        assert migrated.schema_version == 9
        assert migrated.get(entry_id)["entry"]["text_raw"] == "Existing local evidence remains untouched"
        assert memory.get_memory(migrated, confirmed["id"])["claim"] == confirmed["claim"]
        assert external_sources.ExternalSourceRepository(migrated).get(external.record_id)["external_id"] == "existing-message"
        assert continuity.ContinuityRepository(migrated).get_thread(thread["thread_id"])["topic"] == "Jarvis project history"
        assert migrated.conn.execute("SELECT count(*) AS n FROM context_topics").fetchone()["n"] == 0
        assert migrated.conn.execute("SELECT count(*) AS n FROM continuity_thread_topics").fetchone()["n"] == 0
        assert {row["table"] for row in migrated.conn.execute("PRAGMA foreign_key_list(continuity_thread_topics)")} == {
            "continuity_threads", "context_topics"
        }
    finally:
        migrated.close()
