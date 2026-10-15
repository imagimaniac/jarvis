"""Silent background candidate producer wiring in the chat API (Phase 1)."""
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import app
from app.core import storage

TOKEN = "producer-srv-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TOKEN", TOKEN)
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    return TestClient(app, raise_server_exceptions=False)


def _candidate_count(db_path):
    db = storage.JarvisDB(Path(db_path))
    try:
        return db.conn.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE review_state='CANDIDATE'"
        ).fetchone()["n"]
    finally:
        db.close()


def _events(response):
    events = []
    for frame in response.text.strip().split("\n\n"):
        if not frame.strip():
            continue
        name = next(line[6:].strip() for line in frame.splitlines() if line.startswith("event:"))
        data = next(line[5:].strip() for line in frame.splitlines() if line.startswith("data:"))
        events.append((name, data))
    return events


def test_chat_does_not_produce_candidates_from_casual_turn(client, monkeypatch):
    from app.server import recall

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Okay."

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)
    monkeypatch.setattr(recall, "recall", lambda db, query, k, **kwargs: [])
    client.post("/api/chat", json={"message": "Hello Jarvis."}, headers=AUTH)
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 0


def test_chat_producer_disabled_by_default_creates_no_candidates(client, monkeypatch):
    from app.server import recall

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Noted."

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)
    monkeypatch.setattr(recall, "recall", lambda db, query, k, **kwargs: [])
    client.post("/api/chat", json={"message": "I work at Acme."}, headers=AUTH)
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 0


def test_chat_producer_when_enabled_silently_creates_candidates(client, monkeypatch):
    from app.server import recall
    monkeypatch.setenv("JARVIS_MEMORY_CANDIDATES", "1")

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Noted."

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)
    monkeypatch.setattr(recall, "recall", lambda db, query, k, **kwargs: [])
    response = client.post("/api/chat", json={"message": "I work at Acme."}, headers=AUTH)
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 1
    payload = "\n".join(data for _, data in _events(response))
    assert "Acme" not in payload
    assert "CANDIDATE" not in payload
    assert "candidate" not in payload.casefold()


def test_chat_producer_when_enabled_ignores_casual_statements(client, monkeypatch):
    from app.server import recall
    monkeypatch.setenv("JARVIS_MEMORY_CANDIDATES", "1")

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Okay."

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)
    monkeypatch.setattr(recall, "recall", lambda db, query, k, **kwargs: [])
    client.post("/api/chat", json={"message": "The weather is nice today."}, headers=AUTH)
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 0


def test_chat_producer_enabled_still_respects_duplicate_suppression(client, monkeypatch):
    from app.server import recall
    from app.core import memory
    monkeypatch.setenv("JARVIS_MEMORY_CANDIDATES", "1")
    db = storage.JarvisDB(Path(os.environ["JARVIS_DB_PATH"]))
    try:
        source_id = db.insert_entry("Prior", raw_path="prior.txt", raw_relpath="prior.txt")
        candidate = memory.create_candidate(
            db, memory_type="PREFERENCE", claim="I prefer concise answers.",
            source_entry_ids=[source_id], source_attribution="user statement", reason="known",
        )
        memory.confirm(db, candidate["id"], reason="user approved")
    finally:
        db.close()

    class FakeProvider:
        @classmethod
        def from_config(cls, cfg):
            return cls()

        def chat_stream(self, messages):
            yield "Noted."

    monkeypatch.setattr("app.server.OllamaCompatible", FakeProvider)
    monkeypatch.setattr(recall, "recall", lambda db, query, k, **kwargs: [])
    client.post("/api/chat", json={"message": "I prefer concise answers."}, headers=AUTH)
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 0
    assert _candidate_count(os.environ["JARVIS_DB_PATH"]) == 0


def test_config_accessor_honors_env_false_and_yaml_default(tmp_path, monkeypatch):
    import app.core.config as config_mod
    monkeypatch.delenv("JARVIS_MEMORY_CANDIDATES", raising=False)
    assert config_mod.memory_candidates_enabled() is False
    monkeypatch.setenv("JARVIS_MEMORY_CANDIDATES", "true")
    assert config_mod.memory_candidates_enabled() is True
    monkeypatch.setenv("JARVIS_MEMORY_CANDIDATES", "no")
    assert config_mod.memory_candidates_enabled() is False