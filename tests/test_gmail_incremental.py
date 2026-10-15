import base64
import json
import urllib.error
import urllib.error

import pytest

from app.core import external_sources, gmail, gmail_sync_state, memory, storage


def _message(message_id: str, body: str, *, labels=None):
    encoded = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")

    return {
        "id": message_id, "threadId": "thread-" + message_id,
        "labelIds": labels or ["INBOX"], "internalDate": "1790857800000",
        "payload": {"mimeType": "text/plain", "headers": [
            {"name": "From", "value": "sender@example.test"},
            {"name": "Subject", "value": "Incremental test"},
            {"name": "Message-ID", "value": f"<{message_id}@example.test>"},
        ], "body": {"data": encoded}},
    }


class _Client:
    def __init__(self, *, account="owner@example.test", profile_history="100"):
        self.account = account
        self.profile_history = profile_history
        self.full_pages = [{"messages": []}]
        self.history_pages = [{"history": [], "historyId": profile_history}]
        self.message_values = {}
        self.history_calls = []
        self.list_calls = []
        self.fail_ids = set()
        self.query_matches = True

    def get_profile(self):
        return {"emailAddress": self.account, "historyId": self.profile_history}

    def list_messages(self, *, page_token=None, max_results=100, query=None):
        self.list_calls.append({"page_token": page_token, "query": query})
        if query and "rfc822msgid:" in query:
            if not self.query_matches:
                return {"messages": []}
            msgid = query.split("rfc822msgid:", 1)[1]
            wanted = "<" + msgid + ">"
            for mid, message in self.message_values.items():
                headers = message.get("payload", {}).get("headers", [])
                if any(h.get("name", "").casefold() == "message-id" and h.get("value") == wanted for h in headers):
                    return {"messages": [{"id": mid}]}
            return {"messages": []}
        index = 0 if page_token is None else (1 if page_token == "page-2" else int(page_token))
        return self.full_pages[index]

    def get_message(self, message_id):
        if message_id in self.fail_ids:
            raise gmail.GmailAPIError(500)
        value = self.message_values.get(message_id)
        if value is None:
            raise gmail.GmailAPIError(404)
        return value

    def list_history(self, start_history_id, *, page_token=None, max_results=100):
        self.history_calls.append((start_history_id, page_token))
        index = 0 if page_token is None else (1 if page_token == "page-2" else int(page_token))
        return self.history_pages[index]


def _setup(tmp_path):
    db = storage.JarvisDB(tmp_path / "data" / "jarvis.db")
    return db, external_sources.ExternalSourceRepository(db), gmail_sync_state.GmailSyncStateRepository(db)


def test_full_sync_persists_scoped_checkpoint_only_after_full_and_catchup(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        client = _Client(profile_history="100")
        client.full_pages = [{"messages": [{"id": "initial"}]}]
        client.message_values["initial"] = _message("initial", "Full sync evidence")
        client.history_pages = [{"history": [{"messagesAdded": [{"message": {"id": "during-scan"}}]}],
                                "historyId": "105"}]
        client.message_values["during-scan"] = _message("during-scan", "Changed during full sync")
        connector = gmail.GmailConnector(client, query="newer_than:30d")

        summary = gmail.sync_gmail_full(connector, repo, state)
        saved = state.get("owner@example.test", "newer_than:30d")
        assert saved.history_id == "105"
        assert summary["inserted"] == 1 and summary["history_catchup_messages"] == 1
        assert repo.get(external_sources.ExternalSourceRepository.stable_record_id("gmail", "during-scan"))
        raw = dict(db.conn.execute("SELECT * FROM gmail_sync_state").fetchone())
        assert "owner@example.test" not in json.dumps(raw)
        assert "newer_than:30d" not in json.dumps(raw)
        assert "access_token" not in json.dumps(raw)
    finally:
        db.close()


def test_incomplete_or_failed_full_sync_does_not_save_a_checkpoint(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        state.save_success("owner@example.test", None, "90")
        client = _Client(profile_history="100")
        client.full_pages = [{"messages": [], "nextPageToken": "more"}]
        with pytest.raises(RuntimeError, match="no checkpoint"):
            gmail.sync_gmail_full(gmail.GmailConnector(client), repo, state, max_pages=1)
        assert state.get("owner@example.test").history_id == "90"

        class BrokenClient(_Client):
            def list_messages(self, **kwargs):
                raise gmail.GmailAPIError(503)

        with pytest.raises(gmail.GmailAPIError):
            gmail.sync_gmail_full(gmail.GmailConnector(BrokenClient()), repo, state)
        assert state.get("owner@example.test").history_id == "90"
    finally:
        db.close()


def test_incremental_sync_ingests_additions_changes_and_history_pages(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        repo.upsert(gmail.normalize_gmail_message(_message("changed", "Old content")))
        state.save_success("owner@example.test", None, "100")
        client = _Client(profile_history="110")
        client.message_values = {
            "new": _message("new", "Newly received content"),
            "changed": _message("changed", "Updated body", labels=["INBOX", "STARRED"]),
        }
        client.history_pages = [
            {"history": [{"messagesAdded": [{"message": {"id": "new"}}]}],
             "historyId": "105", "nextPageToken": "page-2"},
            {"history": [{"labelsAdded": [{"message": {"id": "changed"}}]}], "historyId": "110"},
        ]
        result = gmail.sync_gmail_incremental(gmail.GmailConnector(client), repo, state)
        assert result["pages"] == 2 and result["messages"] == 2
        assert result["inserted"] == 1 and result["updated"] == 1
        assert state.get("owner@example.test").history_id == "110"
        record_id = external_sources.ExternalSourceRepository.stable_record_id("gmail", "changed")
        assert len(repo.list_versions(record_id)) == 2
        assert client.history_calls == [("100", None), ("100", "page-2")]
        assert memory.list_memories(db, limit=100) == []
    finally:
        db.close()


def test_incremental_deletion_tombstones_current_evidence_but_retains_versions(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        original = repo.upsert(gmail.normalize_gmail_message(_message("deleted", "Delete this searchable phrase")))
        state.save_success("owner@example.test", None, "200")
        client = _Client(profile_history="210")
        client.history_pages = [{"history": [{"messagesDeleted": [{"message": {"id": "deleted"}}]}],
                                "historyId": "210"}]
        result = gmail.sync_gmail_incremental(gmail.GmailConnector(client), repo, state)
        assert result["deleted"] == 1
        assert db.search_fts("searchable phrase", limit=10) == []
        record = repo.get(original.record_id)
        assert record["deleted_at_utc"]
        assert len(repo.list_versions(original.record_id)) == 1
        assert state.get("owner@example.test").history_id == "210"
        assert memory.list_memories(db, limit=100) == []
    finally:
        db.close()


def test_partial_failure_leaves_checkpoint_and_replay_is_safe(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        state.save_success("owner@example.test", None, "300")
        client = _Client(profile_history="310")
        client.history_pages = [{"history": [{"messagesAdded": [{"message": {"id": "a"}},
                                                                    {"message": {"id": "b"}}]}],
                                "historyId": "310"}]
        client.message_values = {"a": _message("a", "Replay safe"), "b": _message("b", "Will fail once")}
        client.fail_ids.add("b")
        connector = gmail.GmailConnector(client)
        with pytest.raises(gmail.GmailAPIError):
            gmail.sync_gmail_incremental(connector, repo, state)
        assert state.get("owner@example.test").history_id == "300"
        first_id = external_sources.ExternalSourceRepository.stable_record_id("gmail", "a")
        assert len(repo.list_versions(first_id)) == 1
        client.fail_ids.clear()
        result = gmail.sync_gmail_incremental(connector, repo, state)
        assert result["unchanged"] == 1 and result["inserted"] == 1
        assert len(repo.list_versions(first_id)) == 1
        assert state.get("owner@example.test").history_id == "310"
    finally:
        db.close()


def test_expired_history_requires_full_sync_and_does_not_advance_checkpoint(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        state.save_success("owner@example.test", None, "400")
        client = _Client()
        client.list_history = lambda *a, **k: (_ for _ in ()).throw(gmail.GmailHistoryExpired(
            "Gmail history checkpoint expired; run a full sync to recover"))
        with pytest.raises(gmail.GmailFullSyncRequired, match="full sync"):
            gmail.sync_gmail_incremental(gmail.GmailConnector(client), repo, state)
        assert state.get("owner@example.test").history_id == "400"
    finally:
        db.close()


@pytest.mark.parametrize("account,query", [("owner@example.test", "newer_than:7d"),
                                           ("other@example.test", None)])
def test_account_and_query_identity_never_reuse_wrong_checkpoint(tmp_path, account, query):
    db, repo, state = _setup(tmp_path)
    try:
        state.save_success("owner@example.test", "newer_than:30d", "500")
        client = _Client(account=account)
        with pytest.raises(gmail_sync_state.GmailFullSyncRequired, match="full sync"):
            gmail.sync_gmail_incremental(gmail.GmailConnector(client, query=query), repo, state)
    finally:
        db.close()


def test_query_scoped_incremental_uses_gmail_query_for_membership_and_soft_removes(tmp_path):
    db, repo, state = _setup(tmp_path)
    try:
        repo.upsert(gmail.normalize_gmail_message(_message("outside", "Previously matched")))
        state.save_success("owner@example.test", "from:team@example.test", "600")
        client = _Client()
        client.message_values["outside"] = _message("outside", "No longer matches query")
        client.query_matches = False
        client.history_pages = [{"history": [{"labelsRemoved": [{"message": {"id": "outside"}}]}],
                                "historyId": "610"}]
        result = gmail.sync_gmail_incremental(
            gmail.GmailConnector(client, query="from:team@example.test"), repo, state
        )
        assert result["deleted"] == 1
        assert any("rfc822msgid:outside@example.test" in (call["query"] or "") for call in client.list_calls)
        assert db.search_fts("Previously matched", limit=10) == []
    finally:
        db.close()


def test_v7_database_migrates_gmail_checkpoint_additively(tmp_path):
    path = tmp_path / "v7.sqlite"
    db = storage.JarvisDB(path)
    try:
        entry = db.insert_entry("Pre-existing evidence survives v8", source="cli")
        record = external_sources.ExternalSourceRepository(db).upsert(
            gmail.normalize_gmail_message(_message("legacy", "Old Gmail evidence"))
        )
        db.conn.execute("ALTER TABLE external_source_records DROP COLUMN deleted_at_utc")
        db.conn.execute("DROP TABLE gmail_sync_state")
        db.conn.execute("UPDATE meta SET value='7' WHERE key='schema_version'")
        db.conn.commit()
    finally:
        db.close()
    migrated = storage.JarvisDB(path)
    try:
        assert migrated.schema_version == 9
        assert migrated.get(entry)["entry"]["text_raw"] == "Pre-existing evidence survives v8"
        assert migrated.search_fts("Old Gmail evidence", limit=10)
        assert migrated.conn.execute("SELECT deleted_at_utc FROM external_source_records WHERE id=?",
                                     (record.record_id,)).fetchone()["deleted_at_utc"] is None
        assert migrated.conn.execute("SELECT COUNT(*) AS n FROM gmail_sync_state").fetchone()["n"] == 0
    finally:
        migrated.close()


def test_history_api_uses_start_checkpoint_and_maps_expiry_safely():
    seen = []

    # Verify exact History API parameters without making network requests.
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"history":[],"historyId":"700"}'

    oauth = type("OAuth", (), {"access_token": lambda self, **kwargs: "opaque-test-token"})()
    client = gmail.GmailAPIClient(oauth, opener=lambda req, timeout: (seen.append(req.full_url) or Response()))
    assert client.list_history("650", page_token="next") == {"history": [], "historyId": "700"}
    assert "startHistoryId=650" in seen[0] and "pageToken=next" in seen[0]

    def gone(*args, **kwargs):
        raise urllib.error.HTTPError("https://gmail.invalid", 404, "gone", {}, None)

    expired = gmail.GmailAPIClient(oauth, opener=gone, retries=0)
    with pytest.raises(gmail.GmailHistoryExpired, match="full sync"):
        expired.list_history("650")
