"""Integrity, rebuild-from-raw, and migration guarantees.

These cover the three defects that made the store untrustworthy:
  F1  verify_hashes flagged every clean file as corrupt (trailing-newline drift)
  F2  rebuild_fts ignored its raw_base argument, so raw was not the source of truth
  F7  no schema_version, so migrations could not be applied safely
"""
from pathlib import Path
import hashlib
import shutil
import tempfile

import pytest

from app.core import rebuild, storage, utils


def _sandbox():
    td = Path(tempfile.mkdtemp())
    raw = td / "memory/raw"
    raw.mkdir(parents=True)
    (td / "data").mkdir()
    return td, raw, td / "data/jarvis.db"


def _add(db, raw, text):
    path, rel, ch = utils.write_raw(raw, text)
    return db.insert_entry(text=text, raw_path=path, raw_relpath=rel, content_hash=ch)


def test_writer_and_database_agree_on_hash():
    """F1: a freshly written entry must verify clean."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _add(db, raw, "I applied to Company X for a Business Analytics role")
        db.close()
        assert rebuild.verify_hashes(dbp) == []
    finally:
        shutil.rmtree(td)


def test_hash_includes_the_newline_actually_written():
    """F1: the hash must cover the bytes on disk, not the pre-write text."""
    text = "thinking about memory"
    path, _rel, ch = utils.write_raw(Path(tempfile.mkdtemp()), text)
    assert ch == rebuild.file_hash(path), "hash must equal sha256 of the file bytes"
    assert utils.content_hash(text) == ch, "content_hash must use the same contract"


def test_verify_detects_real_tampering():
    """A checker that never fails is worse than none: prove it can fail."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _add(db, raw, "original thought")
        db.close()
        target = next(raw.rglob("*.txt"))
        target.write_bytes(b"silently rewritten\n")
        issues = rebuild.verify_hashes(dbp)
        assert len(issues) == 1 and issues[0]["issue"] == "hash_mismatch"
    finally:
        shutil.rmtree(td)


def test_rebuild_restores_text_from_raw():
    """F2: raw files, not the database copy, are the source of truth."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _add(db, raw, "the original wording")
        db.conn.execute("UPDATE entries SET text_raw='CORRUPTED'")
        db.conn.commit()
        db.close()

        r = rebuild.rebuild_fts(dbp, raw)
        assert r["restored"] == 1 and r["created"] == 0
        db = storage.JarvisDB(dbp)
        assert db.conn.execute("SELECT text_raw FROM entries").fetchone()["text_raw"] == "the original wording"
        db.close()
    finally:
        shutil.rmtree(td)


def test_rebuild_recovers_from_total_database_loss():
    """F2: losing the DB must not lose the memories."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _add(db, raw, "I applied to Company X")
        _add(db, raw, "interview invitation for an Oslo-based role")
        db.close()
        dbp.unlink()

        r = rebuild.rebuild_fts(dbp, raw)
        assert r["created"] == 2
        db = storage.JarvisDB(dbp)
        assert db.conn.execute("SELECT COUNT(*) c FROM entries").fetchone()["c"] == 2
        assert len(db.search_fts("Company", limit=5)) == 1, "recovered entries must be searchable"
        db.close()
        assert rebuild.verify_hashes(dbp) == []
    finally:
        shutil.rmtree(td)


def test_rebuild_preserves_existing_provenance():
    """A rebuild must not relabel entries whose provenance is already known."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        path, rel, ch = utils.write_raw(raw, "recruiter email body")
        db.insert_entry(
            text="recruiter email body",
            source="gmail",
            origin_kind="external_record",
            origin_id="msg-1",
            raw_path=path,
            raw_relpath=rel,
            content_hash=ch,
        )
        db.close()
        rebuild.rebuild_fts(dbp, raw)
        db = storage.JarvisDB(dbp)
        row = db.conn.execute("SELECT source, origin_kind, origin_id FROM entries").fetchone()
        assert (row["source"], row["origin_kind"], row["origin_id"]) == ("gmail", "external_record", "msg-1")
        db.close()
    finally:
        shutil.rmtree(td)


def test_schema_version_is_stamped():
    """F7: a version must exist so migrations have an anchor."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        assert db.schema_version == storage.SCHEMA_VERSION
        assert db.conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()["value"]
        db.close()
    finally:
        shutil.rmtree(td)


def test_migrations_apply_once_and_are_idempotent(monkeypatch):
    """F7: re-opening must not re-run or corrupt anything."""
    td, raw, dbp = _sandbox()
    try:
        monkeypatch.setattr(storage, "MIGRATIONS", [
            *storage.MIGRATIONS,
            (storage.SCHEMA_VERSION + 1, ["CREATE TABLE IF NOT EXISTS mig_probe (x INTEGER)"]),
        ])
        db = storage.JarvisDB(dbp)
        target = storage.SCHEMA_VERSION + 1
        assert db.schema_version == target
        db.close()

        reopened = storage.JarvisDB(dbp)
        assert reopened.schema_version == target, "migration must not re-apply"
        reopened.close()
    finally:
        shutil.rmtree(td)


# --- rows that are not backed by their own raw file -------------------------
#
# Regression guard for the 61,374-line false-positive report these tools used
# to produce: every external projection carries raw_path="." and every seed row
# points at the seed JSON, and neither is that entry's raw file.


@pytest.mark.parametrize("raw_path,expected", [
    ("", False),
    (".", False),
    (None, False),
    ("data/seed/canonical_personal_memories.json", False),
    ("/tmp/whatever/2026-10-03-001.txt", True),
    ("memory/raw/2026/10/03/2026-10-03-001.txt", True),
])
def test_has_own_raw_file_only_accepts_writer_style_paths(raw_path, expected):
    assert rebuild.has_own_raw_file(raw_path) is expected


def test_external_projection_without_a_raw_file_verifies_clean():
    """External records hash their text with no trailing newline; accept that."""
    td, raw, dbp = _sandbox()
    try:
        text = "Title\nrecruiter emailed about the next step"
        db = storage.JarvisDB(dbp)
        db.insert_entry(
            text=text, source="gmail", origin_kind="external_record",
            origin_id="msg-1", raw_path=".", raw_relpath=".",
            content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )
        db.close()
        assert rebuild.verify_hashes(dbp) == []
    finally:
        shutil.rmtree(td)


def test_seed_record_pointing_at_its_source_file_verifies_clean():
    """A seed row names the JSON it came from; its hash covers its own text."""
    td, raw, dbp = _sandbox()
    try:
        source = td / "seeds" / "canonical.json"
        source.parent.mkdir(parents=True)
        source.write_text('{"records": []}', encoding="utf-8")
        text = "Curated personal seed: user prefers concise answers."
        db = storage.JarvisDB(dbp)
        db.insert_entry(
            text=text, source="canonical_memory_seed", origin_kind="self",
            raw_path=source, raw_relpath="seeds/canonical.json",
            content_hash=utils.content_hash(text),
        )
        db.close()
        assert rebuild.verify_hashes(dbp) == []
    finally:
        shutil.rmtree(td)


def test_rehash_never_rewrites_a_seed_rows_hash_to_the_source_file():
    """The bug this guards: rehash used to adopt the seed JSON's own hash."""
    td, raw, dbp = _sandbox()
    try:
        source = td / "seeds" / "canonical.json"
        source.parent.mkdir(parents=True)
        source.write_text('{"records": []}', encoding="utf-8")
        text = "Curated personal seed: user prefers concise answers."
        correct = utils.content_hash(text)
        db = storage.JarvisDB(dbp)
        db.insert_entry(
            text=text, source="canonical_memory_seed", origin_kind="self",
            raw_path=source, raw_relpath="seeds/canonical.json",
            content_hash=correct,
        )
        db.close()

        report = rebuild.rehash(dbp, apply=True)
        assert report["checked"] == 0, "a non-writer path must not be rehashed"
        db = storage.JarvisDB(dbp)
        stored = db.conn.execute("SELECT content_hash FROM entries").fetchone()["content_hash"]
        db.close()
        assert stored == correct
        assert rebuild.verify_hashes(dbp) == []
    finally:
        shutil.rmtree(td)


def test_verify_reports_a_self_row_whose_text_hash_is_wrong():
    """The in-DB check must still be able to fail."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        db.insert_entry(
            text="some claim", source="cli", origin_kind="self",
            raw_path=".", raw_relpath=".", content_hash="0" * 64,
        )
        db.close()
        issues = rebuild.verify_hashes(dbp)
        assert len(issues) == 1 and issues[0]["issue"] == "text_hash_mismatch"
    finally:
        shutil.rmtree(td)


def test_rebuild_ignores_the_dot_placeholder_row():
    """F2: "." must not collide with every real relative path during rebuild."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        db.insert_entry(text="external evidence", source="gmail",
                        origin_kind="external_record", raw_path=".", raw_relpath=".",
                        content_hash=hashlib.sha256(b"external evidence").hexdigest())
        _add(db, raw, "a real note")
        db.close()

        r = rebuild.rebuild_fts(dbp, raw)
        assert r["created"] == 0 and r["restored"] == 1
        assert r["fts_rows"] == 2, "the external row must stay indexed"
        db = storage.JarvisDB(dbp)
        assert db.conn.execute("SELECT count(*) c FROM entries").fetchone()["c"] == 2
        db.close()
    finally:
        shutil.rmtree(td)
