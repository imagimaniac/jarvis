import shutil
import tempfile
from pathlib import Path

from app.core import epistemic, models, storage


def _sandbox():
    td = tempfile.mkdtemp()
    tdp = Path(td)
    (tdp / "memory/raw").mkdir(parents=True)
    (tdp / "data").mkdir(parents=True)
    return td, tdp / "memory/raw", tdp / "data/jarvis.db"


def _write(db, raw, text, origin_kind):
    """Mirror the real writer path: insert_entry + add_structured."""
    rel = f"{abs(hash(text))}.txt"
    eid = db.insert_entry(text, source="cli", origin_kind=origin_kind,
                          raw_path=raw / rel, raw_relpath=rel)
    db.add_structured(eid, models.Structured(entry_id=eid, category="note"), inferred_by="heuristic")
    return eid


def test_status_derives_from_origin_not_extractor():
    """The whole point: a heuristic cannot promote a record to fact."""
    assert epistemic.entry_status("self") == epistemic.MEMORY
    assert epistemic.entry_status("external_record") == epistemic.RECORDED
    assert epistemic.FACT not in (epistemic.entry_status("self"), epistemic.entry_status("external_record"))


def test_facets_are_never_talked_up_to_memory():
    """Extracted fields are Jarvis's reading, even from your own words."""
    assert epistemic.facet_status("heuristic") == epistemic.INFERENCE
    assert epistemic.facet_status("llm") == epistemic.INFERENCE
    assert epistemic.facet_status("user") == epistemic.MEMORY


def test_low_confidence_reading_is_a_hypothesis():
    assert epistemic.facet_status("heuristic", confidence=0.1) == epistemic.HYPOTHESIS
    assert epistemic.facet_status("heuristic", confidence=0.9) == epistemic.INFERENCE


def test_inference_may_not_be_presented_as_fact():
    assert epistemic.may_present_as(epistemic.INFERENCE, epistemic.FACT) is False
    assert epistemic.may_present_as(epistemic.MEMORY, epistemic.FACT) is True


def test_migration_backfills_status_from_origin_kind():
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        mine = _write(db, raw, "I decided to use SQLite", "self")
        theirs = _write(db, raw, "They decided to use Postgres", "external_record")
        assert db.schema_version == storage.SCHEMA_VERSION
        got = {r["entry_id"]: r["epistemic_status"]
               for r in db.conn.execute("SELECT entry_id, epistemic_status FROM structured")}
        assert got.get(mine) == epistemic.MEMORY
        assert got.get(theirs) == epistemic.RECORDED
        db.close()
    finally:
        shutil.rmtree(td)


def test_search_carries_status_and_keeps_unstructured_entries():
    """LEFT JOIN: an entry with no structured row must still be findable."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        eid = _write(db, raw, "The quarterly retrospective is on Thursday", "self")
        res = db.search_fts("retrospective", limit=5)
        assert len(res) == 1
        assert res[0]["id"] == eid
        assert res[0]["epistemic_status"] == epistemic.MEMORY
        db.conn.execute("DELETE FROM structured WHERE entry_id=?", (eid,))
        db.conn.commit()
        res2 = db.search_fts("retrospective", limit=5)
        assert len(res2) == 1, "entry must survive having no structured row"
        assert res2[0]["epistemic_status"] is None
        db.close()
    finally:
        shutil.rmtree(td)


def test_v1_database_upgrades_in_place():
    """A real pre-epistemic DB migrates and backfills without data loss."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _write(db, raw, "A durable thought worth keeping", "self")
        # Rewind to v1: the only thing v1 lacked is this column.
        db.conn.execute("ALTER TABLE structured DROP COLUMN epistemic_status")
        db.conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        db.conn.commit()
        db.close()

        db2 = storage.JarvisDB(dbp)
        assert db2.schema_version == storage.SCHEMA_VERSION
        row = db2.conn.execute("SELECT epistemic_status, category FROM structured").fetchone()
        assert row["epistemic_status"] == epistemic.MEMORY
        assert row["category"] == "note", "migration must not drop existing columns"
        assert db2.search_fts("durable", limit=1), "indexes must survive the upgrade"
        db2.close()
    finally:
        shutil.rmtree(td)


def test_migration_is_idempotent_when_replayed():
    """A half-applied v2 (column present, backfill missing) self-heals."""
    td, raw, dbp = _sandbox()
    try:
        db = storage.JarvisDB(dbp)
        _write(db, raw, "Partial migration probe", "external_record")
        db.conn.execute("UPDATE structured SET epistemic_status=NULL")
        db.conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
        db.conn.commit()
        db.close()

        db2 = storage.JarvisDB(dbp)
        row = db2.conn.execute("SELECT epistemic_status FROM structured").fetchone()
        assert row["epistemic_status"] == epistemic.RECORDED
        db2.close()
    finally:
        shutil.rmtree(td)