from pathlib import Path
import tempfile
import os

from app.core import storage, utils, extract

def test_add_and_search():
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        raw = tdp / "memory/raw"
        dbp = tdp / "data/jarvis.db"
        raw.mkdir(parents=True)
        dbp.parent.mkdir(parents=True)
        db = storage.JarvisDB(dbp)
        eid = db.insert_entry("I was thinking about AI and personal memory today", source="cli", origin_kind="self", raw_path=raw/"t.txt", raw_relpath="t.txt")
        assert eid > 0
        res = db.search_fts("AI memory", limit=5)
        assert len(res) >= 1
        db.close()

def test_raw_append_only():
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "memory/raw"
        d = utils.day_path(raw)
        fn = utils.next_entry_filename(d)
        (d / fn).write_text("hello\n", encoding='utf-8')
        fn2 = utils.next_entry_filename(d)
        assert fn2 != fn, "next filename must not collide with an existing entry"
        day1, seq1 = fn[:-4].rsplit("-", 1)
        day2, seq2 = fn2[:-4].rsplit("-", 1)
        assert day1 == day2, "sequence must stay within the same day"
        assert int(seq2) == int(seq1) + 1, "sequence must increment by one"
