from pathlib import Path
import tempfile
from app.core import storage

def test_self_boost_present():
    # basic sanity: search returns results with source/origin_kind
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        raw = tdp / "memory/raw"; dbp = tdp / "data/jarvis.db"
        raw.mkdir(parents=True); dbp.parent.mkdir(parents=True)
        db = storage.JarvisDB(dbp)
        db.insert_entry("I applied to Company X", source="cli", origin_kind="self", raw_path=raw/"a.txt", raw_relpath="a.txt")
        db.insert_entry("Company X invited me to interview", source="gmail", origin_kind="external_record", raw_path=raw/"b.txt", raw_relpath="b.txt")
        res = db.search_fts("Company X", limit=5)
        assert len(res) == 2
        kinds = set(r['origin_kind'] for r in res)
        assert 'self' in kinds and 'external_record' in kinds
        db.close()
