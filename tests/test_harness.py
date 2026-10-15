"""The test harness itself: store isolation and the protected-data guard.

These tests exist because a regression here would let the suite write into the
real 1 GB production database without anybody noticing.
"""
from __future__ import annotations

import pytest

from app.core import config
from tests import conftest


def test_repo_root_is_where_the_project_lives():
    assert (conftest.REPO_ROOT / "app").is_dir()
    assert (conftest.REPO_ROOT / "pyproject.toml").is_file()


def test_guard_points_at_the_real_protected_paths():
    """If someone repoints these at a temp dir the guard becomes a no-op."""
    assert conftest.PROTECTED_DB == conftest.REPO_ROOT / "data" / "jarvis.db"
    assert conftest.PROTECTED_RAW == conftest.REPO_ROOT / "memory" / "raw"


def test_isolate_store_defaults_the_database_to_the_test_tmp_path(tmp_path):
    """A test that forgets JARVIS_DB_PATH must not reach production."""
    assert config.db_path() == tmp_path / "jarvis.db"
    assert config.raw_dir() == tmp_path / "raw"


def test_isolate_store_does_not_touch_repo_paths(tmp_path):
    assert conftest.PROTECTED_DB not in (config.db_path(), config.raw_dir())
    assert str(tmp_path) in str(config.db_path())


def test_describe_diff_reports_nothing_for_an_untouched_store(tmp_path):
    fp = conftest._fingerprint()
    assert conftest.describe_diff(fp, dict(fp)) == []


def test_describe_diff_detects_a_database_write(tmp_path):
    db = tmp_path / "jarvis.db"
    before = {"db": (100, 1, 7), "raw_files": []}
    after = {"db": (250, 1, 7), "raw_files": []}
    problems = conftest.describe_diff(before, after, db_path=db, raw_path=tmp_path)
    assert len(problems) == 1
    assert str(db) in problems[0]
    assert "(size, mtime_ns, inode)" in problems[0]


def test_describe_diff_detects_raw_files_created_and_removed(tmp_path):
    before = {"db": None, "raw_files": ["2026/10/07/2026-10-07-001.txt"]}
    after = {"db": None, "raw_files": ["2026/10/07/2026-10-07-002.txt"]}
    problems = conftest.describe_diff(before, after, db_path=tmp_path / "x",
                                       raw_path=tmp_path)
    assert len(problems) == 2
    assert "2026-10-07-002.txt created" in problems[0]
    assert "2026-10-07-001.txt removed" in problems[1]


def test_guard_fires_when_the_fingerprint_changes(tmp_path, monkeypatch):
    """End-to-end check of the fixture wiring, against a stand-in store."""
    db = tmp_path / "data" / "jarvis.db"
    raw = tmp_path / "memory" / "raw"
    db.parent.mkdir(parents=True)
    raw.mkdir(parents=True)

    monkeypatch.setattr(conftest, "PROTECTED_DB", db)
    monkeypatch.setattr(conftest, "PROTECTED_RAW", raw)
    before = conftest._fingerprint()
    db.write_bytes(b"new bytes")
    (raw / "2026-10-07-001.txt").write_text("captured\n", encoding="utf-8")
    after = conftest._fingerprint()

    problems = conftest.describe_diff(before, after, db_path=db, raw_path=raw)
    assert len(problems) == 2

    # And the guard must stay silent when nothing happened.
    monkeypatch.setattr(conftest, "PROTECTED_RAW", tmp_path / "memory" / "empty")
    unchanged = conftest.describe_diff(before, before, db_path=db,
                                       raw_path=tmp_path / "memory" / "empty")
    assert unchanged == []


@pytest.mark.parametrize("key", ["JARVIS_DB_PATH", "JARVIS_RAW_DIR"])
def test_isolate_store_leaves_explicit_test_configuration_alone(key, monkeypatch,
                                                                tmp_path):
    """A test that chose its own store keeps it (isolate_store runs first)."""
    import os

    chosen = tmp_path / "chosen.db"
    monkeypatch.setenv(key, str(chosen))
    conftest.isolate_store_env(monkeypatch, tmp_path)
    assert os.environ[key] == str(chosen)


def test_isolate_store_env_fills_in_missing_keys(monkeypatch, tmp_path):
    import os

    monkeypatch.delenv("JARVIS_DB_PATH", raising=False)
    monkeypatch.delenv("JARVIS_RAW_DIR", raising=False)
    conftest.isolate_store_env(monkeypatch, tmp_path)
    assert os.environ["JARVIS_DB_PATH"] == str(tmp_path / "jarvis.db")
    assert os.environ["JARVIS_RAW_DIR"] == str(tmp_path / "raw")
