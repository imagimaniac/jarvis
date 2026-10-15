"""Shared test harness.

Two jobs, both about protecting real user data rather than convenience:

1. Default every test to a throwaway store. A test that forgets to set
   ``JARVIS_DB_PATH`` would otherwise resolve through ``app.core.config``
   straight to the production database, and ``POST /api/add`` would append to
   it silently.

2. Prove after the run that the protected store did not change. This catches
   a write that escaped the defaulting in (1), or any future fixture that
   points back at the real paths.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTECTED_DB = REPO_ROOT / "data" / "jarvis.db"
PROTECTED_RAW = REPO_ROOT / "memory" / "raw"

_STORE_KEYS = ("JARVIS_DB_PATH", "JARVIS_RAW_DIR")


def _fingerprint() -> dict:
    """Cheap identity of the protected store: file metadata, never contents."""
    files = sorted(
        str(p.relative_to(PROTECTED_RAW))
        for p in PROTECTED_RAW.rglob("*")
        if p.is_file()
    ) if PROTECTED_RAW.is_dir() else []
    if PROTECTED_DB.is_file():
        st = PROTECTED_DB.stat()
        db = (st.st_size, st.st_mtime_ns, st.st_ino)
    else:
        db = None
    return {"db": db, "raw_files": files}


def describe_diff(before: dict, after: dict, *, db_path=None, raw_path=None) -> list[str]:
    """Human-readable reasons two fingerprints disagree. Empty means unchanged.

    Kept pure so the guard can be tested without touching real user data.
    Paths default to the module globals at call time (not definition time) so
    the guard can be aimed at a stand-in store in tests.
    """
    db_path = PROTECTED_DB if db_path is None else db_path
    raw_path = PROTECTED_RAW if raw_path is None else raw_path
    problems: list[str] = []
    if before.get("db") != after.get("db"):
        problems.append(
            f"{db_path} changed: {before.get('db')} -> {after.get('db')} "
            "(size, mtime_ns, inode)"
        )
    added = sorted(set(after.get("raw_files", [])) - set(before.get("raw_files", [])))
    removed = sorted(set(before.get("raw_files", [])) - set(after.get("raw_files", [])))
    if added:
        problems.append(f"{', '.join(added[:5])} created under {raw_path}")
    if removed:
        problems.append(f"{', '.join(removed[:5])} removed under {raw_path}")
    return problems


@pytest.fixture(scope="session", autouse=True)
def guard_protected_store():
    """Fail the run if the production DB or raw memory was modified."""
    before = _fingerprint()
    yield before
    after = _fingerprint()
    problems = describe_diff(before, after)
    if problems:
        pytest.fail(
            "PROTECTED USER DATA WAS MODIFIED BY THE TEST SUITE:\n  "
            + "\n  ".join(problems)
            + "\nTests must set JARVIS_DB_PATH/JARVIS_RAW_DIR to a tmp_path.",
            pytrace=False,
        )


def isolate_store_env(monkeypatch, tmp_path) -> None:
    """Point the store at a temp dir when a test did not choose one.

    Deliberately a plain function so it can be unit-tested without invoking a
    fixture directly.
    """
    for key, name in (("JARVIS_DB_PATH", "jarvis.db"), ("JARVIS_RAW_DIR", "raw")):
        if key not in os.environ:
            monkeypatch.setenv(key, str(tmp_path / name))


@pytest.fixture(autouse=True)
def isolate_store(monkeypatch, tmp_path):
    """Runs before module-level fixtures, so a test module that deliberately
    clears the environment (``tests/test_config.py``) still wins."""
    isolate_store_env(monkeypatch, tmp_path)
    return tmp_path
