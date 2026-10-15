"""Rebuild utilities: reconstruct state from raw files + integrity checks."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any, Dict, List

from . import storage, utils

# Rows that are not backed by their own raw file. External source projections
# keep their text in SQLite and carry "." as a placeholder in raw_path/raw_relpath;
# treating those as "missing" buries the real mismatches under sixty thousand
# rows. Seed records point at the source file they came from, not a per-entry
# raw file, so they are checked against their own text instead.
NO_RAW_FILE = {"", ".", None}


def has_own_raw_file(raw_path: Any) -> bool:
    """True only for rows write_raw() could have produced.

    ``write_raw()`` is the single raw-file writer and always creates
    ``YYYY-MM-DD-NNN.txt``. Everything else in raw_path is provenance (a seed
    JSON, a connector source) and must never be hashed as if it were the
    entry's bytes -- doing so rewrites a correct content_hash into a hash of an
    unrelated file.
    """
    if raw_path in NO_RAW_FILE:
        return False
    return str(raw_path).endswith(".txt")


def _text_hash(text: str) -> str:
    """utils.content_hash(): SHA256 of the exact bytes write_raw() would store."""
    return utils.content_hash(text)


def _text_hash_no_newline(text: str) -> str:
    """Convention used by external_source projections, which write no file."""
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def _expected_hashes(text: str) -> set[str]:
    """Both accepted in-DB hash conventions for a file-less row."""
    return {_text_hash(text), _text_hash_no_newline(text)}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if p.is_dir():
        return ""
    with open(p, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def _canonical(path: Path) -> str:
    """Text an entry should hold for this raw file: contents minus the one
    trailing newline write_raw() appends."""
    text = path.read_bytes().decode("utf-8", errors="replace")
    return text[:-1] if text.endswith("\n") else text


def rebuild_fts(db_path: Path, raw_base: Path) -> Dict[str, Any]:
    """Rebuild entries' text and the FTS5 index from the raw files on disk.

    Provenance already in the database is preserved: rows are matched on
    raw_relpath and only their text is refreshed. Rows with no matching file are
    left alone. So this both repairs a damaged index and reconstructs a lost
    database from raw alone.
    """
    db = storage.JarvisDB(db_path)
    cur = db.conn.cursor()

    cur.execute("SELECT id, raw_relpath FROM entries")
    existing = {r["raw_relpath"]: r["id"] for r in cur.fetchall()
                if r["raw_relpath"] not in NO_RAW_FILE}

    restored = 0
    created = 0
    raw_base = Path(raw_base)
    for p in sorted(raw_base.rglob("*.txt")):
        rel = str(p.relative_to(raw_base))
        text = _canonical(p)
        ch = utils.content_hash(text)
        size = len(utils.raw_bytes(text))
        if rel in existing:
            cur.execute(
                "UPDATE entries SET text_raw=?, content_hash=?, size_bytes=? WHERE id=?",
                (text, ch, size, existing[rel]),
            )
            restored += 1
        else:
            db.insert_entry(
                text,
                source="rebuild",
                origin_kind="self",
                raw_path=p,
                raw_relpath=rel,
                content_hash=ch,
            )
            created += 1
        existing.pop(rel, None)

    # FTS mirrors entries.text_raw for every row, file-backed or not.
    cur.execute("DELETE FROM entries_fts")
    cur.execute("SELECT id, text_raw FROM entries")
    rows = cur.fetchall()
    for r in rows:
        cur.execute("INSERT INTO entries_fts (entry_id, text_raw) VALUES (?,?)", (r["id"], r["text_raw"]))

    db.conn.commit()
    db.close()
    return {
        "restored": restored,
        "created": created,
        "unbacked": len(existing),
        "fts_rows": len(rows),
    }


def rehash(db_path: Path, apply: bool = False) -> Dict[str, Any]:
    """Recompute content_hash from the raw files for rows whose hash is stale.

    Pre-existing rows were hashed without the trailing newline that write_raw()
    appends, so their stored hash disagrees with the file on disk. Only the
    hash column is touched; raw files are never modified.

    Rows without their own raw file (external projections, seed records) are
    left alone: there is nothing on disk to recompute from.
    """
    db = storage.JarvisDB(db_path)
    cur = db.conn.cursor()
    cur.execute("SELECT id, raw_path, content_hash FROM entries ORDER BY id")
    stale = []
    checked = 0
    for r in cur.fetchall():
        if not has_own_raw_file(r["raw_path"]):
            continue
        p = Path(r["raw_path"])
        if not p.is_file():
            stale.append({"id": r["id"], "issue": "missing_raw", "path": str(p)})
            continue
        checked += 1
        correct = file_hash(p)
        if correct != r["content_hash"]:
            stale.append({"id": r["id"], "issue": "stale_hash", "from": r["content_hash"], "to": correct})
    if apply:
        for s in stale:
            if s["issue"] == "stale_hash":
                cur.execute("UPDATE entries SET content_hash=? WHERE id=?", (s["to"], s["id"]))
        db.conn.commit()
    db.close()
    return {"stale": stale, "applied": apply, "count": len(stale), "checked": checked}


def verify_hashes(db_path: Path) -> List[dict]:
    """Check every row's content_hash against the bytes it is supposed to name.

    * row backed by its own raw file -> SHA256 of that file
    * row with no raw file -> SHA256 of its own text (either accepted
      convention: self/seed rows store the newline-terminated bytes that
      write_raw() would produce; external projections store the text as-is)

    Returns one dict per problem row, empty when everything agrees.
    """
    db = storage.JarvisDB(db_path)
    cur = db.conn.cursor()
    cur.execute("SELECT id, raw_path, content_hash, text_raw FROM entries")
    rows = cur.fetchall()
    mismatches = []
    for r in rows:
        rp = r["raw_path"]
        if not has_own_raw_file(rp):
            if r["content_hash"] not in _expected_hashes(r["text_raw"] or ""):
                mismatches.append({
                    "id": r["id"], "issue": "text_hash_mismatch",
                    "path": "<no raw file>",
                })
            continue
        p = Path(rp)
        if not p.exists() or p.is_dir():
            mismatches.append({'id': r['id'], 'issue': 'missing_or_dir', 'path': str(p)})
            continue
        h = file_hash(p)
        if h != r["content_hash"]:
            mismatches.append({'id': r['id'], 'issue': 'hash_mismatch', 'path': str(p)})
    db.close()
    return mismatches