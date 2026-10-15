"""Utils: IDs, timestamps, hashing, paths."""
from __future__ import annotations
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_iso(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')

def local_now() -> datetime:
    # naive local; tz stored separately
    return datetime.now()

def new_uuid() -> str:
    return str(uuid.uuid4())

def raw_bytes(text: str) -> bytes:
    """Canonical on-disk representation of an entry's raw text.

    Every writer and every hash must go through this. Keeping one definition is
    what makes verify_hashes() agree with the database instead of reporting
    every file as corrupted.
    """
    return (str(text) + "\n").encode("utf-8", errors="replace")

def content_hash(text: str) -> str:
    """SHA256 of the exact bytes the raw file will contain."""
    return hashlib.sha256(raw_bytes(text)).hexdigest()

def content_hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def day_path(base_raw: Path, dt: datetime | None = None) -> Path:
    dt = dt or utc_now()
    d = dt.astimezone(timezone.utc)
    p = base_raw / f"{d.year:04d}" / f"{d.month:02d}" / f"{d.day:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def next_entry_filename(daydir: Path, dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    d = dt.astimezone(timezone.utc)
    # scan for existing YYYY-MM-DD-XXX
    prefix = f"{d.year:04d}-{d.month:02d}-{d.day:02d}-"
    nums = []
    for f in daydir.glob(f"{prefix}*.txt"):
        try:
            num = int(f.name[len(prefix):-4])
            nums.append(num)
        except ValueError:
            pass
    n = max(nums) + 1 if nums else 1
    return f"{prefix}{n:03d}.txt"

def raw_relpath_from_abs(abs_path: Path, base_raw: Path) -> str:
    try:
        return str(abs_path.relative_to(base_raw))
    except Exception:
        return str(abs_path)

def write_raw(base_raw: Path, text: str, dt: datetime | None = None) -> Tuple[Path, str, str]:
    """Append-only raw write. Returns (path, relpath, content_hash).

    The single place raw files are created. Callers must not write raw files
    directly, or the hash contract drifts again.
    """
    dt = dt or utc_now()
    dpath = day_path(base_raw, dt)
    fpath = dpath / next_entry_filename(dpath, dt)
    data = raw_bytes(text)
    fpath.write_bytes(data)
    return fpath, raw_relpath_from_abs(fpath, base_raw), content_hash_bytes(data)
