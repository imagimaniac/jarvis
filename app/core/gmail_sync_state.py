"""Local, credential-free checkpoints for explicit Gmail synchronization."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.core import utils


class GmailFullSyncRequired(RuntimeError):
    """No compatible checkpoint exists, or Gmail expired its history cursor."""


def account_key(email_address: str) -> str:
    value = (email_address or "").strip().casefold()
    if not value or "@" not in value or len(value) > 320:
        raise ValueError("Gmail account identity is unavailable; cannot safely scope sync state")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_key(query: str | None) -> str:
    # Store only a digest: the user's Gmail search expression may reveal interests.
    normalized = (query or "").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_history_id(value: object) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-9]{1,64}", text):
        raise ValueError("Gmail returned an invalid history checkpoint")
    return text


@dataclass(frozen=True)
class GmailSyncCheckpoint:
    account_key: str
    query_key: str
    history_id: str
    updated_at_utc: str


class GmailSyncStateRepository:
    """Read/commit checkpoints, scoped to one hashed account and query identity."""

    def __init__(self, db):
        self.db = db

    def get(self, account: str, query: str | None = None) -> GmailSyncCheckpoint | None:
        akey, qkey = account_key(account), query_key(query)
        row = self.db.conn.execute(
            "SELECT account_key, query_key, history_id, updated_at_utc FROM gmail_sync_state "
            "WHERE account_key=? AND query_key=?", (akey, qkey),
        ).fetchone()
        return GmailSyncCheckpoint(**row) if row else None

    def save_success(self, account: str, query: str | None, history_id: object) -> GmailSyncCheckpoint:
        akey, qkey = account_key(account), query_key(query)
        history = validate_history_id(history_id)
        updated = utils.utc_iso()
        with self.db.conn:
            self.db.conn.execute(
                "INSERT INTO gmail_sync_state(account_key,query_key,history_id,updated_at_utc) VALUES(?,?,?,?) "
                "ON CONFLICT(account_key,query_key) DO UPDATE SET "
                "history_id=excluded.history_id, updated_at_utc=excluded.updated_at_utc",
                (akey, qkey, history, updated),
            )
        return GmailSyncCheckpoint(akey, qkey, history, updated)
