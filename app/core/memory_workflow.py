"""Explicit internal decisions for canonical personal-memory records.

This module is deliberately a small policy layer over :mod:`memory`. It never
creates a second record when a candidate is confirmed and leaves all auditing,
provenance, and state changes to the canonical memory service.
"""
from __future__ import annotations

from typing import Any

from app.core import memory


def confirm_candidate(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict:
    """Confirm a candidate in place, preserving its ID and source links."""
    item = memory.get_memory(db, memory_id)
    if item["lifecycle_status"] != "ACTIVE" or item["review_state"] != "CANDIDATE":
        raise memory.MemoryTransitionError("only an active candidate can be confirmed")
    return memory.confirm(db, memory_id, reason=reason, reviewer=reviewer)


def reject_candidate(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict:
    """Reject an active candidate while retaining its provenance and history."""
    item = memory.get_memory(db, memory_id)
    if item["lifecycle_status"] != "ACTIVE" or item["review_state"] != "CANDIDATE":
        raise memory.MemoryTransitionError("only an active candidate can be rejected")
    return memory.reject(db, memory_id, reason=reason, reviewer=reviewer)


def defer_candidate(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict:
    """Record a judge DEFER decision without altering the candidate's state."""
    item = memory.get_memory(db, memory_id)
    if item["lifecycle_status"] != "ACTIVE" or item["review_state"] != "CANDIDATE":
        raise memory.MemoryTransitionError("only an active candidate can be deferred")
    return memory.defer(db, memory_id, reason=reason, reviewer=reviewer)


def update_confirmed_memory(db, memory_id: str, *, reason: str, reviewer: str | None = None,
                            **changes: Any) -> dict:
    """Update an active confirmed memory, recording its full before/after state."""
    item = memory.get_memory(db, memory_id)
    if item["lifecycle_status"] != "ACTIVE" or item["review_state"] != "CONFIRMED":
        raise memory.MemoryTransitionError("only an active confirmed memory can be updated")
    return memory.update_memory(db, memory_id, reason=reason, reviewer=reviewer, **changes)


def supersede_memory(db, memory_id: str, replacement_memory_id: str, *, reason: str,
                     reviewer: str | None = None) -> dict:
    """Supersede one active confirmed memory with another confirmed memory."""
    return memory.supersede(db, memory_id, replacement_memory_id, reason=reason, reviewer=reviewer)


def forget_memory(db, memory_id: str, *, reason: str, reviewer: str | None = None) -> dict:
    """Forget an active confirmed memory without deleting source evidence."""
    item = memory.get_memory(db, memory_id)
    if item["lifecycle_status"] != "ACTIVE" or item["review_state"] != "CONFIRMED":
        raise memory.MemoryTransitionError("only an active confirmed memory can be forgotten")
    return memory.forget(db, memory_id, reason=reason, reviewer=reviewer)