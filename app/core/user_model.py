"""Evolving User Model: a read-only projection over confirmed canonical memories.

This module never writes to the database and never mutates canonical memory
records. It projects confirmed, active, currently valid memories into a
structured, inspectable, deterministic user model. Unconfirmed candidates are
surfaced only as INFERRED pending suggestions, so an inference can never
present itself as a confirmed fact about the user. A single unconfirmed turn
cannot redefine the model.

Durable sections mirror the canonical memory types; ``patterns`` and
``development`` are deterministic derivations with multiple provenance ids:
a pattern requires at least two consistent confirmed observations, and a
development entry only appears when the canonical lifecycle has superseded one
confirmed memory with another. Consolidation, sets, and conflicts are left
exactly as the confirmed records state them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from . import memory
from .memory_candidates import _tokens


QUALITIES = ("CONFIRMED", "INFERRED", "OBSERVED_PATTERN", "TEMPORARY")
STAGES = ("EMERGING", "ESTABLISHED")
STAGE_THRESHOLD = 3

MAX_SECTION = 200
MAX_TOTAL_CONFIRMED = 300
MAX_PATTERNS = 10
MAX_DEVELOPMENT = 20
MIN_PATTERN_MEMBERS = 2
CLUSTER_MIN_SHARED = 2
_FOREVER = "9999-12-31T23:59:59Z"

_SECTION_BY_TYPE = {
    "FACT": "facts",
    "GOAL": "goals",
    "PROJECT": "projects",
    "PREFERENCE": "preferences",
    "DECISION": "decisions",
    "LEARNING": "learning",
    "HISTORY": "history",
    "HABIT": "habits",
}
DURABLE_SECTIONS: tuple = tuple(_SECTION_BY_TYPE.values())
ALL_SECTIONS: tuple = DURABLE_SECTIONS + ("patterns", "development", "temporary", "pending")


class UserModelError(ValueError):
    """A user model input or boundary is invalid."""


@dataclass(frozen=True)
class Facet:
    """One projected, provenance-carrying statement from the canonical store."""

    section: str
    claim: str
    memory_id: str
    quality: str
    updated_at_utc: str
    created_at_utc: str
    confidence: float
    importance: int
    source_entry_ids: tuple[int, ...]
    related_memory_ids: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "section": self.section,
            "claim": self.claim,
            "memory_id": self.memory_id,
            "quality": self.quality,
            "updated_at_utc": self.updated_at_utc,
            "created_at_utc": self.created_at_utc,
            "confidence": self.confidence,
            "importance": self.importance,
            "source_entry_ids": list(self.source_entry_ids),
            "related_memory_ids": list(self.related_memory_ids),
        }


@dataclass(frozen=True)
class UserModel:
    """Structured, deterministic, read-only projection of the user."""

    sections: Dict[str, List[Facet]]
    quality_counts: Dict[str, int]
    defined: bool
    stage: str
    confirmed_count: int
    pending_count: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sections": {
                section: [facet.as_dict() for facet in self.sections[section]]
                for section in ALL_SECTIONS
            },
            "quality_counts": dict(self.quality_counts),
            "defined": self.defined,
            "stage": self.stage,
            "confirmed_count": self.confirmed_count,
            "pending_count": self.pending_count,
        }


def _normalize_now(now: Any = None) -> str:
    """Return ``now`` as a UTC "Z"-suffixed ISO-8601 string for validity checks."""
    if now is None:
        return _utc_now()
    if isinstance(now, datetime):
        if now.tzinfo is None or now.utcoffset() is None:
            raise UserModelError("now must be timezone-aware")
        return _to_z(now)
    if not isinstance(now, str):
        raise UserModelError("now must be None, ISO-8601 text, or an aware datetime")
    try:
        parsed = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise UserModelError("now must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise UserModelError("now must include a timezone")
    return _to_z(parsed)


def _utc_now() -> str:
    return _to_z(datetime.now(timezone.utc))


def _to_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _valid_iso(value: str) -> str:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise UserModelError("memory validity timestamps must be ISO-8601") from None
    return value


def _valid_at(item: Dict[str, Any], at: str) -> bool:
    valid_from = _valid_iso(item["valid_from_utc"])
    valid_until = item.get("valid_until_utc") or _FOREVER
    if not isinstance(valid_until, str):
        raise UserModelError("valid_until_utc must be text or None")
    valid_until = _valid_iso(valid_until)
    return valid_from <= at <= valid_until


def _iter_confirmed(db, cap: int = MAX_TOTAL_CONFIRMED) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    while len(out) < cap:
        page = memory.list_memories(db, review_state="CONFIRMED", limit=100, offset=offset)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        if len(page) < 100 or offset >= cap:
            break
    return out[:cap]


def _iter_pending(db, cap: int = MAX_TOTAL_CONFIRMED) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    offset = 0
    while len(out) < cap:
        page = memory.list_memories(db, review_state="CANDIDATE", lifecycle_status="ACTIVE",
                                    limit=100, offset=offset)
        if not page:
            break
        out.extend(page)
        offset += len(page)
        if len(page) < 100 or offset >= cap:
            break
    return out[:cap]


def _facet(section: str, item: Dict[str, Any], quality: str,
           related: Iterable[str] = ()) -> Facet:
    return Facet(
        section=section,
        claim=item["claim"],
        memory_id=item["id"],
        quality=quality,
        updated_at_utc=item["updated_at_utc"],
        created_at_utc=item["created_at_utc"],
        confidence=float(item["confidence"]),
        importance=int(item["importance"]),
        source_entry_ids=tuple(sorted(source["entry_id"] for source in item.get("sources") or [])),
        related_memory_ids=tuple(related),
    )


def _surface_order(item: Dict[str, Any]):
    return (item["updated_at_utc"], item["created_at_utc"], item["id"])


def _derive_patterns(habit_items: List[Dict[str, Any]]) -> List[Facet]:
    clusters: List[Dict[str, Any]] = []
    for item in sorted(habit_items, key=_surface_order):
        tokens = set(_tokens(item["claim"]))
        for cluster in clusters:
            if len(tokens & cluster["tokens"]) >= CLUSTER_MIN_SHARED:
                cluster["items"].append(item)
                cluster["tokens"] |= tokens
                break
        else:
            clusters.append({"tokens": tokens, "items": [item]})

    patterns: List[Facet] = []
    for cluster in clusters:
        members = sorted(cluster["items"], key=lambda i: (i["created_at_utc"], i["id"]))
        if len(members) < MIN_PATTERN_MEMBERS:
            continue
        claims = list(dict.fromkeys(item["claim"] for item in members))
        sentence = "; ".join(claims)[:200]
        anchor = members[-1]
        patterns.append(Facet(
            section="patterns",
            claim=f"Observed {len(members)} times: {sentence}",
            memory_id=anchor["id"],
            quality="OBSERVED_PATTERN",
            updated_at_utc=anchor["updated_at_utc"],
            created_at_utc=anchor["created_at_utc"],
            confidence=min(float(item["confidence"]) for item in members),
            importance=max(int(item["importance"]) for item in members),
            source_entry_ids=tuple(sorted(
                source["entry_id"] for item in members for source in item.get("sources") or []
            )),
            related_memory_ids=tuple(item["id"] for item in members),
        ))
        if len(patterns) >= MAX_PATTERNS:
            break
    return patterns


def _derive_development(superseded_items: List[Dict[str, Any]],
                        by_id: Dict[str, Dict[str, Any]]) -> List[Facet]:
    changes: List[Facet] = []
    for old in sorted(superseded_items, key=_surface_order):
        replacement = by_id.get(old["superseded_by_memory_id"])
        if replacement is None or replacement["lifecycle_status"] != "ACTIVE":
            continue
        changes.append(Facet(
            section="development",
            claim=f"{old['claim']} → {replacement['claim']}",
            memory_id=old["id"],
            quality="CONFIRMED",
            updated_at_utc=replacement["updated_at_utc"],
            created_at_utc=replacement["created_at_utc"],
            confidence=min(float(old["confidence"]), float(replacement["confidence"])),
            importance=max(int(old["importance"]), int(replacement["importance"])),
            source_entry_ids=tuple(sorted(
                source["entry_id"] for item in (old, replacement)
                for source in item.get("sources") or []
            )),
            related_memory_ids=(old["id"], replacement["id"]),
        ))
        if len(changes) >= MAX_DEVELOPMENT:
            break
    return changes


def build_user_model(db, *, now: Any = None, include_sensitive: bool = False,
                     include_pending: bool = False,
                     limits: Dict[str, int] | None = None) -> UserModel:
    """Build the deterministic user model projection. Pure read; never writes."""
    at = _normalize_now(now)
    section_limits = {section: MAX_SECTION for section in DURABLE_SECTIONS}
    if isinstance(limits, dict):
        for section, value in limits.items():
            if section not in ALL_SECTIONS:
                raise UserModelError(f"unknown user model section: {section}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UserModelError("user model limits must be non-negative integers")
            section_limits[section] = value

    sections: Dict[str, List[Facet]] = {section: [] for section in ALL_SECTIONS}
    habit_items: List[Dict[str, Any]] = []
    superseded_items: List[Dict[str, Any]] = []

    by_id: Dict[str, Dict[str, Any]] = {}
    confirmed_count = 0
    for item in _iter_confirmed(db):
        by_id[item["id"]] = item
        if item["lifecycle_status"] != "ACTIVE":
            if item["lifecycle_status"] == "SUPERSEDED":
                superseded_items.append(item)
            continue
        if not _valid_at(item, at):
            continue
        confirmed_count += 1
        if item["is_sensitive"] and not include_sensitive:
            continue
        kind = item["memory_type"]
        if kind == "TEMPORARY":
            if len(sections["temporary"]) < section_limits.get("temporary", MAX_SECTION):
                sections["temporary"].append(_facet("temporary", item, "TEMPORARY"))
            continue
        section = _SECTION_BY_TYPE.get(kind)
        if section is None:
            continue
        if kind == "HABIT":
            habit_items.append(item)
        if len(sections[section]) < section_limits.get(section, MAX_SECTION):
            sections[section].append(_facet(section, item, "CONFIRMED"))

    pending_count = 0
    for item in _iter_pending(db):
        if not _valid_at(item, at):
            continue
        pending_count += 1
        if include_pending and len(sections["pending"]) < section_limits.get("pending", MAX_SECTION):
            sections["pending"].append(_facet("pending", item, "INFERRED"))

    sections["patterns"] = _derive_patterns(habit_items)
    sections["development"] = _derive_development(superseded_items, by_id)

    for section in DURABLE_SECTIONS:
        sections[section] = sorted(sections[section], key=lambda f: (f.created_at_utc, f.updated_at_utc, f.memory_id))

    quality_counts = {quality: 0 for quality in QUALITIES}
    for section in DURABLE_SECTIONS + ("temporary", "patterns", "development"):
        for facet in sections[section]:
            quality_counts[facet.quality] = quality_counts.get(facet.quality, 0) + 1

    defined = confirmed_count > 0
    stage = "ESTABLISHED" if confirmed_count >= STAGE_THRESHOLD else "EMERGING"
    return UserModel(
        sections=sections,
        quality_counts=quality_counts,
        defined=defined,
        stage=stage,
        confirmed_count=confirmed_count,
        pending_count=pending_count,
    )


_RENDER_ORDER = ("facts", "preferences", "goals", "projects", "learning",
                 "decisions", "habits", "history")
_RENDER_LABELS = {
    "facts": "Facts",
    "preferences": "Preferences",
    "goals": "Goals",
    "projects": "Projects",
    "learning": "How they learn",
    "decisions": "Decisions",
    "habits": "Habits",
    "history": "History",
}
_MAX_OVERVIEW_PER_SECTION = 12
_MAX_OVERVIEW_CHARS = 2400
_UNCERTAIN_CONFIDENCE = 0.65


def _render_claim(facet: Facet) -> str:
    claim = " ".join(str(facet.claim).split())
    if facet.confidence < _UNCERTAIN_CONFIDENCE:
        return f"Possibly: {claim}"
    return claim


def render_projection(model: UserModel, *, max_chars: int = _MAX_OVERVIEW_CHARS,
                      max_per_section: int = _MAX_OVERVIEW_PER_SECTION) -> str:
    """Render a bounded, human-readable overview of the confirmed user model.

    Deterministic and prompt-safe: only claims are emitted; memory ids,
    provenance, timestamps, and lifecycle labels are omitted, matching the
    personal-context wording contract. Sections are ordered by usefulness and
    each is bounded, highest-importance facets first. Returns "" when the model
    is not defined."""
    if not model.defined:
        return ""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 64:
        raise UserModelError("max_chars must be an integer of at least 64")
    if isinstance(max_per_section, bool) or not isinstance(max_per_section, int) or max_per_section < 1:
        raise UserModelError("max_per_section must be a positive integer")

    blocks: List[str] = []
    for section in _RENDER_ORDER:
        facets = model.sections.get(section) or []
        if not facets:
            continue
        ordered = sorted(
            facets,
            key=lambda f: (f.importance, f.updated_at_utc, f.confidence, f.memory_id),
            reverse=True,
        )[:max_per_section]
        lines = [f"{_RENDER_LABELS.get(section, section.title())}:"]
        lines.extend(f"- {_render_claim(facet)}" for facet in ordered)
        blocks.append("\n".join(lines))
    if not blocks:
        return ""
    text = "\n".join(blocks)
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rsplit("\n", 1)[0].rstrip()
    if not clipped:
        return ""
    return clipped