"""Deterministic Context Engine (Phase 4).

Cross-ratings the lanes the server already builds into one bounded, read-only
package of background context:

* a conservative evidence pass filter. Source evidence is preserved
  byte-for-byte: an entry already surfaced by ``recall.recall`` is dropped only
  when it shares no keyword (or synonym) with the current message and is not
  HIGH importance, and the filter is skipped entirely for external-only
  retrieval where sender/date constraints already decided relevance;
* a reranked, explainable memory lane composed entirely from upstream
  primitives: ``memory.retrieve`` (active/confirmed/validity), the
  ``personal_context`` tokenizer, correction suppression, wording contract and
  recency/type weights, ``user_model.build_user_model`` for the stage, and the
  SUPERSEDED bookkeeping for recently replaced memories.

The engine is deterministic and read-only: given the same store and the same
``now``, ``assemble`` returns identical output, and it never writes to the
database.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from app.core import memory, recall, user_model
from app.core.personal_context import (
    CONTEXT_GUIDANCE,
    CONTEXT_GUIDANCE_PERSONAL,
    TYPE_WEIGHT,
    format_claim,
    is_overridden,
    recency_bonus,
    tokens,
)

MAX_PROMPT_CHARS = 9000
MAX_EVIDENCE_ITEMS = 5
MAX_MEMORY_ITEMS = 5
MAX_MEMORY_CHARS = 1800
MAX_PERSONAL_CHARS = 3800
MIN_MEMORY_COVERAGE = 0.15

_STRONG_CONFIDENCE = 0.70
_MEDIUM_CONFIDENCE = 0.50
_WEAK_CONFIDENCE = 0.40

EMERGING_SHIFT = 0.02
DEVELOPMENT_BUMP = 0.02

_TEMPORARY_PENALTY = 0.10
_STALE_HISTORY_DAYS = 1095  # three years
_STALE_HISTORY_PENALTY = 0.25

_MEMORY_HEADER = "Relevant personal context (background only):"
_BROAD_MEMORY_HEADER = "Additional relevant personal records:"
_OVERVIEW_HEADER = (
    "Personal overview (confirmed facts, preferences, goals, projects, and history from the user's "
    "local Jarvis memory):"
)

_SECTION_HINTS = {
    "GOAL": {
        "aim", "aims", "become", "build", "building", "goal", "goals", "hoping",
        "plan", "planning", "plans", "start", "starting", "want", "wanted", "wants",
    },
    "PREFERENCE": {
        "enjoy", "favorite", "favourite", "instead", "like", "likes", "love",
        "prefer", "preference", "preferences", "prefers", "rather",
    },
    "DECISION": {
        "bought", "buy", "chose", "chosen", "decide", "decided", "decision",
        "decisions", "moved", "pick", "picked", "switch", "switched",
    },
    "PROJECT": {
        "build", "building", "develop", "developing", "initiative",
        "initiatives", "project", "projects", "roadmap",
    },
    "HABIT": {
        "always", "daily", "every", "habit", "habits", "normally", "often",
        "routine", "usually", "weekly",
    },
    "HISTORY": {"ago", "before", "earlier", "last", "old", "past", "previous", "yesterday"},
}


class ContextEngineError(ValueError):
    """Raised for invalid engine inputs; the caller may treat it as a soft miss."""


def _normalize_now(now: str | datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if isinstance(now, datetime):
        at = now
    else:
        try:
            at = datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            raise ContextEngineError("now must be an ISO-8601 timestamp") from None
    if at.tzinfo is None:
        raise ContextEngineError("now must include a timezone")
    return at.astimezone(timezone.utc)


def _confidence_weight(confidence: float) -> float:
    if confidence >= _STRONG_CONFIDENCE:
        return 0.06
    if confidence >= _MEDIUM_CONFIDENCE:
        return 0.02
    if confidence >= _WEAK_CONFIDENCE:
        return -0.04
    return -0.12


def _section_fit(memory_type: str, query_terms: set[str]) -> float:
    hints = _SECTION_HINTS.get(memory_type)
    if not hints:
        return 0.0
    return 0.03 if query_terms & hints else 0.0


def _stale_penalty(item: dict, at: datetime, qcover: float) -> float:
    memory_type = item["memory_type"]
    if memory_type == "TEMPORARY":
        return _TEMPORARY_PENALTY
    if memory_type != "HISTORY":
        return 0.0
    try:
        updated = datetime.fromisoformat(item["updated_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (at - updated.astimezone(timezone.utc)).total_seconds() / 86400)
    if age_days > _STALE_HISTORY_DAYS and qcover < 0.5:
        return _STALE_HISTORY_PENALTY
    return 0.0


def _memory_score(
    item: dict,
    query_terms: set[str],
    at: datetime,
    stage: str,
    replacement_ids: set[str],
) -> float:
    claim_terms = tokens(item["claim"])
    overlap = len(claim_terms & query_terms)
    qcover = overlap / max(1, len(query_terms))
    importance = max(1, min(5, int(item["importance"])))
    score = qcover + 0.25 * (importance / 5.0) * qcover
    score += recency_bonus(item["updated_at_utc"], at)
    score += TYPE_WEIGHT.get(item["memory_type"], 0.0)
    score += _confidence_weight(float(item["confidence"]))
    if stage == "EMERGING":
        score -= EMERGING_SHIFT
    score += _section_fit(item["memory_type"], query_terms)
    if item["id"] in replacement_ids:
        score += DEVELOPMENT_BUMP
    score -= _stale_penalty(item, at, qcover)
    return score


def _replacement_ids(db) -> set[str]:
    """Ids of currently-active memories that just superseded an older record.

    A recent replacement is a lightweight signal that the user actively
    restated/updated the fact, worth a small, explainable bump.
    """
    ids: set[str] = set()
    try:
        for old in memory.list_memories(
            db, lifecycle_status="SUPERSEDED", review_state="CONFIRMED", limit=100
        ):
            replacement = old.get("superseded_by_memory_id")
            if replacement:
                ids.add(str(replacement))
    except Exception:
        pass
    return ids


def _gather_memory_items(
    db,
    query_terms: set[str],
    at: datetime,
    *,
    user_message: str,
    include_sensitive: bool,
    max_memories: int,
    broad: bool = False,
) -> list["ContextItem"]:
    if not query_terms:
        return []
    pool = memory.retrieve(db, user_message, limit=100, now=at) or []
    try:
        stage = user_model.build_user_model(db, now=at).stage
    except Exception:
        stage = "ESTABLISHED"
    replacement_ids = _replacement_ids(db)
    ranked: list[tuple[float, dict, float]] = []
    for item in pool:
        if item.get("is_sensitive") and not include_sensitive:
            continue
        claim_terms = tokens(item["claim"])
        if not (claim_terms & query_terms):
            continue
        qcover = len(claim_terms & query_terms) / max(1, len(query_terms))
        if qcover < MIN_MEMORY_COVERAGE:
            continue
        if is_overridden(item["claim"], user_message):
            continue
        score = _memory_score(item, query_terms, at, stage, replacement_ids)
        ranked.append((score, item, qcover))
    ranked.sort(
        key=lambda entry: (
            entry[0],
            float(entry[1]["confidence"]),
            entry[1]["importance"],
            entry[1]["updated_at_utc"],
            entry[1]["id"],
        ),
        reverse=True,
    )
    items: list[ContextItem] = []

    def _as_item(item: dict, qcover: float, score: float) -> "ContextItem":
        source_ids = tuple(
            str(source.get("entry_id"))
            for source in (item.get("sources") or [])
            if source.get("entry_id") is not None
        )
        return ContextItem(
            memory_id=item["id"],
            claim=item["claim"],
            memory_type=item["memory_type"],
            quality="TEMPORARY" if item["memory_type"] == "TEMPORARY" else "CONFIRMED",
            confidence=float(item["confidence"]),
            importance=max(1, min(5, int(item["importance"]))),
            source_entry_ids=source_ids,
            updated_at_utc=item["updated_at_utc"],
            qcover=qcover,
            score=score,
        )

    for score, item, qcover in ranked[:max_memories]:
        items.append(_as_item(item, qcover, score))

    # Broad self-query fallback: when the message asks about the user rather
    # than a narrow topic, top up the lane with the most important confirmed
    # memories even if their wording does not overlap the question. The
    # coverage gate above is deliberately NOT weakened for normal questions.
    if broad and len(items) < max_memories:
        seen = {entry.memory_id for entry in items}
        try:
            broad_pool = memory.overview(db, limit=max_memories * 4, now=at)
        except Exception:
            broad_pool = []
        for item in broad_pool:
            if len(items) >= max_memories:
                break
            if item["id"] in seen:
                continue
            if item.get("is_sensitive") and not include_sensitive:
                continue
            if is_overridden(item["claim"], user_message):
                continue
            claim_terms = tokens(item["claim"])
            qcover = len(claim_terms & query_terms) / max(1, len(query_terms))
            score = _memory_score(item, query_terms, at, stage, replacement_ids)
            items.append(_as_item(item, qcover, score))
            seen.add(item["id"])
    return items


def apply_evidence_filter(evidence: list[dict] | None, query: str, *, external_only: bool = False) -> list[dict]:
    """Conservative evidence pass filter.

    Source evidence already surfaced by ``recall.recall`` is dropped only when
    it shares no keyword or synonym with the current message and is not HIGH
    importance. External-only retrieval skips the filter entirely, because the
    sender/date constraints already decided relevance.
    """
    if not evidence:
        return []
    if external_only:
        return list(evidence)
    query_terms = set(recall.expand_keywords(query))
    if not query_terms:
        return list(evidence)
    kept = []
    for entry in evidence:
        text_terms = set(recall.keywords(entry.get("text") or ""))
        importance = (entry.get("importance") or "").upper()
        if (text_terms & query_terms) or importance == "HIGH":
            kept.append(entry)
    return kept


@dataclass(frozen=True)
class ContextItem:
    """A single ranked, already-vetted memory for the background memory lane."""

    memory_id: str
    claim: str
    memory_type: str
    quality: str  # CONFIRMED, or TEMPORARY for time-boxed commitments
    confidence: float
    importance: int
    source_entry_ids: tuple[str, ...]
    updated_at_utc: str
    qcover: float
    score: float

    def as_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "claim": self.claim,
            "memory_type": self.memory_type,
            "quality": self.quality,
            "confidence": self.confidence,
            "importance": self.importance,
            "source_entry_ids": list(self.source_entry_ids),
            "updated_at_utc": self.updated_at_utc,
            "qcover": self.qcover,
            "score": self.score,
        }


@dataclass
class ContextPackage:
    """Bounded, ranked background context for a single incoming message."""

    query: str
    keywords: tuple[str, ...]
    now: str
    evidence: list[dict]
    evidence_filtered: int
    memories: list[ContextItem]
    broad: bool = False
    overview: str = ""

    def render_memory_lane(self, budget: int | None = None) -> str:
        """Render the memory lane with the exact wording/truncation of the
        ``personal_context`` contract so the two paths stay interchangeable."""
        if not self.memories:
            return ""
        budget = MAX_MEMORY_CHARS if budget is None else budget
        lines = [_MEMORY_HEADER]
        for item in self.memories:
            line = f"- {format_claim({'claim': item.claim, 'confidence': item.confidence})}"
            proposed = "\n".join([*lines, line])
            if len(proposed) <= budget - len(CONTEXT_GUIDANCE) - 2:
                lines.append(line)
                continue
            available = budget - len(CONTEXT_GUIDANCE) - len("\n".join(lines)) - 4
            if available >= 32:
                shortened = line[: available - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
                lines.append(shortened)
            break
        if len(lines) == 1:
            return ""
        result = CONTEXT_GUIDANCE + "\n\n" + "\n".join(lines)
        return result[:budget]

    def render_personal_lane(self, budget: int | None = None) -> str:
        """Render the full personal-context system string.

        Identical to ``render_memory_lane`` for ordinary (narrow) retrieval. For
        an explicit self query (``broad``) it uses the active personal-query
        guidance and prepends the bounded user-model overview, so the model
        answers about the user from the retrieved record instead of denying
        knowledge."""
        if not self.broad:
            return self.render_memory_lane(budget)
        guidance = CONTEXT_GUIDANCE_PERSONAL
        budget = MAX_PERSONAL_CHARS if budget is None else budget
        blocks: list[str] = []
        if self.overview.strip():
            blocks.append(_OVERVIEW_HEADER + "\n" + self.overview.strip())
        if self.memories:
            lines = [_BROAD_MEMORY_HEADER]
            for item in self.memories:
                line = f"- {format_claim({'claim': item.claim, 'confidence': item.confidence})}"
                prefix = len(guidance) + 2 + (len("\n\n".join(blocks)) + 2 if blocks else 0)
                proposed = "\n".join([*lines, line])
                if prefix + len(proposed) <= budget:
                    lines.append(line)
                    continue
                available = budget - prefix - len("\n".join(lines)) - 4
                if available >= 32:
                    shortened = line[: available - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
                    lines.append(shortened)
                break
            if len(lines) > 1:
                blocks.append("\n".join(lines))
        if not blocks:
            return ""
        result = guidance + "\n\n" + "\n\n".join(blocks)
        return result[:budget]

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "keywords": list(self.keywords),
            "now": self.now,
            "evidence": self.evidence,
            "evidence_filtered": self.evidence_filtered,
            "memories": [item.as_dict() for item in self.memories],
        }


def assemble(
    db,
    message: str,
    *,
    evidence: list[dict] | None = None,
    now: str | datetime | None = None,
    include_sensitive: bool = False,
    external_only: bool = False,
    broad: bool = False,
    max_evidence: int = MAX_EVIDENCE_ITEMS,
    max_memories: int = MAX_MEMORY_ITEMS,
    max_memory_chars: int = MAX_MEMORY_CHARS,
) -> ContextPackage:
    """Cross-rate evidence and memory lanes into one deterministic package.

    ``evidence`` is optional so callers who already ran ``recall.recall`` can
    hand it over for the pass filter; the engine never reaches behind the
    caller's own recall decisions. ``broad`` switches the personal lane into
    explicit self-query mode (user-model overview plus importance-ranked
    memories) without changing narrow behavior.
    """
    if not isinstance(message, str):
        raise ContextEngineError("message must be text")
    if isinstance(max_evidence, bool) or not isinstance(max_evidence, int) or not 1 <= max_evidence <= 20:
        raise ContextEngineError("max_evidence must be between 1 and 20")
    if isinstance(max_memories, bool) or not isinstance(max_memories, int) or not 1 <= max_memories <= 20:
        raise ContextEngineError("max_memories must be between 1 and 20")
    if isinstance(max_memory_chars, bool) or not isinstance(max_memory_chars, int) or not 600 <= max_memory_chars <= 5000:
        raise ContextEngineError("max_memory_chars must be between 600 and 5000")
    at = _normalize_now(now)
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list):
        raise ContextEngineError("evidence must be a list of source-evidence dicts")
    evidence = list(evidence)[:max_evidence]
    filtered = apply_evidence_filter(evidence, message, external_only=external_only)[:max_evidence]
    query_terms = tokens(message)
    memories = _gather_memory_items(
        db,
        query_terms,
        at,
        user_message=message,
        include_sensitive=include_sensitive,
        max_memories=max_memories,
        broad=broad,
    )
    overview = ""
    if broad:
        try:
            overview = user_model.render_projection(
                user_model.build_user_model(db, now=at),
                max_chars=MAX_PERSONAL_CHARS,
            )
        except Exception:
            overview = ""
    keywords = tuple(recall.keywords(message)) or tuple(sorted(query_terms))
    return ContextPackage(
        query=message,
        keywords=keywords,
        now=at.isoformat().replace("+00:00", "Z"),
        evidence=filtered,
        evidence_filtered=len(evidence) - len(filtered),
        memories=memories,
        broad=broad,
        overview=overview,
    )