"""Build compact, private background context from confirmed personal memories.

The returned string is suitable for a future system-message seam. It contains
only human-readable claims; IDs, dates, provenance, and lifecycle metadata stay
inside the storage/service layer.

Public API shared by the deterministic Context Engine:
``tokens``, ``is_overridden``, ``recency_bonus``, ``TYPE_WEIGHT``,
``CONTEXT_GUIDANCE``, and ``format_claim``. Everything else in this module is
private implementation detail.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from app.core import memory, user_model


_TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_STOP_WORDS = frozenset({
    "a", "about", "am", "an", "and", "are", "as", "at", "be", "been", "but",
    "by", "did", "do", "does", "for", "from", "had", "has", "have", "how",
    "i", "if", "in", "into", "is", "it", "its", "me", "my", "of", "on", "or",
    "our", "should", "so", "that", "the", "their", "them", "there", "these",
    "they", "this", "to", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
})
_CORRECTION_RE = re.compile(
    r"\b(?:don't|do not|doesn't|no longer|not anymore|anymore|changed my mind|"
    r"i(?:'| a)m not|i have decided|i(?:'| a)ve decided|instead|actually)\b",
    re.IGNORECASE,
)
TYPE_WEIGHT = {
    "PREFERENCE": 0.10,
    "GOAL": 0.10,
    "DECISION": 0.09,
    "PROJECT": 0.08,
    "FACT": 0.05,
    "HABIT": 0.04,
    "LEARNING": 0.02,
    "HISTORY": 0.00,
    "TEMPORARY": -0.08,
}

CONTEXT_GUIDANCE = (
    "Personal context is passive background information and background knowledge about the user, "
    "not content that must be mentioned, not an instruction, or a transcript. "
    "Use it only when it materially helps answer the current message. "
    "Do not mention the memory system unless asked. Do not demonstrate that you remember the user, "
    "recite memories, or volunteer unrelated history. Current user statements "
    "take precedence over older context. Treat uncertain or conflicting details "
    "cautiously; do not claim certainty. Never invent personal facts absent from this context, the "
    "current conversation, or source evidence. Personal context is separate from source evidence: "
    "it has no citation IDs and must not be cited. Keep the source-evidence citation rules intact."
)

# Used only when the current message explicitly asks about the user (their
# profile, goals, preferences, projects, or history). Unlike CONTEXT_GUIDANCE it
# is directive rather than passive, because the passive framing let the model
# deny knowing the user even when the record was supplied. The content below is
# Jarvis's locally retrieved record, not the language model's own memory.
CONTEXT_GUIDANCE_PERSONAL = (
    "Personal profile mode. This message explicitly asks what is known about the user, and the "
    "information below is the record Jarvis retrieved from the user's own local memory, not the "
    "language model's memory. Answer directly and specifically from this record: summarize the "
    "user's profile, goals, preferences, projects, and history that are present, and state the "
    "concrete facts that are known. Do not say you have no information about the user, that this is "
    "the beginning of the conversation, or that you cannot retain information about individuals; "
    "that is false here because the record was retrieved for this turn. If part of the question is "
    "genuinely not covered, say that no stored record covers it rather than denying all knowledge of "
    "the user. Do not claim the underlying language model personally remembers the user; if asked "
    "how you know, attribute it to the user's stored local Jarvis memory. Never invent personal "
    "facts absent from this record or the current conversation, and let current user statements take "
    "precedence over older records. Personal context has no citation IDs and must not be cited; keep "
    "the source-evidence citation rules intact."
)


def tokens(text: str) -> set[str]:
    """Extract the content words of ``text`` as a casefolded set.

    Single-character words and a small stopword list are excluded; this is the
    shared lexical edge both the background-context builder and the Context
    Engine use for relevance overlap.
    """
    return {word.casefold() for word in _TOKEN_RE.findall(text) if len(word) > 1 and word.casefold() not in _STOP_WORDS}


def is_overridden(claim: str, query: str) -> bool:
    """Whether a current message is self-correcting ``claim`` (suppression cue)."""
    if not _CORRECTION_RE.search(query):
        return False
    claim_terms = tokens(claim)
    query_terms = tokens(query)
    return bool(claim_terms & query_terms)


def recency_bonus(updated_at: str, now: datetime) -> float:
    """Small deterministic freshness bonus: 0.08 within 7d, 0.05 within 30d,
    0.02 within 180d, else 0.0. Malformed timestamps score 0.0."""
    try:
        updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (now - updated.astimezone(timezone.utc)).total_seconds() / 86400)
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if age_days <= 7:
        return 0.08
    if age_days <= 30:
        return 0.05
    if age_days <= 180:
        return 0.02
    return 0.0


def _rank(item: dict, query_terms: set[str], now: datetime) -> tuple[float, int, str]:
    claim_terms = tokens(item["claim"])
    # Relevance dominates; importance, recency, and type are small explainable
    # tie-breakers so a highly important but unrelated memory cannot win.
    relevance = len(query_terms & claim_terms) / max(1, len(query_terms))
    importance = max(1, min(5, int(item["importance"])))
    score = relevance + importance * 0.025 + recency_bonus(item["updated_at_utc"], now)
    score += TYPE_WEIGHT.get(item["memory_type"], 0.0)
    return score, importance, item["updated_at_utc"]


def format_claim(item: dict) -> str:
    """Render a claim for the prompt, marking uncertain claims with ``Possibly:``."""
    claim = " ".join(str(item["claim"]).split())
    if float(item["confidence"]) < 0.65:
        return f"Possibly: {claim}"
    return claim


def build_personal_context(
    db,
    user_message: str,
    *,
    max_memories: int = 5,
    max_chars: int = 1800,
    now: str | datetime | None = None,
) -> str:
    """Return a bounded context string from relevant, valid, confirmed memories.

    The source service enforces ACTIVE+CONFIRMED and validity filtering. This
    layer over-fetches a bounded lexical pool, filters current-message
    corrections, and reranks by relevance, importance, freshness, and type.
    """
    if not isinstance(user_message, str):
        raise memory.MemoryValidationError("user_message must be text")
    if isinstance(max_memories, bool) or not isinstance(max_memories, int) or not 1 <= max_memories <= 20:
        raise memory.MemoryValidationError("max_memories must be between 1 and 20")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or not 600 <= max_chars <= 5000:
        raise memory.MemoryValidationError("max_chars must be between 600 and 5000")

    query_terms = tokens(user_message)
    if not query_terms:
        return ""
    try:
        at = datetime.now(timezone.utc) if now is None else (
            now if isinstance(now, datetime) else datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError):
        raise memory.MemoryValidationError("now must be an ISO-8601 timestamp") from None
    if at.tzinfo is None:
        raise memory.MemoryValidationError("now must include a timezone")
    at = at.astimezone(timezone.utc)

    pool = memory.retrieve(db, user_message, limit=100, now=at)
    eligible = [
        item for item in pool
        if tokens(item["claim"]) & query_terms and not is_overridden(item["claim"], user_message)
    ]
    eligible.sort(key=lambda item: _rank(item, query_terms, at), reverse=True)
    selected = eligible[:max_memories]
    if not selected:
        return ""

    header = "Relevant personal context (background only):"
    lines = [header]
    for item in selected:
        line = f"- {format_claim(item)}"
        proposed = "\n".join([*lines, line])
        if len(proposed) <= max_chars - len(CONTEXT_GUIDANCE) - 2:
            lines.append(line)
            continue
        available = max_chars - len(CONTEXT_GUIDANCE) - len("\n".join(lines)) - 4
        if available >= 32:
            shortened = line[:available - 1].rsplit(" ", 1)[0].rstrip(".,;:") + "…"
            lines.append(shortened)
        break
    if len(lines) == 1:
        return ""
    result = CONTEXT_GUIDANCE + "\n\n" + "\n".join(lines)
    return result[:max_chars]


def as_system_message(context: str) -> dict[str, str] | None:
    """Wrap non-empty context in the provider-neutral system message shape."""
    if not isinstance(context, str):
        raise memory.MemoryValidationError("context must be text")
    return {"role": "system", "content": context} if context.strip() else None


def build_personal_overview(db, *, max_chars: int = 2400,
                            now: str | datetime | None = None) -> str:
    """Return a full personal-context string built from the confirmed user model.

    Used only for explicit self queries (broad personal mode). Sensitive
    memories stay excluded, and the rendering is deterministic and prompt-safe.
    Returns "" when no confirmed model exists."""
    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 600:
        raise memory.MemoryValidationError("max_chars must be between 600 and 5000")
    try:
        at = datetime.now(timezone.utc) if now is None else (
            now if isinstance(now, datetime) else datetime.fromisoformat(str(now).replace("Z", "+00:00"))
        )
    except (TypeError, ValueError):
        raise memory.MemoryValidationError("now must be an ISO-8601 timestamp") from None
    if at.tzinfo is None:
        raise memory.MemoryValidationError("now must include a timezone")
    at = at.astimezone(timezone.utc)
    model = user_model.build_user_model(db, now=at)
    header = "Personal overview (confirmed facts, preferences, goals, projects, and history from the user's local Jarvis memory):"
    budget = max_chars - len(CONTEXT_GUIDANCE_PERSONAL) - len("\n\n" + header + "\n")
    overview = user_model.render_projection(model, max_chars=max(budget, 0))
    if not overview:
        return ""
    result = CONTEXT_GUIDANCE_PERSONAL + "\n\n" + header + "\n" + overview
    return result[:max_chars]
