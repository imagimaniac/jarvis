"""Epistemic vocabulary.

The rule this module exists to enforce: **Jarvis never presents an inference as
something you said, and never presents anything as bare fact.**

Two separate questions, deliberately kept apart:

  entry_status  - what kind of record is this? Only you can be MEMORY.
                 Only a third party can be RECORDED. Never FACT.
  facet_status  - the structured fields (topics, people, importance, category)
                 are Jarvis's *reading* of the text. That reading is an
                 INFERENCE until a human confirms it, regardless of how good
                 the source record is.

Conflating these is how an assistant ends up inventing that you decided
something you only read about.
"""
from __future__ import annotations

FACT = "FACT"
MEMORY = "MEMORY"
RECORDED = "RECORDED"
INFERENCE = "INFERENCE"
HYPOTHESIS = "HYPOTHESIS"

STATUSES = (FACT, MEMORY, RECORDED, INFERENCE, HYPOTHESIS)

# Below this confidence a reading is flagged as a hypothesis, not a conclusion.
HYPOTHESIS_THRESHOLD = 0.4


def entry_status(origin_kind: str) -> str:
    """Status of the entry itself. `origin_kind` is the only input; it cannot be
    talked up into a fact."""
    return MEMORY if origin_kind == "self" else RECORDED


def facet_status(inferred_by: str | None, confidence: float | None = None) -> str:
    """Status of anything Jarvis derived rather than read verbatim."""
    if inferred_by == "user":
        return MEMORY
    if confidence is not None and confidence < HYPOTHESIS_THRESHOLD:
        return HYPOTHESIS
    return INFERENCE


def may_present_as(status: str, claim: str) -> bool:
    """Guard for the answer path. An inference may not be spoken as fact."""
    return not (status == INFERENCE and claim == FACT)


def label_for(entry: dict, structured: dict | None) -> dict:
    """Display labels for one entry, for use in search/show/answers."""
    es = entry_status(entry.get("origin_kind") or "self")
    s = structured or {}
    return {
        "entry_status": es,
        "facet_status": facet_status(s.get("inferred_by"), s.get("confidence")),
        "needs_review": bool(s.get("needs_review")),
    }