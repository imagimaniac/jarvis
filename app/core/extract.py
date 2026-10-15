"""Extraction: heuristic-first (offline). LLM optional."""
from __future__ import annotations
import re
from datetime import datetime
from typing import Any, Dict, List

from . import models

QUESTION_RE = re.compile(r'^(what|when|why|how|where|who|is|are|can|should|did|do|will)\b', re.I)
DECISION_RE = re.compile(r'\b(decided|decision|will do|going to|plan to)\b', re.I)
LEARNING_RE = re.compile(r'\b(learned|realized|takeaway|lesson|figured out)\b', re.I)
IDEA_RE = re.compile(r'\b(idea|thought|thinking|maybe|could try)\b', re.I)
EVENT_RE = re.compile(r'\b(interview|meeting|appointment|deadline|due|event)\b', re.I)
PROBLEM_RE = re.compile(r'\b(problem|issue|bug|frustrated|stuck|worried)\b', re.I)

NAME_RE = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')  # simple names
TAG_RE = re.compile(r'#(\w+)')
DATE_RE = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')

def _to_list(x: Any) -> List[str]:
    if not x:
        return []
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]

class HeuristicExtractor:
    def extract(self, text: str, created_at_utc: str | None = None) -> Dict[str, Any]:
        t = text or ""
        low = t.lower()
        etype = "note"
        if '?' in t:
            etype = "question"
        if DECISION_RE.search(low):
            etype = "decision"
        if LEARNING_RE.search(low):
            etype = "learning"
        if IDEA_RE.search(low):
            etype = "idea"
        if PROBLEM_RE.search(low):
            etype = "problem"
        if EVENT_RE.search(low):
            etype = "event"
        if QUESTION_RE.match(t.strip()):
            etype = "question"

        topics: List[str] = []
        for m in TAG_RE.findall(t):
            topics.append(m.lower())
        # common terms
        for kw in ["ai", "memory", "jarvis", "gmail", "calendar", "interview", "job", "application", "recruiter", "bank", "alert", "receipt"]:
            if kw in low and kw not in topics:
                topics.append(kw)

        people = []
        for m in NAME_RE.findall(t):
            nm = m.strip()
            if nm and nm not in people:
                people.append(nm)

        projects = []
        for kw in ["project", "startup", "thesis", "portfolio"]:
            if kw in low:
                projects.append(kw)

        places = []
        for kw in ["pune", "mumbai", "bangalore", "delhi", "hyderabad", "remote", "onsite", "hybrid"]:
            if kw in low:
                places.append(kw)

        events = []
        for m in DATE_RE.findall(t):
            events.append(m)

        questions = []
        if etype == "question":
            questions.append(t.strip()[:200])

        decisions = []
        if etype == "decision":
            decisions.append(t.strip()[:200])

        learnings = []
        if etype == "learning":
            learnings.append(t.strip()[:200])

        importance = "MEDIUM"
        if any(k in low for k in ["suspicious", "unusual transaction", "security alert", "urgent", "payment failure", "account locked"]):
            importance = "CRITICAL"
        elif any(k in low for k in ["interview", "scheduled", "invitation", "progressed", "offer", "rejected", "deadline", "alert"]):
            importance = "HIGH"

        category = None
        if any(k in low for k in ["interview", "recruiter", "application", "applied", "job"]):
            category = "job_app"
        if any(k in low for k in ["bank", "transaction", "credit card", "debit", "emi", "loan", "tax", "statement"]):
            category = "financial"

        return {
            "date": None,
            "time_local": None,
            "type": etype,
            "mood": None,
            "people": people,
            "places": places,
            "projects": projects,
            "topics": topics,
            "events": events,
            "questions": questions,
            "decisions": decisions,
            "learnings": learnings,
            "tags": topics,
            "category": category,
            "importance": importance,
            "confidence": 0.6,
            "inferred_by": "heuristic",
            "needs_review": 0,
            "reason": None,
            "is_sensitive": 1 if category == "financial" else 0,
        }
