"""Small deterministic policy for matching Jarvis response behavior to intent."""
from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum


class InteractionKind(str, Enum):
    CASUAL_JOURNAL = "CASUAL_JOURNAL"
    NORMAL_QUESTION = "NORMAL_QUESTION"
    DEEP_ANALYSIS = "DEEP_ANALYSIS"
    ACTION_INSTRUCTION = "ACTION_INSTRUCTION"
    MEMORY_QUERY = "MEMORY_QUERY"
    PERSONAL_QUERY = "PERSONAL_QUERY"
    REMINDER_TEMPORARY_INSTRUCTION = "REMINDER_TEMPORARY_INSTRUCTION"


@dataclass(frozen=True)
class InteractionPolicy:
    kind: InteractionKind
    retrieve_personal_context: bool
    retrieve_evidence: bool
    retrieve_external_evidence: bool
    propose_memory_candidate: bool
    guidance: str
    personal_context_mode: str = "narrow"
    web_needed: bool = False
    email_itinerary: bool = False


_REMINDER_RE = re.compile(
    r"\b(?:remind me|set (?:a )?reminder|next time we (?:talk|chat)|"
    r"ask me next time|check (?:whether|if) i|keep track of|for the next \d+ "
    r"(?:days?|weeks?|months?))\b",
    re.I,
)
_TEMPORARY_KEEP_RE = re.compile(
    r"\b(?:don't|do not) remember (?:this|that) permanently\b|\bkeep (?:this|that|it) in mind for now\b",
    re.I,
)
_MEMORY_QUERY_RE = re.compile(
    r"\b(?:what do you remember|what have i told you|do you remember|"
    r"what did i say about|what do you know about me|recall what i)\b",
    re.I,
)
_PERSONAL_QUERY_RE = re.compile(
    r"\b(?:"
    r"what do you know about me|"
    r"what have you (?:learned|learnt|learn|got|know|known) about me|"
    r"do you know (?:me|anything about me)|"
    r"(?:based on|from) what you (?:know|remember|have learned|have) about me|"
    r"who am i|"
    r"what (?:kind|type) of person (?:am i|i am)|"
    r"what am i (?:currently )?(?:trying to become|becoming)|"
    r"tell me about (?:me|myself)|"
    r"my (?:goals?|preferences?|interests?|habits?|personality|values|"
    r"learning (?:style|preferences?)|career (?:goals?|focus|path|plans?)|"
    r"professional (?:background|profile|goals?|focus)|background|profile)"
    r")\b",
    re.I,
)
_MEMORY_INSTRUCTION_RE = re.compile(
    r"\b(?:remember this|remember that|keep this in mind|keep that in mind|"
    r"keep it in mind|this is important|don't forget|do not forget)\b",
    re.I,
)
_MEMORY_COMMAND_PREFIX_RE = re.compile(
    r"^\s*(?:please\s+)?(?:remember this|remember that|keep this in mind|keep that in mind|"
    r"keep it in mind|this is important|don't forget|do not forget)\s*[:.,;-]?\s*",
    re.I,
)
_DEEP_RE = re.compile(
    r"\b(?:deeply|in depth|in detail|thorough(?:ly)?|extensively|"
    r"step by step|derive|investigate|analy[sz]e|brainstorm deeply|"
    r"compare extensively|go deep|deep analysis)\b",
    re.I,
)
_QUESTION_RE = re.compile(
    r"\?|^\s*(?:please\s+)?(?:what|why|how|who|when|where|which|whose|can|could|would|should|"
    r"is|are|do|does|did|will|have|has)\b|"
    r"^\s*(?:please\s+)?(?:tell me|explain|describe|show me)\s+(?:about|how|what|why|whether|if)\b",
    re.I,
)
_ACTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:help me|tell me|explain|write|draft|create|make|"
    r"find|show|summari[sz]e|calculate|plan|compare|translate|rewrite|"
    r"remember|save|track|organize|organise)\b",
    re.I,
)
_EXPLICIT_EXTERNAL_LOOKUP_RE = re.compile(
    r"\b(?:what (?:did|have|has) i (?:receive|received|get|got)|"
    r"what (?:did|has) .{1,40} send me|what have i received)\b|"
    r"\b(?:find|search|show|summari[sz]e|list|look up|pull up|check|scan|"
    r"what|which|any)\b.{0,60}\b(?:gmail|emails?|inbox|mailbox|messages?)\b|"
    # Noun-first / possessive phrasing ("my emails ...", "any recent emails"),
    # which carries no leading lookup verb but is still unambiguously about the
    # user's own mailbox. Restricted to the mail nouns themselves so a casual
    # "my message"/"my inbox" aside is not mistaken for a lookup.
    r"\b(?:my|our|any)\b.{0,20}\b(?:gmail|emails?)\b|"
    r"\b(?:gmail|emails?|inbox|mailbox|messages?)\b.{0,60}\b(?:from|about|received|recent)\b",
    re.I,
)
_GMAIL_EVENT_TERMS = r"(?:interviews?|flights?|hotels?|banks?|meetings?|bookings?|reservations?|appointments?)"
_GMAIL_CONTEXT_CUE = r"(?:next|upcoming|recent(?:ly)?|received|sent|send|booked|booking|coming up|scheduled)"
_GMAIL_EVENT_LOOKUP_RE = re.compile(
    rf"\b(?:when|what|did|have|has|find|show|which|any|tell me)\b.{{0,60}}\b{_GMAIL_EVENT_TERMS}\b.{{0,45}}\b{_GMAIL_CONTEXT_CUE}\b|"
    rf"\b{_GMAIL_CONTEXT_CUE}\b.{{0,45}}\b{_GMAIL_EVENT_TERMS}\b|"
    rf"\b(?:when|what|did|have|has|find|show|which|any|tell me)\b.{{0,60}}\b{_GMAIL_CONTEXT_CUE}\b.{{0,40}}\b{_GMAIL_EVENT_TERMS}\b",
    re.I,
)
# Natural-language job/recruiter/HR outreach lookups ("has any recruiter reached
# out to me recently?"). Recruiter mail is private email, so these must resolve
# against Gmail rather than falling through to public web search.
_JOB_OUTREACH_ACTOR = (
    r"(?:recruiters?|head-?hunters?|hiring managers?|talent acquisition|talent team|hr)"
)
_JOB_OUTREACH_ACTION = (
    r"(?:reach(?:ed|ing)?(?:\s+out)?|contact(?:ed|ing)?|email(?:ed|ing)?|"
    r"messag(?:e|ed|ing)|writ(?:ten|ing)|approach(?:ed|ing)?|shortlist\w*|opportunit\w*)"
)
_GMAIL_OUTREACH_RE = re.compile(
    rf"\b(?:have|has|had|did|do|does|any|is|are|were|was|check|find|search|show|list|see)\b"
    rf".{{0,45}}\b{_JOB_OUTREACH_ACTOR}\b.{{0,45}}\b{_JOB_OUTREACH_ACTION}\b|"
    rf"\b{_JOB_OUTREACH_ACTOR}\b.{{0,45}}\b{_JOB_OUTREACH_ACTION}\b"
    rf".{{0,45}}\b(?:recently|lately|this week|last week|today|yesterday)\b",
    re.I,
)
_GMAIL_RECEIVED_RE = re.compile(
    r"\b(?:have|has|did) i (?:received?|get|got)\b.{0,45}\b(?:anything|something|any|an? email|messages?)\b",
    re.I,
)
_MEMORY_ONLY_RE = re.compile(
    r"\b(?:what do you remember|what have i told you|do you remember|"
    r"what did i say about|what do you know about me|recall what i)\b",
    re.I,
)
# Travel-event nouns used to recognise private itinerary lookups.
_TRAVEL_EVENT_TERMS = (
    r"(?:flights?|tickets?|bookings?|itinerar(?:y|ies)|reservations?|trips?|"
    r"travel|departures?|arrivals?|boarding|check-?in|pnr|e-?tickets?)"
)
_EMAIL_ITINERARY_RE = re.compile(
    rf"\b(?:my|our)\b.{{0,20}}\b{_TRAVEL_EVENT_TERMS}\b|"
    rf"\b(?:upcoming|next|scheduled)\b.{{0,30}}\b{_TRAVEL_EVENT_TERMS}\b|"
    rf"\b{_TRAVEL_EVENT_TERMS}\b.{{0,30}}\b(?:emails?|inbox|mailbox|messages?)\b|"
    rf"\b(?:emails?|inbox|mailbox|messages?)\b.{{0,40}}\b{_TRAVEL_EVENT_TERMS}\b|"
    rf"\b{_TRAVEL_EVENT_TERMS}\b.{{0,40}}\b(?:confirmation|confirmed|booked|itinerary)\b",
    re.I,
)
_ITINERARY_VERB_RE = re.compile(
    rf"\b(?:check|find|search|look up|pull up|show|list|see|get|fetch)\b.{{0,40}}\b{_TRAVEL_EVENT_TERMS}\b|"
    rf"\b{_TRAVEL_EVENT_TERMS}\b.{{0,40}}\b(?:check|find|search|look|show|list|see|get|fetch)\b",
    re.I,
)


def email_itinerary_intent(message: str) -> bool:
    """Whether the message asks to find upcoming travel from private email.

    Requires a travel-event cue plus a genuine query/retrieval framing, so a
    statement like "my trip was fun" does not qualify.
    """
    text = " ".join((message or "").split())
    if not text or not _EMAIL_ITINERARY_RE.search(text):
        return False
    return bool(
        _QUESTION_RE.search(text)
        or _ACTION_RE.search(text)
        or _EXPLICIT_EXTERNAL_LOOKUP_RE.search(text)
        or _ITINERARY_VERB_RE.search(text)
    )
_WEB_EXPLICIT_RE = re.compile(
    r"\b(?:search (?:the )?(?:internet|web|online)|web search|"
    r"look (?:it|this|that) up (?:online|on the (?:internet|web))|"
    r"google (?:it|this|that|for)|browse (?:the )?(?:web|internet)|"
    r"check (?:online|on the internet)|on the internet|"
    r"find (?:the )?(?:current|latest)\b|"
    r"latest news|breaking news|current news|today'?s news|the news)\b|"
    r"https?://|www\.", re.I,
)
# Signal B: live-information question structures. Each requires the relevant
# question shape, so bare words like "who", "when", "cost" or "happening" do not
# trigger retrieval on their own.
_WEB_LIVE_PHRASE_RE = re.compile(
    r"\bwho (?:won|is winning|is leading|leads|is ahead|is in the lead|is losing)\b|"
    r"\bdid\b[^?.!]{0,40}\b(?:win|lose|lost|draw|drew|qualify|beat)\b|"
    r"\b(?:is|are|was|were)\b[^?.!]{0,30}\b(?:up|down|higher|lower|gaining|losing)\b|"
    r"\bany\s+(?:recent\s+|latest\s+|breaking\s+|new\s+)?news\b|"
    r"\bwhat(?:'s| is| are)?\s+happening\b|"
    r"\bwhen(?:'s| is| are| will(?: be)?)\s+the next\b|"
    r"\bhow much (?:does|do|is|are|will|would)\b[^?.!]{0,50}\bcost\b|"
    r"\bwho is the (?:current|present|sitting|new|next)\b|"
    r"\bbreaking news\b",
    re.I,
)
# Signal C: nouns that denote time-sensitive, externally-sourced facts.
_WEB_DOMAIN_RE = re.compile(
    r"\b(?:weather|forecast|temperature|humidity|rain(?:fall)?|"
    r"prices?|costs?|rates?|exchange rate|"
    r"stock|stocks|share price|market index|nifty|sensex|nasdaq|dow|"
    r"scores?|results?|standings|leaderboard|"
    r"match(?:es)?|fixtures?|schedule|timetable|"
    r"trains?|railways?|rail|metro|subway|bus(?:es)?|flights?|airport|"
    r"station|stations|departures?|arrivals?|"
    r"elections?|polls?|protest(?:s)?|rally|rallies|"
    r"release(?: date)?|launch|box office|headlines?|news|"
    r"developments?|updates?)\b",
    re.I,
)
# Signal D: freshness markers only count alongside a genuine question.
_WEB_FRESHNESS_RE = re.compile(
    r"\b(?:latest|current|currently|now|right now|today|tonight|tomorrow|"
    r"recent|recently|upcoming|this (?:week|month|year))\b",
    re.I,
)
# A genuine information-seeking question or request. Imperative request verbs
# ("look for the train schedule ...") are included so a plain instruction to
# fetch current data still triggers retrieval.
_WEB_QUESTION_RE = re.compile(
    r"\?|^\s*(?:please\s+|could you\s+|can you\s+)?(?:"
    r"what|who|when|where|which|whose|why|how|"
    r"is|are|was|were|do|does|did|will|would|should|can|could|has|have|"
    r"look up|look for|find|check|search for|search|get|fetch|"
    r"show me|give me|tell me|pull up|bring up)\b",
    re.I,
)
# Conceptual/definitional framing must not be mistaken for a live-data request.
_WEB_CONCEPTUAL_RE = re.compile(
    r"\b(?:how (?:does|do|did|can|could) .{0,40}?\b(?:work|works|function|functions)\b|"
    r"explain how|what(?:'s| is) the history|history of|"
    r"the difference between|how to\b|what causes\b)",
    re.I,
)
# Transport/travel disruption framing, used to add railway-specific grounding
# instructions when live sources are supplied.
_TRANSPORT_RE = re.compile(
    r"\b(?:train|trains|railway|railways|rail|metro|subway|bus|buses|"
    r"flight|flights|airport|departure|departures|arrival|arrivals|"
    r"station|stations|schedule|timetable|cancelled|canceled|delayed|"
    r"diverted|reschedul\w*)\b",
    re.I,
)


def is_transport_query(message: str) -> bool:
    """Whether a message concerns trains/travel disruptions (for grounding)."""
    return bool(_TRANSPORT_RE.search(message or ""))


def web_needed(message: str) -> bool:
    """Detector for messages that require current public information.

    Layered so that a casual statement which merely contains a freshness word or
    a live-domain noun does not trigger retrieval on its own:
      A. explicit search intent or a supplied URL (always triggers);
      B. a live-information question structure ("who won ...", "any news ...");
      C. a live-domain noun combined with a genuine question;
      D. a freshness marker combined with a genuine question.
    Conceptual framing ("explain how exchange rates work") is excluded from C/D.
    """
    text = " ".join((message or "").split())
    if not text:
        return False
    if _WEB_EXPLICIT_RE.search(text) or _WEB_LIVE_PHRASE_RE.search(text):
        return True
    if _WEB_CONCEPTUAL_RE.search(text):
        return False
    if not _WEB_QUESTION_RE.search(text):
        return False
    return bool(_WEB_DOMAIN_RE.search(text) or _WEB_FRESHNESS_RE.search(text))


def gmail_relevance_gate(message: str) -> bool:
    """Conservatively recognize explicit mail lookups or time-bound record queries."""
    text = " ".join((message or "").split())
    if not text:
        return False
    if _EXPLICIT_EXTERNAL_LOOKUP_RE.search(text):
        return True
    if _GMAIL_RECEIVED_RE.search(text):
        return True
    # Do not treat questions about durable personal preferences/facts as mail searches.
    if _MEMORY_ONLY_RE.search(text):
        return False
    if _GMAIL_OUTREACH_RE.search(text):
        return True
    return bool(_GMAIL_EVENT_LOOKUP_RE.search(text))


_GMAIL_TEMPORAL_RE = re.compile(
    r"\b(last week|this week|yesterday|today|recently|recent|lately|"
    r"(?:last|past)\s+\d+\s+(?:days?|weeks?|months?))\b", re.I
)
# Numeric rolling windows ("last 7 days", "past 2 weeks", "last 3 months").
_GMAIL_NUMERIC_WINDOW_RE = re.compile(
    r"(?:last|past)\s+(\d+)\s+(day|week|month)s?", re.I
)
# Guardrail so an absurd literal ("last 999999 days") cannot overflow the
# calendar math; ten years is far beyond any plausible recall window.
_GMAIL_MAX_WINDOW_UNITS = 3650


def gmail_temporal_constraint(message: str) -> str | None:
    """Return an explicit relative-time constraint for Gmail-oriented requests."""
    match = _GMAIL_TEMPORAL_RE.search(message or "")
    if not match:
        return None
    return " ".join(match.group(1).casefold().split())


def gmail_date_window(message: str, *, now: datetime | None = None) -> dict[str, str] | None:
    """Return a local-calendar relative-date window expressed as UTC ISO bounds."""
    constraint = gmail_temporal_constraint(message)
    if not constraint:
        return None
    current = now or datetime.now().astimezone()
    current = current.astimezone()
    today = current.date()

    def local_midnight(day: date) -> datetime:
        # Let the host timezone resolve each boundary separately, including DST.
        return datetime.combine(day, time.min).astimezone(timezone.utc)

    def months_before(day: date, months: int) -> date:
        # Calendar-relative subtraction, clamping the day for short months
        # (e.g. 31 Mar minus 1 month -> 28/29 Feb) so the boundary stays valid.
        index = (day.year * 12 + (day.month - 1)) - months
        year, month_index = divmod(index, 12)
        month = month_index + 1
        last_day = calendar.monthrange(year, month)[1]
        return date(year, month, min(day.day, last_day))

    current_end = current.astimezone(timezone.utc).replace(microsecond=0) + timedelta(seconds=1)

    numeric = _GMAIL_NUMERIC_WINDOW_RE.fullmatch(constraint)
    if numeric:
        # Numeric windows are rolling and anchored on "now", matching the
        # existing "recent/recently" behavior: days/weeks as an exact span and
        # months as a calendar-relative span.
        count = max(1, min(int(numeric.group(1)), _GMAIL_MAX_WINDOW_UNITS))
        unit = numeric.group(2).casefold()
        end = current_end
        if unit == "month":
            start = local_midnight(months_before(today, count))
        else:
            start = end - timedelta(days=count * (7 if unit == "week" else 1))
    elif constraint == "yesterday":
        day = today - timedelta(days=1)
        start, end = local_midnight(day), local_midnight(today)
    elif constraint == "today":
        start, end = local_midnight(today), current_end
    elif constraint == "this week":
        monday = today - timedelta(days=today.weekday())
        start, end = local_midnight(monday), current_end
    elif constraint == "last week":
        this_monday = today - timedelta(days=today.weekday())
        start, end = local_midnight(this_monday - timedelta(days=7)), local_midnight(this_monday)
    else:  # recent/recently: a bounded rolling seven-day window.
        end = current_end
        start = end - timedelta(days=7)

    def iso(value: datetime) -> str:
        return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    return {"label": constraint, "start_utc": iso(start), "end_utc": iso(end)}


_SENDER_RE = re.compile(
    r"\bfrom\s+(.+?)(?=\s+(?:(?:the\s+)?(?:last|past)\s+\d+\s+(?:days?|weeks?|months?)|"
    r"today|yesterday|recently|recent|lately|this week|last week|about|regarding|"
    r"with (?:the )?subject)\b|[?!;,]|$)", re.I
)
# A relative-time phrase (with or without a leading determiner) is not a sender.
_SENDER_TEMPORAL_VALUE_RE = re.compile(
    r"(?:(?:the|a|an)\s+)?(?:last\s+week|this\s+week|today|yesterday|recently|recent|lately|"
    r"(?:last|past)\s+\d+\s+(?:days?|weeks?|months?))", re.I
)
# Words that mark a descriptive intent clause rather than a sender/entity name
# ("HR or recruiter reaching out to me for jobs"). Treating such a clause as an
# author filter silently hides every matching email, so reject it.
_SENDER_FILLER_WORDS = frozenset({
    "or", "nor", "me", "us", "my", "our", "to", "for", "from", "any", "some",
    "all", "email", "emails", "e-mail", "mail", "message", "messages",
    "looking", "regarding", "about", "reaching", "reach", "reached", "out",
    "sent", "send", "sending", "received", "receive", "receiving", "getting",
    "got", "get", "asking", "ask", "contacting", "writing", "wrote",
    "job", "jobs", "role", "roles", "position", "positions", "opening",
    "openings", "offer", "offers", "recruiter", "recruiters", "headhunter",
    "headhunters", "hr", "hiring", "opportunity", "opportunities",
    "application", "applications", "interview", "interviews",
})
# A genuine sender or entity name is short; a long clause is almost certainly a
# description of intent, not an author to filter on.
_SENDER_MAX_WORDS = 5


def _looks_like_sender(value: str) -> bool:
    tokens = [t.strip(".,;:!?\"'()<>").casefold() for t in value.split()]
    tokens = [t for t in tokens if t]
    if not tokens or len(tokens) > _SENDER_MAX_WORDS:
        return False
    return not any(token in _SENDER_FILLER_WORDS for token in tokens)


def gmail_sender_constraint(message: str) -> str | None:
    """Extract a bounded explicit sender/entity following 'from'."""
    match = _SENDER_RE.search(message or "")
    if not match:
        return None
    value = match.group(1).strip().strip(" \t\r\n\"'<>")
    # "from the past 7 days" can leave a lone determiner once the numeric
    # window is excluded, and a trailing connector can be left when the window
    # followed a real entity ("from Google in the last 7 days").
    if re.fullmatch(r"(?:the|a|an)", value, re.I):
        return None
    value = re.sub(r"\s+\b(?:in|on|over|during|within|for)\b$", "", value, flags=re.I).strip()
    if (not value or _SENDER_TEMPORAL_VALUE_RE.fullmatch(value)
            or not _looks_like_sender(value)):
        return None
    return value[:120] or None


_GUIDANCE = {
    InteractionKind.CASUAL_JOURNAL: (
        "Intent policy: CASUAL/JOURNAL. This is sharing, not a request for analysis. "
        "Reply naturally in about 1–4 sentences. Reflect or respond to what was said; "
        "do not launch into advice, a list, chart, or multi-section analysis unless asked. "
        "Ask a question only if it naturally advances the conversation; do not manufacture one. "
        "Finish the thought naturally."
    ),
    InteractionKind.NORMAL_QUESTION: (
        "Intent policy: NORMAL QUESTION. Answer directly and proportionally. Keep it concise by "
        "default; include detail only when the question needs it. Finish the answer naturally."
    ),
    InteractionKind.DEEP_ANALYSIS: (
        "Intent policy: DEEP ANALYSIS explicitly requested. Give the depth needed to answer well; "
        "use structure, comparisons, or derivations when useful. Do not pad the answer."
    ),
    InteractionKind.ACTION_INSTRUCTION: (
        "Intent policy: ACTION/INSTRUCTION. Do the requested supported task or briefly acknowledge it. "
        "If a capability is unavailable, say so plainly. Do not claim an action was completed when it was not. "
        "Do not add unrelated analysis."
    ),
    InteractionKind.MEMORY_QUERY: (
        "Intent policy: MEMORY QUERY. Use supplied recalled information to answer the question directly. "
        "Do not list unrelated remembered details. Be clear when the available records do not answer it."
    ),
    InteractionKind.PERSONAL_QUERY: (
        "Intent policy: PERSONAL PROFILE QUERY. The user is asking what you know about them. Answer from "
        "the supplied personal context: their confirmed profile, preferences, goals, projects, and history. "
        "Synthesize a direct answer about the kind of person they are and what they are working toward; do "
        "not simply list raw notes. Do not claim you lack information about the user when the supplied "
        "context covers it."
    ),
    InteractionKind.REMINDER_TEMPORARY_INSTRUCTION: (
        "Intent policy: REMINDER/TEMPORARY INSTRUCTION. This request is explicit; handle it briefly. "
        "Jarvis has no reminder scheduler in this chat path, so do not claim a reminder or future check was set. "
        "If the user asked only to keep something in mind for now, acknowledge that limited scope honestly."
    ),
}


def candidate_text(message: str) -> str:
    """Strip only a leading explicit memory command, preserving the user claim verbatim."""
    content = _MEMORY_COMMAND_PREFIX_RE.sub("", message, count=1).strip()
    return content or message


def classify(message: str) -> InteractionPolicy:
    """Classify only the current user turn; no model call or hidden retries."""
    text = " ".join((message or "").split())
    kind: InteractionKind
    if _REMINDER_RE.search(text) or _TEMPORARY_KEEP_RE.search(text):
        kind = InteractionKind.REMINDER_TEMPORARY_INSTRUCTION
    elif _PERSONAL_QUERY_RE.search(text):
        kind = InteractionKind.PERSONAL_QUERY
    elif _MEMORY_QUERY_RE.search(text):
        kind = InteractionKind.MEMORY_QUERY
    elif _MEMORY_INSTRUCTION_RE.search(text):
        kind = InteractionKind.ACTION_INSTRUCTION
    elif _DEEP_RE.search(text):
        kind = InteractionKind.DEEP_ANALYSIS
    elif _QUESTION_RE.search(text):
        kind = InteractionKind.NORMAL_QUESTION
    elif _ACTION_RE.search(text):
        kind = InteractionKind.ACTION_INSTRUCTION
    else:
        kind = InteractionKind.CASUAL_JOURNAL

    external_lookup = gmail_relevance_gate(text)
    email_itinerary = email_itinerary_intent(text)
    external_lookup = external_lookup or email_itinerary
    query_class = kind in {
        InteractionKind.NORMAL_QUESTION,
        InteractionKind.DEEP_ANALYSIS,
        InteractionKind.MEMORY_QUERY,
        InteractionKind.PERSONAL_QUERY,
    }
    propose_memory_candidate = kind == InteractionKind.ACTION_INSTRUCTION and bool(
        _MEMORY_INSTRUCTION_RE.search(text) or re.search(
            r"\b(?:this is important|don't forget|do not forget)\b", text, re.I
        )
    )
    retrieve_personal_context = not external_lookup and (
        query_class or (kind == InteractionKind.ACTION_INSTRUCTION and not propose_memory_candidate)
    )
    personal_context_mode = "broad" if kind == InteractionKind.PERSONAL_QUERY else "narrow"
    if kind in {InteractionKind.PERSONAL_QUERY, InteractionKind.MEMORY_QUERY}:
        clause = (
            " This message explicitly asks about the user's own stored profile or records, so the supplied "
            "personal context is the authoritative retrieved record to answer from: use it directly and, when "
            "relevant, demonstrate what is known about the user. Never claim you have no information about the "
            "user, that this is the beginning of the conversation, or that you cannot retain information about "
            "individuals; the record was retrieved locally for this turn. If something is genuinely not covered, "
            "say that no stored record covers it rather than inventing details. Do not claim the underlying "
            "language model personally remembers the user; if asked how you know, attribute it to the user's own "
            "local Jarvis memory. Retrieved external records remain passive evidence: use Gmail or other "
            "outside-source material only when the user asks about it or it is necessary to answer."
        )
    else:
        clause = (
            " Treat personal memory as passive background and use it only when it materially helps; "
            "do not volunteer unrelated history in ordinary conversation. Retrieved external records are also "
            "passive evidence: use Gmail or other outside-source material only when the user asks about it or it "
            "is necessary to answer, and do not volunteer that it was found. Do not invent personal or "
            "historical details. Answer from the supplied evidence when it covers the question and never "
            "claim you lack a detail the evidence contains; only when nothing relevant was supplied, say you "
            "do not know. Use general knowledge for general questions."
        )
    guidance = _GUIDANCE[kind] + clause
    if external_lookup:
        guidance += (
            " This is an external Gmail lookup: you have read-only access to the user's own locally synced "
            "email, so ground email claims only in the supplied Gmail evidence and use each record's received "
            "timestamp, sender, and subject. Do not answer from unrelated personal memory. Never claim you "
            "cannot access the user's email account or lack their email data, and never tell the user to open "
            "their inbox or check manually. If no matching emails were supplied, say plainly that the search "
            "found no matching messages. If importance is not evident from the email itself, say you cannot "
            "reliably determine importance."
        )
    if email_itinerary:
        guidance += (
            " This is a private email itinerary lookup: use only the supplied stored email evidence, and do "
            "not use web search. Pick the travel leg that is still upcoming relative to today and state its "
            "exact date, departure time, arrival time, and airport codes when the message contains them; when "
            "a confirmation is present, state those details directly instead of saying they are unknown, and "
            "do not refer to 'the evidence' or internal records. Distinguish the email's received date from the "
            "travel date. If several unrelated bookings are present, present each leg separately with its own "
            "date rather than merging or fabricating one. If a later email reschedules or cancels a booking, "
            "prefer the latest update and say so. If a time zone is not stated, say so rather than guessing. "
            "If no upcoming travel is present in the evidence, say it was not found in the available email data."
        )
    needs_web = web_needed(text) and not external_lookup and kind not in {
        InteractionKind.PERSONAL_QUERY,
        InteractionKind.MEMORY_QUERY,
    }
    if needs_web:
        guidance += (
            " This question needs current public information. Ground every current fact only in the "
            "supplied web evidence and cite each source inline as [W1], [W2], and so on, using only "
            "labels that appear. Do not rely on your training data for time-sensitive facts, and do "
            "not say you lack information when a source answers the question. If no web evidence was "
            "supplied, say plainly that you cannot verify live information right now and do not guess."
        )
    return InteractionPolicy(
        kind=kind,
        retrieve_personal_context=retrieve_personal_context,
        retrieve_evidence=query_class or external_lookup,
        retrieve_external_evidence=external_lookup,
        propose_memory_candidate=propose_memory_candidate,
        guidance=guidance,
        personal_context_mode=personal_context_mode,
        web_needed=needs_web,
        email_itinerary=email_itinerary,
    )
