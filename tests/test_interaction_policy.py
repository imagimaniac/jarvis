from app.core import interaction_policy as policy
from datetime import datetime, timezone


def test_casual_and_journal_messages_get_short_conversational_policy_without_retrieval():
    for message in (
        "I was thinking about changing jobs.",
        "Today was a pretty exhausting day.",
        "That idea has been on my mind.",
    ):
        result = policy.classify(message)
        assert result.kind == policy.InteractionKind.CASUAL_JOURNAL
        assert not result.retrieve_evidence
        assert not result.retrieve_personal_context
        assert not result.propose_memory_candidate
        assert "1–4 sentences" in result.guidance
        assert "do not manufacture one" in result.guidance


def test_normal_and_explicit_deep_questions_have_distinct_depth_policy():
    normal = policy.classify("How does a heat pump work?")
    deep = policy.classify("Analyze this deeply and derive the thermodynamic tradeoffs.")
    assert normal.kind == policy.InteractionKind.NORMAL_QUESTION
    assert "concise by default" in normal.guidance
    assert deep.kind == policy.InteractionKind.DEEP_ANALYSIS
    assert "depth needed" in deep.guidance
    assert normal.retrieve_evidence and normal.retrieve_personal_context


def test_explicit_memory_commands_and_queries_are_distinguished():
    instruction = policy.classify("Remember this: I prefer concise answers.")
    query = policy.classify("What do you remember about my work?")
    assert instruction.kind == policy.InteractionKind.ACTION_INSTRUCTION
    assert instruction.propose_memory_candidate
    assert query.kind == policy.InteractionKind.MEMORY_QUERY
    assert query.retrieve_personal_context


def test_reminder_and_temporary_requests_are_classified_without_claiming_support():
    for message in (
        "Remind me about this for the next two weeks.",
        "Next time we talk, ask me whether I finished this.",
        "Don't remember this permanently; just keep it in mind for now.",
    ):
        result = policy.classify(message)
        assert result.kind == policy.InteractionKind.REMINDER_TEMPORARY_INSTRUCTION
        assert not result.propose_memory_candidate
        assert "no reminder scheduler" in result.guidance
        assert "do not claim a reminder" in result.guidance


def test_gmail_evidence_requires_an_explicit_source_lookup_request():
    journal = policy.classify("I was thinking about changing jobs.")
    regular = policy.classify("What options should I consider about changing jobs?")
    emails = policy.classify("What have I received recently about changing jobs?")
    explicit = policy.classify("Show me recent emails about the role.")
    assert not journal.retrieve_evidence
    assert regular.retrieve_evidence and not regular.retrieve_external_evidence
    assert emails.retrieve_external_evidence
    assert explicit.retrieve_external_evidence


def test_gmail_gate_covers_time_bound_personal_records_but_not_memory_or_general_questions():
    positive = (
        "When is my next interview?",
        "Did I get any interview emails recently?",
        "What was the flight booking I received?",
        "Did the bank send me anything important?",
        "What meetings do I have coming up?",
        "Find the email about my hotel booking.",
    )
    negative = (
        "I was thinking about changing jobs.",
        "How do airline interviews usually work?",
        "What do you remember about my preferences for meetings?",
        "Explain how hotel booking systems work.",
        "What is the capital of France?",
    )
    assert all(policy.classify(message).retrieve_external_evidence for message in positive)
    assert not any(policy.classify(message).retrieve_external_evidence for message in negative)
    assert all(not policy.classify(message).retrieve_personal_context for message in positive)
    assert policy.classify("Have I received anything important recently?").retrieve_external_evidence


def test_gmail_relative_windows_are_local_calendar_bounded_and_dynamic():
    now = datetime(2025, 1, 8, 12, 0, tzinfo=timezone.utc)
    yesterday = policy.gmail_date_window("Did I get mail yesterday?", now=now)
    today = policy.gmail_date_window("Any mail today?", now=now)
    recent = policy.gmail_date_window("Have I received anything recently?", now=now)
    assert yesterday["label"] == "yesterday"
    assert today["label"] == "today"
    assert yesterday["start_utc"] < yesterday["end_utc"]
    assert today["start_utc"] < today["end_utc"]
    assert recent["end_utc"] == "2025-01-08T12:00:01Z"
    assert policy.gmail_date_window("Show emails this week", now=now)["label"] == "this week"
    assert policy.gmail_date_window("Show emails last week", now=now)["label"] == "last week"
    assert policy.gmail_date_window("Show emails", now=now) is None


def test_gmail_sender_constraint_extracts_sender_without_temporal_or_subject_text():
    assert policy.gmail_sender_constraint("Did I receive an email from JobsNest yesterday?") == "JobsNest"
    assert policy.gmail_sender_constraint("Did I get anything from jobs@example.test recently?") == "jobs@example.test"
    assert policy.gmail_sender_constraint("Find email about my hotel booking") is None


def test_personal_profile_questions_route_to_broad_personal_context():
    for message in (
        "Based on what you know about me from our previous interactions, what kind of person am I "
        "professionally and what am I currently trying to become?",
        "Who am I and what are my goals?",
        "What are some of my learning preferences and how should Jarvis help me learn?",
        "Tell me about myself.",
        "What do you know about me?",
    ):
        result = policy.classify(message)
        assert result.kind == policy.InteractionKind.PERSONAL_QUERY
        assert result.personal_context_mode == "broad"
        assert result.retrieve_personal_context and result.retrieve_evidence
        assert not result.retrieve_external_evidence
        assert "authoritative" in result.guidance


def test_specific_memory_and_normal_questions_stay_narrow():
    remembered = policy.classify("What do you remember about my work?")
    normal = policy.classify("How does a heat pump work?")
    assert remembered.kind == policy.InteractionKind.MEMORY_QUERY
    assert remembered.personal_context_mode == "narrow"
    assert normal.kind == policy.InteractionKind.NORMAL_QUESTION
    assert normal.personal_context_mode == "narrow"
    # A normal general question must keep the passive-background clause.
    assert "passive background" in normal.guidance


def test_flight_itinerary_lookup_is_private_email_retrieval_not_web():
    exact = policy.classify(
        "Check my emails and tell me when my upcoming flight is. "
        "Give me the exact date and time."
    )
    assert exact.email_itinerary
    assert exact.retrieve_external_evidence
    assert not exact.web_needed, "a private email itinerary must not trigger public web search"
    assert "web search" in exact.guidance

    for message in (
        "when is my upcoming flight",
        "find my flight ticket",
        "what is my flight booking",
        "do I have any upcoming flight",
        "what's my upcoming trip",
    ):
        result = policy.classify(message)
        assert result.email_itinerary, message
        assert result.retrieve_external_evidence, message
        assert not result.web_needed, message


def test_flight_itinerary_ignores_statements_and_keeps_genuine_web_queries():
    for message in ("my trip to Goa was fun", "how do flights work", "book a flight for me"):
        result = policy.classify(message)
        assert not result.email_itinerary, message
    # Explicit public web requests survive the itinerary suppression.
    assert policy.classify("search the web for cheap flights").web_needed
    assert not policy.classify("search the web for cheap flights").email_itinerary


def test_noun_first_and_possessive_email_phrasings_are_private_lookups():
    for message in (
        "my emails in the last 7 days",
        "My emails from the past 7 days.",
        "emails I received over the last 14 days",
        "show me my emails from last week",
        "what emails did I get yesterday?",
        "any recent emails from Google?",
    ):
        result = policy.classify(message)
        assert policy.gmail_relevance_gate(message), message
        assert result.retrieve_external_evidence, message
        assert result.retrieve_evidence, message
        assert not result.web_needed, message
        assert not result.retrieve_personal_context, message


def test_email_lookup_covers_plural_singular_and_common_mail_nouns():
    for message in (
        "my email in the last 7 days",
        "my gmail messages from yesterday",
        "search my gmail for receipts",
        "check my inbox",
        "list recent messages from the bank",
        "any new mail in my mailbox",
    ):
        assert policy.gmail_relevance_gate(message), message


def test_numeric_temporal_windows_are_rolling_and_calendar_relative():
    now = datetime(2025, 3, 15, 12, 0, tzinfo=timezone.utc)
    seven = policy.gmail_date_window("my emails in the last 7 days", now=now)
    past = policy.gmail_date_window("my emails from the past 7 days", now=now)
    two_weeks = policy.gmail_date_window("emails I received over the last 14 days", now=now)
    three_months = policy.gmail_date_window("my emails from the last 3 months", now=now)
    assert seven["label"] == "last 7 days"
    assert past["label"] == "past 7 days"
    assert seven["start_utc"] == "2025-03-08T12:00:01Z"
    assert seven["end_utc"] == "2025-03-15T12:00:01Z"
    assert two_weeks["label"] == "last 14 days"
    assert two_weeks["start_utc"] == "2025-03-01T12:00:01Z"
    # Months resolve on the local calendar rather than a fixed 30-day span.
    assert three_months["label"] == "last 3 months"
    assert three_months["start_utc"].startswith("2024-12")
    assert three_months["start_utc"] < three_months["end_utc"]
    # An unbounded literal is clamped instead of overflowing.
    clamped = policy.gmail_date_window("my emails from the last 999999 days", now=now)
    assert clamped["start_utc"] < clamped["end_utc"]
    assert policy.gmail_date_window("my emails", now=now) is None


def test_relative_time_words_alone_do_not_create_an_email_lookup():
    for message in (
        "I ran every day for the last 7 days.",
        "The last 14 days have been busy.",
        "Today was a pretty exhausting day.",
        "I was thinking about changing jobs.",
        "my message to the team was clear",
    ):
        assert not policy.gmail_relevance_gate(message), message


def test_personal_memory_questions_are_never_email_lookups():
    for message in (
        "What do you remember about my preferences for meetings?",
        "What do you know about me?",
        "What do you remember about my work?",
        "Who am I and what are my goals?",
    ):
        result = policy.classify(message)
        assert not result.retrieve_external_evidence, message
        assert not policy.gmail_relevance_gate(message), message


def test_sender_extraction_is_not_confused_by_relative_time_phrases():
    assert policy.gmail_sender_constraint("my emails from the past 7 days") is None
    assert policy.gmail_sender_constraint("show me my emails from last week") is None
    assert policy.gmail_sender_constraint("any recent emails from Google?") == "Google"
    assert policy.gmail_sender_constraint("emails from Google in the last 7 days") == "Google"
    assert policy.gmail_sender_constraint("Did I get anything from jobs@example.test recently?") == "jobs@example.test"


_JOB_OUTREACH_QUERIES = (
    "Hey, can you tell me, looking at my emails, have I received any email from HR or "
    "recruiter reaching out to me for jobs recently?",
    "have I received any email from HR or recruiter reaching out to me for jobs recently?",
    "has any recruiter reached out to me recently?",
    "did any recruiter email me recently?",
    "any emails from recruiters or HR lately?",
    "did I get any job-related emails recently?",
    "have I received anything from a recruiter recently?",
    "any recruiter reaching out for jobs in my emails?",
)


def test_job_outreach_questions_are_private_gmail_lookups_never_web():
    for message in _JOB_OUTREACH_QUERIES:
        result = policy.classify(message)
        assert policy.gmail_relevance_gate(message), message
        assert result.retrieve_external_evidence, message
        assert result.retrieve_evidence, message
        assert not result.web_needed, message


def test_descriptive_outreach_clauses_are_not_treated_as_senders():
    # A category clause after "from" must not become an author filter, which
    # would silently hide every matching email.
    for message in _JOB_OUTREACH_QUERIES:
        assert policy.gmail_sender_constraint(message) is None, message
    assert policy.gmail_sender_constraint(
        "any emails from recruiters or HR lately?"
    ) is None


def test_lately_is_a_relative_time_window():
    now = datetime(2025, 3, 15, 12, 0, tzinfo=timezone.utc)
    window = policy.gmail_date_window("any emails from recruiters or HR lately?", now=now)
    assert window is not None
    assert window["label"] == "lately"
    assert window["start_utc"] < window["end_utc"]


def test_external_lookup_guidance_forbids_claiming_no_email_access():
    guidance = policy.classify(_JOB_OUTREACH_QUERIES[0]).guidance
    assert "read-only access" in guidance
    assert "Never claim you cannot access the user's email account" in guidance
    assert "found no matching messages" in guidance


