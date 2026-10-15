"""Read-only Gmail connector for Jarvis's provider-neutral source pipeline.

Message text is normalized as source evidence only. This module never calls
memory extraction and never logs request bodies, headers, or API responses.
"""
from __future__ import annotations

import base64
import email
import email.policy
import email.utils
import html.parser
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Callable
import unicodedata

from app.core.external_sources import MAX_NORMALIZED_TEXT_CHARS, NormalizedExternalRecord, SyncPage
from app.core.gmail_oauth import GmailOAuth, GmailOAuthConfig
from app.core.gmail_sync_state import GmailFullSyncRequired, GmailSyncStateRepository, validate_history_id

GMAIL_API_ROOT = "https://gmail.googleapis.com/gmail/v1/users/me"
_TRUNCATION_MARKER = (
    "\n\n[Jarvis note: email body truncated at the local size limit; only the prefix is stored.]"
)


class GmailAPIError(RuntimeError):
    """A Gmail API failure that intentionally excludes response contents."""

    def __init__(self, status: int | None, message: str = "Gmail API request failed"):
        self.status = status
        super().__init__(f"{message} (HTTP {status})" if status is not None else message)


class GmailHistoryExpired(GmailFullSyncRequired):
    """The requested History API checkpoint is no longer available."""


class GmailAPIClient:
    def __init__(self, oauth: GmailOAuth, *, opener: Callable = urllib.request.urlopen,
                 sleeper: Callable[[float], None] = time.sleep, retries: int = 3):
        self.oauth = oauth
        self._open = opener
        self._sleep = sleeper
        self.retries = max(0, min(int(retries), 5))

    def _get(self, path: str, params: dict | None = None) -> dict:
        url = GMAIL_API_ROOT + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        refreshed = False
        attempts = 0
        while True:
            token = self.oauth.access_token(force_refresh=refreshed)
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token, "Accept": "application/json"})
            try:
                with self._open(req, timeout=30) as response:
                    result = json.loads(response.read().decode("utf-8"))
                if not isinstance(result, dict):
                    raise GmailAPIError(None, "Gmail API returned an invalid response")
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    exc.close()
                except Exception:
                    pass
                if status == 401 and not refreshed:
                    refreshed = True
                    continue
                if status in (408, 429, 500, 502, 503, 504) and attempts < self.retries:
                    self._sleep(min(2 ** attempts, 8))
                    attempts += 1
                    continue
                raise GmailAPIError(status) from None
            except GmailAPIError:
                raise
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempts < self.retries:
                    self._sleep(min(2 ** attempts, 8))
                    attempts += 1
                    continue
                raise GmailAPIError(None, "Gmail API is unavailable") from None
            except (UnicodeError, ValueError, TypeError):
                raise GmailAPIError(None, "Gmail API returned malformed data") from None

    def list_messages(self, *, page_token: str | None = None, max_results: int = 100, query: str | None = None) -> dict:
        params: dict[str, str | int] = {"maxResults": max(1, min(int(max_results), 500))}
        if page_token:
            params["pageToken"] = page_token
        if query:
            params["q"] = query
        return self._get("/messages", params)

    def get_message(self, message_id: str) -> dict:
        if not isinstance(message_id, str) or not message_id or len(message_id) > 512 or "/" in message_id:
            raise ValueError("invalid Gmail message id")
        return self._get("/messages/" + urllib.parse.quote(message_id, safe=""), {"format": "full"})

    def get_profile(self) -> dict:
        return self._get("/profile")

    def list_history(self, start_history_id: str, *, page_token: str | None = None,
                     max_results: int = 100) -> dict:
        params: dict[str, str | int] = {
            "startHistoryId": validate_history_id(start_history_id),
            "maxResults": max(1, min(int(max_results), 500)),
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            return self._get("/history", params)
        except GmailAPIError as exc:
            if exc.status == 404:
                raise GmailHistoryExpired("Gmail history checkpoint expired; run a full sync to recover") from None
            raise


class _VisibleHTML(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag.casefold() in {"script", "style", "head", "title"}:
            self.hidden += 1
        elif tag.casefold() in {"p", "div", "br", "li", "tr", "h1", "h2", "h3"} and self.hidden == 0:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag.casefold() in {"script", "style", "head", "title"} and self.hidden:
            self.hidden -= 1
        elif tag.casefold() in {"p", "div", "li", "tr"} and self.hidden == 0:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.hidden == 0:
            self.parts.append(data)


def _html_to_text(value: str) -> str:
    parser = _VisibleHTML()
    try:
        parser.feed(value)
        parser.close()
        text = "".join(parser.parts)
    except Exception:
        text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1\s*>", " ", value)
        text = re.sub(r"(?s)<[^>]*>", " ", text)
    return re.sub(r"[ \t\f\v]+", " ", text).strip()


def _decode_body(data: str, mime_type: str, headers: dict) -> str:
    try:
        raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    except (ValueError, TypeError):
        try:
            raw = base64.b64decode(data + "=" * (-len(data) % 4), validate=False)
        except Exception:
            return ""
    charset = "utf-8"
    try:
        content_type = headers.get("Content-Type", "")
        if content_type:
            charset = email.message_from_string("Content-Type: " + content_type + "\n\n").get_content_charset() or charset
    except Exception:
        pass
    text = raw.decode(charset, errors="replace")
    return _html_to_text(text) if mime_type.casefold() == "text/html" else text


def _payload_text(payload: dict) -> str:
    plain: list[str] = []
    html: list[str] = []

    def walk(part):
        if not isinstance(part, dict):
            return
        filename = part.get("filename")
        body = part.get("body") if isinstance(part.get("body"), dict) else {}
        # Attachments are not imported as email body content.
        if filename or body.get("attachmentId"):
            return
        mime = str(part.get("mimeType", "")).casefold()
        data = body.get("data")
        headers = {str(h.get("name", "")).casefold(): h.get("value", "")
                   for h in part.get("headers", []) if isinstance(h, dict)}
        headers = {"Content-Type": headers.get("content-type", "")}
        if isinstance(data, str):
            decoded = _decode_body(data, mime, headers)
            if mime == "text/plain" and decoded:
                plain.append(decoded)
            elif mime == "text/html" and decoded:
                html.append(decoded)
        children = part.get("parts", [])
        if isinstance(children, list):
            for child in children:
                walk(child)

    walk(payload)
    return "\n\n".join(plain if plain else html).strip()


def _headers(payload: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in payload.get("headers", []) if isinstance(payload.get("headers", []), list) else []:
        if isinstance(item, dict) and isinstance(item.get("name"), str):
            key = item["name"].casefold()
            if key in {"from", "to", "cc", "subject", "date", "message-id"}:
                result[key] = str(item.get("value", ""))[:4096]
    return result


def _timestamp(message: dict, headers: dict) -> str | None:
    internal = message.get("internalDate")
    try:
        dt = datetime.fromtimestamp(int(internal) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        try:
            parsed = email.utils.parsedate_to_datetime(headers.get("date", ""))
            if parsed is None:
                return None
            dt = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
        except (TypeError, ValueError, OverflowError):
            return None
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _bounded_body(body: str) -> tuple[str, bool, int | None]:
    """Normalize deterministically and fit the shared external-text boundary."""
    normalized = unicodedata.normalize("NFC", body.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
    if len(normalized) <= MAX_NORMALIZED_TEXT_CHARS:
        return normalized, False, None
    original_length = len(normalized)
    prefix_length = MAX_NORMALIZED_TEXT_CHARS - len(_TRUNCATION_MARKER)
    prefix = normalized[:prefix_length].rstrip()
    # Whitespace trimming may leave space, but enforce the contract exactly.
    bounded = (prefix + _TRUNCATION_MARKER)[:MAX_NORMALIZED_TEXT_CHARS]
    return bounded, True, original_length


def normalize_gmail_message(message: dict) -> NormalizedExternalRecord:
    if not isinstance(message, dict):
        raise ValueError("Gmail message must be an object")
    msg_id, thread_id = message.get("id"), message.get("threadId")
    if not isinstance(msg_id, str) or not msg_id.strip() or len(msg_id) > 512:
        raise ValueError("Gmail message is missing a valid id")
    headers = _headers(message.get("payload") if isinstance(message.get("payload"), dict) else {})
    subject = headers.get("subject", "").strip()
    body = _payload_text(message.get("payload", {}))
    if not body:
        snippet = message.get("snippet")
        body = snippet.strip() if isinstance(snippet, str) else ""
    if not body:
        body = "[No readable message body]"
    body, body_truncated, original_body_length = _bounded_body(body)
    occurred = _timestamp(message, headers)
    metadata = {
        "thread_id": thread_id[:512] if isinstance(thread_id, str) else None,
        "recipients": {key: headers[key] for key in ("to", "cc") if headers.get(key)},
        "label_ids": [str(x)[:128] for x in message.get("labelIds", [])[:100] if isinstance(x, str)]
        if isinstance(message.get("labelIds", []), list) else [],
        "internal_date_ms": str(message.get("internalDate")) if str(message.get("internalDate", "")).isdigit() else None,
        "body_truncated": body_truncated,
    }
    if original_body_length is not None:
        metadata["body_original_char_count"] = original_body_length
    return NormalizedExternalRecord(
        provider_type="gmail", external_id=msg_id.strip(), title=subject, occurred_at_utc=occurred,
        source_created_at_utc=occurred, source_updated_at_utc=occurred,
        author=headers.get("from"), normalized_text=body,
        metadata=metadata,
        source_url="https://mail.google.com/mail/u/0/#all/" + urllib.parse.quote(msg_id.strip(), safe=""),
        provenance={"connector": "gmail-readonly-v1", "provider": "gmail"},
    )


class GmailConnector:
    """Page-based Gmail adapter implementing ExternalSourceConnector."""

    provider_type = "gmail"

    def __init__(self, client: GmailAPIClient, *, query: str | None = None, page_size: int = 100):
        self.client = client
        self.query = query
        self.page_size = max(1, min(int(page_size), 500))

    def fetch_page(self, cursor: str | None = None) -> SyncPage:
        listing = self.client.list_messages(page_token=cursor, max_results=self.page_size, query=self.query)
        messages = listing.get("messages", [])
        if not isinstance(messages, list):
            raise GmailAPIError(None, "Gmail API returned malformed message listing")
        records: list[NormalizedExternalRecord] = []
        seen: set[str] = set()
        for item in messages:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in seen:
                continue
            message_id = item["id"]
            seen.add(message_id)
            try:
                message = self.client.get_message(message_id)
            except GmailAPIError as exc:
                if exc.status == 404:  # deleted between list and fetch
                    continue
                raise
            if message.get("id") != message_id:
                continue
            records.append(normalize_gmail_message(message))
        next_cursor = listing.get("nextPageToken")
        if next_cursor is not None and (not isinstance(next_cursor, str) or len(next_cursor) > 4096):
            raise GmailAPIError(None, "Gmail API returned an invalid pagination cursor")
        return SyncPage(tuple(records), next_cursor)


def sync_gmail(connector: GmailConnector, repository, *, max_pages: int | None = None,
               require_complete: bool = False) -> dict[str, int]:
    """Explicitly fetch and ingest pages; cursor lives only for this invocation."""
    from app.core.external_sources import ingest_connector_page

    if max_pages is not None and (isinstance(max_pages, bool) or not isinstance(max_pages, int) or max_pages < 1):
        raise ValueError("max_pages must be a positive integer")
    cursor = None
    pages = inserted = updated = unchanged = 0
    while max_pages is None or pages < max_pages:
        summary = ingest_connector_page(connector, repository, cursor=cursor)
        pages += 1
        inserted += summary.inserted
        updated += summary.updated
        unchanged += summary.unchanged
        cursor = summary.next_cursor
        if cursor is None:
            break
    if require_complete and cursor is not None:
        raise RuntimeError("Gmail full sync stopped before all pages completed; no checkpoint was saved")
    return {"pages": pages, "inserted": inserted, "updated": updated, "unchanged": unchanged}


def _message_ids_from_history(history: object) -> set[str]:
    if not isinstance(history, list):
        raise GmailAPIError(None, "Gmail History API returned malformed data")
    ids: set[str] = set()
    for item in history:
        if not isinstance(item, dict):
            raise GmailAPIError(None, "Gmail History API returned malformed data")
        candidates = []
        candidates.extend(item.get("messages", []) if isinstance(item.get("messages", []), list) else [])
        for event_name in ("messagesAdded", "messagesDeleted", "labelsAdded", "labelsRemoved"):
            events = item.get(event_name, [])
            if isinstance(events, list):
                candidates.extend(event.get("message") for event in events if isinstance(event, dict))
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            message_id = candidate.get("id")
            if isinstance(message_id, str) and message_id and len(message_id) <= 512:
                ids.add(message_id)
    return ids


def _query_matches_message(connector: GmailConnector, message: dict) -> bool:
    """Ask Gmail to apply its own query to one RFC822 Message-ID."""
    if not connector.query:
        return True
    payload = message.get("payload") if isinstance(message.get("payload"), dict) else {}
    message_id = _headers(payload).get("message-id", "").strip()
    if not (message_id.startswith("<") and message_id.endswith(">")):
        raise GmailAPIError(None, "Cannot safely evaluate configured Gmail query for a changed message; run a full sync")
    token = message_id[1:-1]
    if (not token or "@" not in token or len(token) > 998
            or any(ch.isspace() or ch in '"\\(){}' for ch in token)):
        raise GmailAPIError(None, "Cannot safely evaluate configured Gmail query for a changed message; run a full sync")
    result = connector.client.list_messages(
        page_token=None, max_results=10, query=f"({connector.query}) rfc822msgid:{token}"
    )
    rows = result.get("messages", [])
    return isinstance(rows, list) and any(isinstance(row, dict) and row.get("id") == message.get("id") for row in rows)


def _apply_history(connector: GmailConnector, repository, start_history_id: str) -> tuple[dict[str, int], str]:
    """Replay all history pages and materialize each changed message's current state."""
    page_token = None
    ids: set[str] = set()
    latest = validate_history_id(start_history_id)
    pages = 0
    while True:
        page = connector.client.list_history(start_history_id, page_token=page_token)
        pages += 1
        ids.update(_message_ids_from_history(page.get("history", [])))
        latest_page = validate_history_id(page.get("historyId"))
        latest = str(max(int(latest), int(latest_page)))
        next_token = page.get("nextPageToken")
        if next_token is None:
            break
        if not isinstance(next_token, str) or not next_token or len(next_token) > 4096:
            raise GmailAPIError(None, "Gmail History API returned an invalid pagination cursor")
        page_token = next_token

    inserted = updated = unchanged = deleted = 0
    for message_id in sorted(ids):
        try:
            message = connector.client.get_message(message_id)
        except GmailAPIError as exc:
            if exc.status != 404:
                raise
            if repository.mark_deleted("gmail", message_id):
                deleted += 1
            continue
        if message.get("id") != message_id:
            raise GmailAPIError(None, "Gmail API returned an unexpected message")
        if not _query_matches_message(connector, message):
            if repository.mark_deleted("gmail", message_id):
                deleted += 1
            continue
        existing = repository.db.conn.execute(
            "SELECT content_hash FROM external_source_records WHERE provider_type='gmail' AND external_id=?",
            (message_id,),
        ).fetchone()
        result = repository.upsert(normalize_gmail_message(message))
        if not result.changed:
            unchanged += 1
        elif existing:
            updated += 1
        else:
            inserted += 1
    return {"pages": pages, "messages": len(ids), "inserted": inserted, "updated": updated,
            "unchanged": unchanged, "deleted": deleted}, latest


def _profile_identity(connector: GmailConnector) -> tuple[str, str]:
    profile = connector.client.get_profile()
    email_address = profile.get("emailAddress") if isinstance(profile, dict) else None
    history_id = validate_history_id(profile.get("historyId") if isinstance(profile, dict) else None)
    if not isinstance(email_address, str) or not email_address.strip():
        raise GmailAPIError(None, "Gmail profile did not provide an account identity")
    return email_address, history_id


def sync_gmail_full(connector: GmailConnector, repository, state: GmailSyncStateRepository,
                    *, max_pages: int | None = None) -> dict[str, int]:
    """Run full query sync, catch up from its starting history ID, then commit."""
    account, baseline = _profile_identity(connector)
    summary = sync_gmail(connector, repository, max_pages=max_pages, require_complete=True)
    catchup, checkpoint = _apply_history(connector, repository, baseline)
    state.save_success(account, connector.query, checkpoint)
    summary["history_catchup_messages"] = catchup["messages"]
    summary["deleted"] = catchup["deleted"]
    return summary


def sync_gmail_incremental(connector: GmailConnector, repository,
                           state: GmailSyncStateRepository) -> dict[str, int]:
    """Apply Gmail History changes; only commit the cursor after all changes succeed."""
    account, _profile_history = _profile_identity(connector)
    checkpoint = state.get(account, connector.query)
    if checkpoint is None:
        raise GmailFullSyncRequired("No Gmail checkpoint exists for this account/query; run a full sync first")
    summary, next_history = _apply_history(connector, repository, checkpoint.history_id)
    state.save_success(account, connector.query, next_history)
    return summary


def configured_connector(*, oauth_config: GmailOAuthConfig | None = None) -> GmailConnector:
    """Build configured local OAuth/API/connector stack without opening a DB."""
    import os

    oauth = GmailOAuth(oauth_config or GmailOAuthConfig.from_env())
    query = (os.environ.get("JARVIS_GMAIL_QUERY") or "").strip() or None
    return GmailConnector(GmailAPIClient(oauth), query=query)
