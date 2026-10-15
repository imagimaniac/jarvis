"""Provider-neutral local ingestion and repository for external source records.

External records remain source evidence. They are indexed through immutable
``entries`` projections for compatibility with current FTS, recall, and numeric
citations, but are never written to canonical personal memory.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import parse_qsl, urlsplit

from app.core import storage


_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")
MAX_NORMALIZED_TEXT_CHARS = 200_000
_SECRET_KEY_PARTS = (
    "token", "secret", "password", "credential", "authorization", "privatekey",
    "apikey", "bearer", "cookie",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/-]{12,})"),
    re.compile(r"(?i)\b(?:access[_-]?token|refresh[_-]?token|client[_-]?secret|api[_-]?key|password)\s*[:=]\s*([A-Za-z0-9._~+/-]{6,})"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class ExternalSourceValidationError(ValueError):
    """An external record is malformed or contains credential material."""


@dataclass(frozen=True)
class NormalizedExternalRecord:
    provider_type: str
    external_id: str
    normalized_text: str
    title: str = ""
    occurred_at_utc: str | None = None
    source_created_at_utc: str | None = None
    source_updated_at_utc: str | None = None
    author: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_url: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SyncPage:
    records: tuple[NormalizedExternalRecord, ...]
    next_cursor: str | None = None


@dataclass(frozen=True)
class UpsertResult:
    record_id: str
    entry_id: int
    version: int
    changed: bool


@dataclass(frozen=True)
class IngestionSummary:
    inserted: int
    updated: int
    unchanged: int
    next_cursor: str | None


class ExternalSourceConnector(Protocol):
    """A future provider adapter that yields normalized, credential-free records."""

    provider_type: str

    def fetch_page(self, cursor: str | None = None) -> SyncPage: ...


def _has_secret_material(value: str) -> bool:
    for pattern in _SECRET_VALUE_PATTERNS:
        match = pattern.search(value)
        if not match:
            continue
        if not match.lastindex:
            return True
        candidate = match.group(1).strip(".,;:)]}>")
        # Natural language and common documentation placeholders often follow
        # credential labels in email text. Require a plausible opaque value;
        # do not reject a record merely because it discusses OAuth field names.
        looks_opaque = len(candidate) >= 24 or (
            len(candidate) >= 12
            and re.search(r"[A-Za-z]", candidate)
            and re.search(r"[0-9._~+/-]", candidate)
        )
        is_placeholder = re.match(
            r"(?i)^(?:your|example|sample|placeholder|redacted|dummy|fake|replace|insert)(?:[_-]|$)",
            candidate,
        )
        if looks_opaque and not is_placeholder:
            return True
    return False


def _safe_mapping(value: Any, field_name: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, Mapping):
        raise ExternalSourceValidationError(f"{field_name} must be an object")

    def check(node: Any) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise ExternalSourceValidationError(f"{field_name} keys must be text")
                compact = re.sub(r"[^a-z0-9]", "", key.casefold())
                if any(part in compact for part in _SECRET_KEY_PARTS):
                    raise ExternalSourceValidationError(f"{field_name} must not contain credential fields")
                check(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                check(child)
        elif isinstance(node, str):
            if _has_secret_material(node):
                raise ExternalSourceValidationError(f"{field_name} appears to contain credential material")
        elif node is not None and not isinstance(node, (bool, int, float)):
            raise ExternalSourceValidationError(f"{field_name} must contain JSON values")

    result = dict(value)
    check(result)
    try:
        encoded = json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ExternalSourceValidationError(f"{field_name} must be JSON serializable") from None
    if len(encoded.encode("utf-8")) > 65536:
        raise ExternalSourceValidationError(f"{field_name} exceeds the supported size")
    return result, encoded


def _text(value: Any, field_name: str, *, required: bool = False, max_length: int = MAX_NORMALIZED_TEXT_CHARS) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise ExternalSourceValidationError(f"{field_name} must be text")
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    if field_name == "normalized_text":
        normalized = "\n".join(line.rstrip() for line in normalized.splitlines()).strip()
    else:
        normalized = normalized.strip()
    if required and not normalized:
        raise ExternalSourceValidationError(f"{field_name} must not be blank")
    if len(normalized) > max_length:
        raise ExternalSourceValidationError(f"{field_name} exceeds the supported size")
    if _has_secret_material(normalized):
        raise ExternalSourceValidationError(f"{field_name} appears to contain credential material")
    return normalized


def _timestamp(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ExternalSourceValidationError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ExternalSourceValidationError(f"{field_name} must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExternalSourceValidationError(f"{field_name} must include a timezone")
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _source_url(value: Any) -> str | None:
    url = _text(value, "source_url", max_length=2048)
    if url is None or not url:
        return None
    try:
        parts = urlsplit(url)
        # A reference may be a URL or local file URI, but never embedded credentials.
        if parts.scheme and parts.scheme.casefold() not in {"http", "https", "file", "mailto"}:
            raise ExternalSourceValidationError("source_url has an unsupported scheme")
        if parts.username or parts.password:
            raise ExternalSourceValidationError("source_url must not include credentials")
        def is_credential_parameter(key: str) -> bool:
            compact = re.sub(r"[^a-z0-9]", "", key.casefold())
            return compact in {"key", "auth", "code"} or any(
                part in compact for part in (
                    "token", "secret", "password", "authorization", "credential", "apikey", "bearer", "cookie",
                )
            )

        parameters = parse_qsl(parts.query, keep_blank_values=True)
        parameters += parse_qsl(parts.fragment, keep_blank_values=True)
        if any(is_credential_parameter(key) for key, _ in parameters):
            raise ExternalSourceValidationError("source_url must not include credential query parameters")
        if parts.scheme.casefold() in {"http", "https"} and not parts.hostname:
            raise ExternalSourceValidationError("source_url must include a host")
    except ValueError:
        raise ExternalSourceValidationError("source_url is malformed") from None
    return url


def normalize_external_record(value: NormalizedExternalRecord | Mapping[str, Any]) -> NormalizedExternalRecord:
    """Validate and normalize the provider-neutral record contract."""
    if isinstance(value, NormalizedExternalRecord):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise ExternalSourceValidationError("external record must be an object")
    allowed = {
        "provider_type", "external_id", "title", "occurred_at_utc", "source_created_at_utc",
        "source_updated_at_utc", "author", "normalized_text", "metadata", "source_url", "provenance",
    }
    if set(value) - allowed:
        raise ExternalSourceValidationError("external record contains unsupported fields")
    provider = value.get("provider_type")
    external_id = value.get("external_id")
    if not isinstance(provider, str) or not _PROVIDER_RE.fullmatch(provider.strip().casefold()):
        raise ExternalSourceValidationError("provider_type must be a lowercase provider identifier")
    provider = provider.strip().casefold()
    if not isinstance(external_id, str) or not external_id.strip() or len(external_id.strip()) > 512:
        raise ExternalSourceValidationError("external_id must be non-empty text of at most 512 characters")
    external_id = unicodedata.normalize("NFC", external_id.strip())
    if _has_secret_material(external_id):
        raise ExternalSourceValidationError("external_id appears to contain credential material")
    metadata, _metadata_json = _safe_mapping(value.get("metadata", {}), "metadata")
    provenance, _provenance_json = _safe_mapping(value.get("provenance", {}), "provenance")
    return NormalizedExternalRecord(
        provider_type=provider,
        external_id=external_id,
        title=_text(value.get("title", ""), "title", max_length=2000) or "",
        occurred_at_utc=_timestamp(value.get("occurred_at_utc"), "occurred_at_utc"),
        source_created_at_utc=_timestamp(value.get("source_created_at_utc"), "source_created_at_utc"),
        source_updated_at_utc=_timestamp(value.get("source_updated_at_utc"), "source_updated_at_utc"),
        author=_text(value.get("author"), "author", max_length=512),
        normalized_text=_text(value.get("normalized_text"), "normalized_text", required=True) or "",
        metadata=metadata,
        source_url=_source_url(value.get("source_url")),
        provenance=provenance,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _content_hash(record: NormalizedExternalRecord) -> str:
    content = {
        "provider_type": record.provider_type,
        "external_id": record.external_id,
        "title": record.title,
        "occurred_at_utc": record.occurred_at_utc,
        "source_created_at_utc": record.source_created_at_utc,
        "source_updated_at_utc": record.source_updated_at_utc,
        "author": record.author,
        "normalized_text": record.normalized_text,
        "metadata": record.metadata,
        "source_url": record.source_url,
    }
    encoded = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ExternalSourceRepository:
    """SQLite repository with provider/external-ID upsert and immutable versions."""

    def __init__(self, db):
        self.db = db

    @staticmethod
    def stable_record_id(provider_type: str, external_id: str) -> str:
        normalized = normalize_external_record({
            "provider_type": provider_type, "external_id": external_id, "normalized_text": "id-check",
        })
        return str(uuid.uuid5(uuid.NAMESPACE_URL,
                              f"jarvis-external-source:{normalized.provider_type}:{normalized.external_id}"))

    def upsert(self, value: NormalizedExternalRecord | Mapping[str, Any]) -> UpsertResult:
        record = normalize_external_record(value)
        content_hash = _content_hash(record)
        imported = _utc_now()
        provenance_json = _json(record.provenance)
        metadata_json = _json(record.metadata)
        current = self.db.conn.execute(
            "SELECT * FROM external_source_records WHERE provider_type=? AND external_id=?",
            (record.provider_type, record.external_id),
        ).fetchone()
        if current and current["content_hash"] == content_hash:
            merged_provenance = json.loads(current["provenance_json"])
            merged_provenance.update(record.provenance)
            self.db.conn.execute(
                "UPDATE external_source_records SET imported_at_utc=?, provenance_json=?, deleted_at_utc=NULL WHERE id=?",
                (imported, _json(merged_provenance), current["id"]),
            )
            self.db.conn.commit()
            return UpsertResult(current["id"], current["current_entry_id"], current["content_version"], False)

        record_id = current["id"] if current else self.stable_record_id(record.provider_type, record.external_id)
        version = current["content_version"] + 1 if current else 1
        evidence_text = f"{record.title}\n{record.normalized_text}" if record.title else record.normalized_text
        version_origin_id = f"{record_id}:v{version}"
        prior_entry = self.db.conn.execute(
            "SELECT id FROM entries WHERE source=? AND origin_id=?", (record.provider_type, version_origin_id)
        ).fetchone()
        entry_id = prior_entry["id"] if prior_entry else self.db.insert_entry(
            evidence_text,
            source=record.provider_type,
            origin_kind="external_record",
            source_type="external_source",
            origin_id=version_origin_id,
            external_ref=record.external_id,
            content_hash=hashlib.sha256(evidence_text.encode("utf-8")).hexdigest(),
        )
        if not entry_id:
            # INSERT OR IGNORE may be reached after an interrupted prior attempt.
            prior_entry = self.db.conn.execute(
                "SELECT id FROM entries WHERE source=? AND origin_id=?", (record.provider_type, version_origin_id)
            ).fetchone()
            if not prior_entry:
                raise RuntimeError("could not create external evidence projection")
            entry_id = prior_entry["id"]

        fields = (
            record.title, record.occurred_at_utc, record.source_created_at_utc,
            record.source_updated_at_utc, record.author, record.normalized_text,
            metadata_json, record.source_url, imported, content_hash, entry_id,
            provenance_json,
        )
        with self.db.conn:
            if current:
                self.db.conn.execute(
                    "UPDATE external_source_records SET title=?, occurred_at_utc=?, source_created_at_utc=?, "
                    "source_updated_at_utc=?, author=?, normalized_text=?, metadata_json=?, source_url=?, "
                    "imported_at_utc=?, content_hash=?, content_version=?, current_entry_id=?, provenance_json=? "
                    ", deleted_at_utc=NULL WHERE id=?",
                    (*fields[:-2], version, entry_id, provenance_json, record_id),
                )
            else:
                self.db.conn.execute(
                    "INSERT INTO external_source_records "
                    "(id,provider_type,external_id,title,occurred_at_utc,source_created_at_utc,source_updated_at_utc,"
                    "author,normalized_text,metadata_json,source_url,imported_at_utc,content_hash,content_version,"
                    "current_entry_id,provenance_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?)",
                    (record_id, record.provider_type, record.external_id, *fields),
                )
            self.db.conn.execute(
                "INSERT INTO external_source_versions "
                "(record_id,version,content_hash,title,occurred_at_utc,source_created_at_utc,source_updated_at_utc,"
                "author,normalized_text,metadata_json,source_url,imported_at_utc,entry_id,provenance_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record_id, version, content_hash, record.title, record.occurred_at_utc,
                 record.source_created_at_utc, record.source_updated_at_utc, record.author,
                 record.normalized_text, metadata_json, record.source_url, imported, entry_id, provenance_json),
            )
        return UpsertResult(record_id, entry_id, version, True)

    def get(self, record_id: str) -> dict[str, Any] | None:
        row = self.db.conn.execute("SELECT * FROM external_source_records WHERE id=?", (record_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metadata"] = json.loads(result.pop("metadata_json"))
        result["provenance"] = json.loads(result.pop("provenance_json"))
        return result

    def list_versions(self, record_id: str) -> list[dict[str, Any]]:
        rows = self.db.conn.execute(
            "SELECT * FROM external_source_versions WHERE record_id=? ORDER BY version", (record_id,)
        )
        result = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            item["provenance"] = json.loads(item.pop("provenance_json"))
            result.append(item)
        return result

    def mark_deleted(self, provider_type: str, external_id: str, *, deleted_at_utc: str | None = None) -> bool:
        """Hide a source record from current evidence while retaining its versions."""
        normalized_provider = str(provider_type or "").strip().casefold()
        normalized_external_id = str(external_id or "").strip()
        if not normalized_provider or not normalized_external_id:
            raise ExternalSourceValidationError("provider and external id are required")
        stamp = deleted_at_utc or _utc_now()
        changed = self.db.conn.execute(
            "UPDATE external_source_records SET deleted_at_utc=? "
            "WHERE provider_type=? AND external_id=? AND deleted_at_utc IS NULL",
            (stamp, normalized_provider, normalized_external_id),
        ).rowcount
        self.db.conn.commit()
        return bool(changed)


def ingest_connector_page(
    connector: ExternalSourceConnector,
    repository: ExternalSourceRepository,
    *,
    cursor: str | None = None,
) -> IngestionSummary:
    """Fetch one provider page and upsert normalized source records locally."""
    page = connector.fetch_page(cursor)
    if not isinstance(page, SyncPage):
        raise ExternalSourceValidationError("connector must return a SyncPage")
    inserted = updated = unchanged = 0
    for item in page.records:
        record = normalize_external_record(item)
        if record.provider_type != connector.provider_type.strip().casefold():
            raise ExternalSourceValidationError("record provider does not match connector")
        current = repository.db.conn.execute(
            "SELECT content_hash FROM external_source_records WHERE provider_type=? AND external_id=?",
            (record.provider_type, record.external_id),
        ).fetchone()
        result = repository.upsert(record)
        if not result.changed:
            unchanged += 1
        elif current:
            updated += 1
        else:
            inserted += 1
    return IngestionSummary(inserted, updated, unchanged, page.next_cursor)
