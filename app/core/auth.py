"""Bearer-token authentication for the HTTP API.

Fails closed. If ``JARVIS_TOKEN`` is unset, every ``/api/*`` request is
rejected rather than waved through: the store behind these routes holds real
personal data, so "auth not configured" must never mean "auth skipped".

The token is compared in constant time and is never echoed back, logged, or
included in any error body.
"""
from __future__ import annotations

import hmac
import logging
import re
from typing import Optional

from fastapi import HTTPException, Request

from . import config

BEARER = "bearer"
API_PREFIX = "/api"

# Query parameters whose values must never reach a log file.
_SECRET_PARAMS = ("token", "access_token", "api_key", "secret", "apikey")
_SECRET_QUERY = re.compile(
    r"([?&](?:%s)=)[^&\s\"]*" % "|".join(_SECRET_PARAMS), re.IGNORECASE
)


def unauthorized() -> HTTPException:
    """A single opaque 401. The detail never says which part was wrong."""
    return HTTPException(
        status_code=401,
        detail="unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


def extract_bearer(header: Optional[str]) -> Optional[str]:
    """Pull the credential out of an Authorization header.

    Returns None for anything malformed: wrong scheme, no scheme, no
    credential, or extra tokens.

    RFC 7235 separates the scheme from the credential with 1*SP, so the split
    is on spaces only. Splitting on generic whitespace would also accept tabs,
    which is a parser differential: a proxy that normalises tabs differently
    could disagree with us about what a request contains.
    """
    if not header:
        return None
    fields = header.strip().split(" ")
    if len(fields) != 2:
        return None
    scheme, credential = fields
    if scheme.lower() != BEARER:
        return None
    if not credential or any(ch.isspace() for ch in credential):
        return None
    return credential


def tokens_match(candidate: Optional[str], expected: Optional[str]) -> bool:
    """Constant-time compare. Returns False when either side is missing."""
    if not candidate or not expected:
        return False
    # Compare bytes so a non-ASCII token cannot raise inside compare_digest.
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def is_api_path(path: str) -> bool:
    return path == API_PREFIX or path.startswith(API_PREFIX + "/")


def require_token(request: Request) -> None:
    """Raise 401 unless the request carries the correct bearer token."""
    # A token in the query string is refused outright rather than ignored: URLs
    # get written to access logs and forwarded in Referer headers, so accepting
    # one here would quietly reopen the leak the header requirement avoids.
    if "token" in request.query_params:
        raise unauthorized()

    supplied = extract_bearer(request.headers.get("authorization"))
    if not tokens_match(supplied, config.token()):
        raise unauthorized()


def redact_secrets(text: str) -> str:
    """Blank out the value of any secret-looking query parameter."""
    return _SECRET_QUERY.sub(r"\1[REDACTED]", text)


class RedactSecretsFilter(logging.Filter):
    """Strip secret query values from uvicorn access logs before they are written.

    Rejecting ``/api/*?token=...`` inside the request handler is necessary but
    not sufficient on its own. uvicorn writes the whole request line, including
    the query string, to its access log while the response is still being
    built -- before the application is consulted at all. So a client that puts
    the token in the URL gets a 401 and the token still lands on disk in
    plaintext. Redacting at the logging layer is what actually closes it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        # uvicorn's access record is (client_addr, method, path_with_query,
        # http_version, status_code) -- index 2 is the one that can leak.
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str):
            cleaned = redact_secrets(args[2])
            if cleaned != args[2]:
                record.args = (*args[:2], cleaned, *args[3:])
        if isinstance(record.msg, str):
            record.msg = redact_secrets(record.msg)
        return True


def install_access_log_redaction() -> None:
    """Attach the redaction filter to uvicorn's access logger.

    Idempotent. ``dictConfig`` only adds the filters named in its own config
    and does not clear ones already attached, so a filter installed at startup
    survives uvicorn's logging setup.
    """
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactSecretsFilter) for f in access.filters):
        access.addFilter(RedactSecretsFilter())