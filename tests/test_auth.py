"""Bearer-token auth on /api/* (Phase 1B/1D).

These tests assert the security boundary, so they are deliberately blunt about
what must NOT pass: no header, wrong token, wrong scheme, malformed headers,
query-string tokens, and the unset-token case.
"""
import logging

import pytest
from fastapi.testclient import TestClient

from app.core import auth
from app.server import app

TOKEN = "test-token-0123456789abcdef"

AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TOKEN", TOKEN)
    # Hermetic store: the suite must never open the real data/jarvis.db, which
    # holds actual personal data. config.* reads the environment per request,
    # so redirecting here takes effect immediately.
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def anonymous(monkeypatch, tmp_path):
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    return TestClient(app, raise_server_exceptions=False)


# --- accepted ----------------------------------------------------------------

def test_valid_bearer_token_is_accepted(client):
    assert client.get("/api/stats", headers=AUTH).status_code == 200


def test_bearer_scheme_is_case_insensitive(client):
    # RFC 7235: the scheme token is case-insensitive.
    r = client.get("/api/stats", headers={"Authorization": f"bearer {TOKEN}"})
    assert r.status_code == 200


def test_health_and_root_are_not_gated(client):
    """Only /api/* is protected; the shell and health check must stay reachable."""
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200


# --- rejected ----------------------------------------------------------------

def test_missing_token_is_401(client):
    assert client.get("/api/stats").status_code == 401


def test_invalid_token_is_401(client):
    r = client.get("/api/stats", headers={"Authorization": "Bearer wrong-token"})
    assert r.status_code == 401


def test_token_that_is_a_prefix_of_the_real_one_is_401(client):
    """A truncated token must not pass."""
    assert client.get("/api/stats", headers={"Authorization": f"Bearer {TOKEN[:8]}"}).status_code == 401
    assert client.get("/api/stats", headers={"Authorization": f"Bearer {TOKEN}x"}).status_code == 401


@pytest.mark.parametrize("header", [
    "",                              # empty
    "Bearer",                        # no credential
    "Bearer ",                       # blank credential
    "Basic " + TOKEN,                # wrong scheme
    "Token " + TOKEN,                # wrong scheme
    TOKEN,                           # no scheme at all
    f"Bearer {TOKEN} extra",         # extra tokens
    f"Basic {TOKEN} Bearer",         # confused ordering
    f"{TOKEN} Bearer",               # reversed
    f"Bearer\t{TOKEN}",              # tab-separated, not a valid single token
])
def test_malformed_authorization_header_is_401(client, header):
    assert client.get("/api/stats", headers={"Authorization": header}).status_code == 401


@pytest.mark.parametrize("path", ["/api/stats", "/api/search?q=jarvis", "/api/recent", "/api/show/1"])
def test_every_api_route_requires_the_token(client, path):
    assert client.get(path).status_code == 401


def test_write_routes_are_also_protected(client):
    """Auth must gate mutation, not just reads."""
    r = client.post("/api/add", json={"text": "should not be stored"})
    assert r.status_code == 401


# --- query-string tokens are refused -----------------------------------------

def test_query_string_token_is_rejected_even_with_valid_header(client):
    """A correct header does not excuse a token leaked into the URL."""
    r = client.get(f"/api/stats?token={TOKEN}", headers=AUTH)
    assert r.status_code == 401


def test_query_string_token_is_rejected_without_header(client):
    assert client.get(f"/api/stats?token={TOKEN}").status_code == 401


def test_query_string_token_on_write_route_is_rejected(client):
    r = client.post(f"/api/add?token={TOKEN}", json={"text": "nope"}, headers=AUTH)
    assert r.status_code == 401


def test_unrelated_query_params_are_fine(client):
    """Only 'token' is special; normal parameters must not be collaterally blocked."""
    assert client.get("/api/search?q=memory&limit=5", headers=AUTH).status_code == 200


# --- fail closed -------------------------------------------------------------

def test_unset_token_refuses_all_api_requests(anonymous):
    """No configured token must mean locked, not open."""
    assert anonymous.get("/api/stats").status_code == 401
    assert anonymous.get("/api/stats", headers=AUTH).status_code == 401
    assert anonymous.post("/api/add", json={"text": "x"}, headers=AUTH).status_code == 401


def test_blank_token_env_is_treated_as_unset(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TOKEN", "   ")
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "auth.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/api/stats", headers=AUTH).status_code == 401


# --- leak checks -------------------------------------------------------------

def test_401_body_never_echoes_the_token(client):
    body = client.get("/api/stats", headers={"Authorization": f"Bearer {TOKEN}wrong"}).text
    assert TOKEN not in body
    assert "wrong" not in body or TOKEN not in body


def test_401_detail_is_opaque(client):
    """The response must not reveal whether the token was missing or wrong."""
    missing = client.get("/api/stats").json()
    wrong = client.get("/api/stats", headers={"Authorization": "Bearer nope"}).json()
    assert missing == wrong == {"detail": "unauthorized"}


def test_401_advertises_the_scheme(client):
    r = client.get("/api/stats")
    assert r.headers.get("www-authenticate") == "Bearer"


# --- CORS stays closed --------------------------------------------------------

def test_no_cors_headers_are_emitted(client):
    """A preflight must not be answered with an allow-origin header."""
    r = client.options("/api/stats", headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "GET",
    })
    assert r.headers.get("access-control-allow-origin") is None


def test_cross_origin_get_gets_no_allow_origin(client):
    r = client.get("/api/stats", headers={**AUTH, "Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") is None


# --- log redaction ------------------------------------------------------------

def test_secret_query_values_are_redacted():
    assert auth.redact_secrets("/api/stats?token=supersecret") == \
        "/api/stats?token=[REDACTED]"
    assert "supersecret" not in auth.redact_secrets("/api/stats?token=supersecret")


@pytest.mark.parametrize("param", ["token", "access_token", "api_key", "secret"])
def test_all_secret_params_are_redacted(param):
    out = auth.redact_secrets(f"/api/x?{param}=leakme")
    assert "leakme" not in out


def test_only_the_secret_value_is_blanked_not_the_whole_query():
    """Useful params must survive so the access log stays diagnostic."""
    out = auth.redact_secrets("/api/search?q=memory&token=leakme&limit=5")
    assert "q=memory" in out and "limit=5" in out
    assert "leakme" not in out


def test_redaction_is_case_insensitive():
    assert "leakme" not in auth.redact_secrets("/api/x?Token=leakme")


def test_access_log_filter_rewrites_the_path_argument():
    """This is the actual leak: uvicorn logs the full request line itself."""
    rec = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d',
        ("127.0.0.1", "GET", "/api/stats?token=supersecret", "1.1", 401),
        None,
    )
    auth.RedactSecretsFilter().filter(rec)
    rendered = rec.getMessage()
    assert "supersecret" not in rendered
    assert "/api/stats" in rendered, "the path should still be readable"
    assert "401" in rendered


def test_filter_tolerates_unexpected_record_shapes():
    """A logging filter that raises would take the server down."""
    f = auth.RedactSecretsFilter()
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                            "plain message", None, None)
    assert f.filter(rec) is True
    rec2 = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                             "short %s", ("only-one-arg",), None)
    assert f.filter(rec2) is True


def test_installer_is_idempotent():
    auth.install_access_log_redaction()
    auth.install_access_log_redaction()
    access = logging.getLogger("uvicorn.access")
    n = sum(isinstance(f, auth.RedactSecretsFilter) for f in access.filters)
    assert n == 1


def test_app_installs_redaction_on_startup():
    """The filter must actually be wired into the served app, not just defined."""
    with TestClient(app):
        access = logging.getLogger("uvicorn.access")
        assert any(isinstance(f, auth.RedactSecretsFilter) for f in access.filters)