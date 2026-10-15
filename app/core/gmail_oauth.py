"""Local installed-app OAuth for the Gmail read-only API.

OAuth material is held in a permission-restricted local token file, never in
SQLite or provider records. This module deliberately uses only the stdlib.
"""
from __future__ import annotations

import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from app.core import config

GMAIL_READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class GmailConfigurationError(RuntimeError):
    """Safe-to-display Gmail setup or local OAuth state error."""


class GmailOAuthError(RuntimeError):
    """OAuth failure with credential/provider response details suppressed."""


@dataclass(frozen=True)
class GmailOAuthConfig:
    client_id: str
    client_secret: str = field(repr=False)
    token_file: Path = field(repr=False)

    @classmethod
    def from_env(cls, environ=None) -> "GmailOAuthConfig":
        env = os.environ if environ is None else environ
        client_id = (env.get("JARVIS_GMAIL_CLIENT_ID") or "").strip()
        client_secret = (env.get("JARVIS_GMAIL_CLIENT_SECRET") or "").strip()
        secret_file = (env.get("JARVIS_GMAIL_CLIENT_SECRETS_FILE") or "").strip()
        if bool(client_id) != bool(client_secret):
            raise GmailConfigurationError(
                "Set both JARVIS_GMAIL_CLIENT_ID and JARVIS_GMAIL_CLIENT_SECRET, or configure a client-secrets file."
            )
        if secret_file:
            if client_id or client_secret:
                raise GmailConfigurationError("Configure Gmail OAuth using either environment credentials or a client-secrets file.")
            path = Path(secret_file).expanduser()
            if not path.is_absolute():
                path = config.BASE / path
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                installed = payload.get("installed")
                client_id = installed["client_id"]
                client_secret = installed["client_secret"]
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                raise GmailConfigurationError("Gmail OAuth client-secrets file is missing or invalid.") from None
        if not client_id or not client_secret:
            raise GmailConfigurationError(
                "Gmail OAuth is not configured. Set JARVIS_GMAIL_CLIENT_SECRETS_FILE or both Gmail client environment variables."
            )
        token_file = Path((env.get("JARVIS_GMAIL_TOKEN_FILE") or "data/gmail_oauth_token.json")).expanduser()
        if not token_file.is_absolute():
            token_file = config.BASE / token_file
        return cls(client_id=client_id, client_secret=client_secret, token_file=token_file)


class GmailOAuth:
    def __init__(self, oauth_config: GmailOAuthConfig, *, opener: Callable = urllib.request.urlopen,
                 browser_open: Callable = webbrowser.open, callback_timeout: float = 180):
        self.config = oauth_config
        self._open = opener
        self._browser_open = browser_open
        self.callback_timeout = callback_timeout
        self._token_lock = threading.RLock()

    def _read_token(self) -> dict | None:
        path = self.config.token_file
        try:
            if path.is_symlink():
                raise GmailConfigurationError("Gmail token file must not be a symbolic link.")
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except GmailConfigurationError:
            raise
        except (OSError, ValueError, TypeError):
            raise GmailConfigurationError("Local Gmail OAuth token file is unreadable or invalid.") from None
        if not isinstance(data, dict) or not isinstance(data.get("refresh_token"), str) or not data["refresh_token"]:
            raise GmailConfigurationError("Local Gmail OAuth token file is invalid; remove it and authorize again.")
        if data.get("access_token") is not None and not isinstance(data.get("access_token"), str):
            raise GmailConfigurationError("Local Gmail OAuth token file is invalid; remove it and authorize again.")
        try:
            float(data.get("expires_at", 0))
        except (TypeError, ValueError, OverflowError):
            raise GmailConfigurationError("Local Gmail OAuth token file is invalid; remove it and authorize again.") from None
        scopes = data.get("scope", "")
        if GMAIL_READONLY_SCOPE not in scopes.split():
            raise GmailConfigurationError("Saved Gmail authorization lacks the required read-only scope; authorize again.")
        return data

    def _write_token(self, data: dict) -> None:
        path = self.config.token_file
        if path.is_symlink():
            raise GmailConfigurationError("Gmail token file must not be a symbolic link.")
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                path.parent.chmod(0o700)
            except OSError:
                pass
            temp = path.with_name(path.name + ".tmp-" + secrets.token_hex(6))
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            fd = os.open(temp, flags, 0o600)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(data, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, path)
                path.chmod(0o600)
            finally:
                try:
                    temp.unlink()
                except FileNotFoundError:
                    pass
        except OSError:
            raise GmailConfigurationError("Could not securely save local Gmail authorization. Check token-file permissions.") from None

    @staticmethod
    def _response_json(response) -> dict:
        try:
            raw = response.read()
            result = json.loads(raw.decode("utf-8"))
            if not isinstance(result, dict):
                raise ValueError
            return result
        except Exception:
            raise GmailOAuthError("Google OAuth returned an invalid response.") from None

    def _exchange(self, values: dict) -> dict:
        body = urllib.parse.urlencode(values).encode("ascii")
        request = urllib.request.Request(TOKEN_URL, data=body, headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with self._open(request, timeout=20) as response:
                return self._response_json(response)
        except GmailOAuthError:
            raise
        except Exception:
            raise GmailOAuthError("Google OAuth token exchange failed; provider details were withheld.") from None

    def _authorize_interactively(self) -> dict:
        state = secrets.token_urlsafe(24)
        result: dict[str, str] = {}
        received = threading.Event()

        class CallbackHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(handler):
                parsed = urllib.parse.urlparse(handler.path)
                query = urllib.parse.parse_qs(parsed.query)
                if parsed.path != "/oauth2callback" or query.get("state", [None])[0] != state:
                    handler.send_error(400)
                    return
                result["code"] = query.get("code", [""])[0]
                result["error"] = query.get("error", [""])[0]
                handler.send_response(200 if result.get("code") else 400)
                handler.send_header("Content-Type", "text/plain; charset=utf-8")
                handler.end_headers()
                handler.wfile.write(b"Jarvis Gmail authorization received. You may close this window.")
                received.set()

            def log_message(handler, fmt, *args):
                return  # callback query may contain a one-time authorization code

        server = http.server.HTTPServer(("127.0.0.1", 0), CallbackHandler)
        server.timeout = 1
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        redirect = f"http://127.0.0.1:{server.server_port}/oauth2callback"
        query = urllib.parse.urlencode({
            "client_id": self.config.client_id, "redirect_uri": redirect,
            "response_type": "code", "scope": GMAIL_READONLY_SCOPE,
            "access_type": "offline", "prompt": "consent", "state": state,
        })
        try:
            if not self._browser_open(AUTH_URL + "?" + query):
                raise GmailOAuthError("Could not open the browser for Gmail authorization.")
            if not received.wait(self.callback_timeout):
                raise GmailOAuthError("Gmail authorization timed out.")
            if not result.get("code"):
                raise GmailOAuthError("Gmail authorization was declined or failed.")
            return self._exchange({
                "client_id": self.config.client_id, "client_secret": self.config.client_secret,
                "code": result["code"], "grant_type": "authorization_code", "redirect_uri": redirect,
            })
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def access_token(self, *, force_refresh: bool = False) -> str:
        with self._token_lock:
            token = self._read_token()
            if token and not force_refresh and float(token.get("expires_at", 0)) > time.time() + 60 and token.get("access_token"):
                return token["access_token"]
            if token:
                refreshed = self._exchange({
                    "client_id": self.config.client_id, "client_secret": self.config.client_secret,
                    "refresh_token": token["refresh_token"], "grant_type": "refresh_token",
                })
                try:
                    expires_in = int(refreshed.get("expires_in", 3600))
                except (TypeError, ValueError, OverflowError):
                    raise GmailOAuthError("Google OAuth returned an invalid token lifetime.") from None
                fresh = {**token, **refreshed, "refresh_token": token["refresh_token"],
                         "expires_at": time.time() + expires_in,
                         "scope": refreshed.get("scope", token.get("scope", ""))}
            else:
                fresh = self._authorize_interactively()
                if not fresh.get("refresh_token"):
                    raise GmailOAuthError("Google did not return offline authorization; revoke Jarvis access and authorize again.")
                try:
                    expires_in = int(fresh.get("expires_in", 3600))
                except (TypeError, ValueError, OverflowError):
                    raise GmailOAuthError("Google OAuth returned an invalid token lifetime.") from None
                fresh["expires_at"] = time.time() + expires_in
                fresh["scope"] = fresh.get("scope", GMAIL_READONLY_SCOPE)
            if GMAIL_READONLY_SCOPE not in str(fresh.get("scope", "")).split():
                raise GmailOAuthError("Google authorization did not grant the required read-only scope.")
            self._write_token(fresh)
            access = fresh.get("access_token")
            if not isinstance(access, str) or not access:
                raise GmailOAuthError("Google OAuth response did not contain an access token.")
            return access
