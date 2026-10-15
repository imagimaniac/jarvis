"""Static PWA assets and the one-time token bootstrap page.

Assets are served from an explicit allowlist rather than by mounting a
directory. A mount (``app.mount("/", StaticFiles(...))``) would also become a
catch-all for anything not already routed, and a directory mount serves
whatever happens to be sitting in it. The allowlist means a path that is not
named here simply does not exist, so there is no way to walk out of ``static/``
with ``..`` or reach ``config.yaml`` through a crafted URL.
"""
from __future__ import annotations

from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parents[2] / "static"

# filename -> content type. This mapping *is* the routing table.
ASSETS: dict[str, str] = {
    "manifest.json": "application/manifest+json",
    "sw.js": "text/javascript; charset=utf-8",
    "icon-192.png": "image/png",
    "icon-512.png": "image/png",
    "maskable-512.png": "image/png",
    "apple-touch-icon.png": "image/png",
}

# Cache policy per asset. The service worker script is deliberately revalidated
# every time so a new worker can take over; browsers ignore HTTP caching for
# it in many cases anyway, but being explicit avoids a stale worker surviving a
# deploy. Icons are content-stable, so a day is plenty.
_CACHE_CONTROL = {
    "sw.js": "no-cache",
    "manifest.json": "no-cache",
    "icon-192.png": "public, max-age=86400",
    "icon-512.png": "public, max-age=86400",
    "maskable-512.png": "public, max-age=86400",
    "apple-touch-icon.png": "public, max-age=86400",
}

# Where the browser should stash the bearer token. Named once so the server
# never has to agree with the frontend about it.
TOKEN_STORAGE_KEY = "jarvis_token"

# The only place a token may arrive as a query parameter.
BOOTSTRAP_PATH = "/"

# Response headers required on the bootstrap response.
BOOTSTRAP_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "Pragma": "no-cache",
    "X-Content-Type-Options": "nosniff",
}


def asset_headers(filename: str) -> dict[str, str]:
    headers = {
        "Cache-Control": _CACHE_CONTROL.get(filename, "no-cache"),
        "X-Content-Type-Options": "nosniff",
    }
    if filename == "sw.js":
        # Without this a worker served from a subpath could only control that
        # subpath. It is served from the root here, so the header is belt and
        # braces, but it makes the intent explicit and survives a future move.
        headers["Service-Worker-Allowed"] = "/"
    return headers


def is_bootstrap(path: str, token: str | None) -> bool:
    """True when this is the one sanctioned ``/`` + ``?token=`` bootstrap."""
    return path == BOOTSTRAP_PATH and bool(token)


# The bootstrap page. Deliberately hand-built rather than rendered from the app
# template, for three reasons:
#
#   1. It contains no application JavaScript, so there is nothing that could
#      accidentally read the token later and leak it somewhere else.
#   2. The token is read from ``location.search`` at runtime and never written
#      into this markup, so it never appears in browser-visible HTML, in the
#      response body on disk, or in a ``View Source``.
#   3. It is served ``no-store``, so neither the HTTP cache nor the service
#      worker keeps a copy of a response that was requested with a token in it.
#
# ``location.replace`` (not ``location.href``) so the token-bearing URL does
# not survive in session history or in the back button.
BOOTSTRAP_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta name="robots" content="noindex, nofollow">
<title>Jarvis</title>
<style>
  html,body{margin:0;height:100%;background:#000;color:#eee;
    font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
    display:grid;place-items:center;-webkit-text-size-adjust:100%}
  main{text-align:center;padding:24px;max-width:22rem}
  h1{font-size:1.25rem;letter-spacing:.02em;margin:0 0 .5rem}
  p{margin:0;font-size:.875rem;opacity:.7}
</style>
</head>
<body>
<main>
  <h1>Jarvis</h1>
  <p id="s">Signing in&hellip;</p>
</main>
<script>
(function () {
  var KEY = "jarvis_token";
  var status = document.getElementById("s");
  var token = new URLSearchParams(location.search).get("token");
  if (!token) { location.replace("/"); return; }
  try {
    localStorage.setItem(KEY, token);
  } catch (e) {
    status.textContent = "Storage unavailable. Enable site data and retry.";
    return;
  }
  status.textContent = "Done.";
  // Drop the token from the visible URL and from history in one step.
  location.replace("/");
})();
</script>
</body>
</html>
"""