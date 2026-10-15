"""PWA foundation: static assets, token bootstrap, service worker (Group 2).

The service worker is JavaScript and there is deliberately no JS test runner in
this project, so the worker is verified two ways: by asserting on its source for
the rules that matter (never cache /api/*, never cache a tokenised URL), and by
asserting the server serves it correctly at the root scope.
"""
import json
import re
import shutil
import struct
import subprocess

import pytest
from fastapi.testclient import TestClient

from app.core import config, pwa
from app.server import app

TOKEN = "pwa-token-0123456789abcdef"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

ASSET_NAMES = ["manifest.json", "sw.js", "icon-192.png", "icon-512.png",
               "maskable-512.png", "apple-touch-icon.png"]

# Correct pixel dimensions per asset.
EXPECTED_SIZES = {
    "icon-192.png": (192, 192),
    "icon-512.png": (512, 512),
    "maskable-512.png": (512, 512),
    "apple-touch-icon.png": (180, 180),
}

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_TOKEN", TOKEN)
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "pwa.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    return TestClient(app, raise_server_exceptions=False)


def png_dimensions(data: bytes):
    """Read width/height out of the IHDR chunk."""
    assert data[:8] == PNG_MAGIC, "not a PNG"
    assert data[12:16] == b"IHDR", "IHDR must be the first chunk"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


# --- static assets -----------------------------------------------------------

@pytest.mark.parametrize("name", ASSET_NAMES)
def test_asset_is_served(client, name):
    r = client.get(f"/{name}")
    assert r.status_code == 200, f"/{name} should be served"
    assert r.content, f"/{name} is empty"


@pytest.mark.parametrize("name,expected_type", [
    ("manifest.json", "application/manifest+json"),
    ("sw.js", "text/javascript"),
    ("icon-192.png", "image/png"),
    ("apple-touch-icon.png", "image/png"),
])
def test_asset_content_type(client, name, expected_type):
    r = client.get(f"/{name}")
    assert expected_type in r.headers["content-type"]


@pytest.mark.parametrize("name,size", sorted(EXPECTED_SIZES.items()))
def test_icon_is_valid_png_with_correct_dimensions(client, name, size):
    r = client.get(f"/{name}")
    assert r.content[:8] == PNG_MAGIC
    assert png_dimensions(r.content) == size


def test_icons_need_no_auth(client):
    """The shell must load before the user has a token."""
    for name in ASSET_NAMES:
        assert client.get(f"/{name}").status_code == 200


def test_root_still_works(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Jarvis" in r.text


def test_unknown_static_path_is_404(client):
    """No arbitrary filesystem access: an unmapped name does not exist."""
    for name in ["config.yaml", "requirements.txt", "index.html.bak",
                 "config.py", "app.py", "__init__.py"]:
        assert client.get(f"/{name}").status_code == 404, f"/{name} should 404"


@pytest.mark.parametrize("path", [
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
])
def test_fastapi_documentation_routes_are_disabled(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", [
    "/../config.yaml",
    "/..%2fconfig.yaml",
    "/%2e%2e/config.yaml",
    "/static/../config.yaml",
    "//../config.yaml",
    "/.%2e/config.yaml",
])
def test_path_traversal_is_refused(client, path):
    """Traversal must never reach config.yaml or any source file."""
    r = client.get(path)
    assert r.status_code in (400, 404), f"{path} returned {r.status_code}"
    assert "database_url" not in r.text.lower()
    assert "provider" not in r.text.lower()


def test_assets_send_nosniff(client):
    """Stops a browser from second-guessing the declared content type."""
    assert client.get("/manifest.json").headers.get("x-content-type-options") == "nosniff"


def test_sw_is_served_at_root_scope(client):
    """A worker must be served from the root to control the whole origin."""
    r = client.get("/sw.js")
    assert r.status_code == 200
    assert r.headers.get("service-worker-allowed") == "/"
    assert "no-cache" in r.headers.get("cache-control", "")


# --- manifest ----------------------------------------------------------------

def test_manifest_is_valid_json(client):
    r = client.get("/manifest.json")
    manifest = json.loads(r.text)  # raises if malformed
    assert isinstance(manifest, dict)


@pytest.mark.parametrize("key,expected", [
    ("name", "Jarvis Assistant"),
    ("short_name", "Jarvis"),
    ("start_url", "/"),
    ("display", "standalone"),
    ("background_color", "#0B0F14"),
    ("theme_color", "#0B0F14"),
])
def test_manifest_required_fields(client, key, expected):
    manifest = json.loads(client.get("/manifest.json").text)
    assert manifest.get(key) == expected, f"manifest.{key} wrong"


def test_manifest_declares_the_real_icons(client):
    manifest = json.loads(client.get("/manifest.json").text)
    sources = {i["src"] for i in manifest["icons"]}
    assert "/icon-192.png" in sources
    assert "/icon-512.png" in sources
    assert "/maskable-512.png" in sources
    # every declared icon must actually resolve
    for src in sources:
        assert client.get(src).status_code == 200, f"{src} declared but missing"


def test_manifest_has_a_maskable_icon(client):
    manifest = json.loads(client.get("/manifest.json").text)
    purposes = " ".join(i.get("purpose", "") for i in manifest["icons"])
    assert "maskable" in purposes


def test_manifest_start_url_is_not_tokenised(client):
    """Launching the installed app must not put a token in the URL."""
    manifest = json.loads(client.get("/manifest.json").text)
    assert "token" not in manifest["start_url"]


# --- HTML metadata -----------------------------------------------------------

@pytest.mark.parametrize("snippet", [
    '<link rel="manifest" href="/manifest.json">',
    '<meta name="theme-color" content="#0B0F14">',
    '<meta name="apple-mobile-web-app-capable" content="yes">',
    '<meta name="mobile-web-app-capable" content="yes">',
    '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">',
    'width=device-width, initial-scale=1, viewport-fit=cover',
    '<link rel="apple-touch-icon" href="/apple-touch-icon.png">',
])
def test_html_has_pwa_metadata(client, snippet):
    assert snippet in client.get("/").text


def test_html_registers_the_service_worker(client):
    html = client.get("/").text
    assert '"/sw.js"' in html
    assert "serviceWorker" in html


# --- UI authentication repair ------------------------------------------------

def test_ui_uses_the_authorization_header(client):
    """The repair: API calls must carry a bearer header, built in one place."""
    html = client.get("/").text
    assert 'headers["Authorization"] = "Bearer " + token' in html
    assert 'var TOKEN_KEY = "jarvis_token"' in html


def test_ui_never_puts_the_token_in_a_url(client):
    """No API request may carry the token as a query parameter.

    The one legitimate mention of ``?token=`` is the ``/?token=YOUR_TOKEN``
    bootstrap hint shown in the unlock panel: a placeholder with no secret in
    it, on the root path, never on ``/api/``.
    """
    html = client.get("/").text
    for line in html.splitlines():
        if "?token=" not in line:
            continue
        assert "/api/" not in line, f"token on an API URL: {line.strip()}"
        assert "YOUR_TOKEN" in line or "&lt;" in line, \
            f"unexpected token usage: {line.strip()}"
    # The helper must build the header, not a query parameter.
    assert 'headers["Authorization"] = "Bearer " + token' in html


def test_ui_does_not_hardcode_a_token(client):
    """No literal credential anywhere in the shipped page."""
    html = client.get("/").text
    assert TOKEN not in html
    assert "Bearer sk-" not in html


def test_frontend_stores_only_the_token(client):
    """No API responses may be persisted to localStorage."""
    html = client.get("/").text
    assert html.count("localStorage.setItem") == 1
    assert "localStorage.setItem" in html


def test_capture_search_recent_remain_available_through_authenticated_api(client):
    """The primary shell is chat-only; underlying API capabilities remain."""
    html = client.get("/").text.split("<script>", 1)[0]
    for marker in ('Capture', 'Search', 'Recent', 'Sources', 'Evidence'):
        assert marker not in html
    added = client.post("/api/add", json={"text": "UI-hidden capture check"}, headers=AUTH)
    assert added.status_code == 200
    assert client.get("/api/search?q=UI-hidden", headers=AUTH).status_code == 200
    assert client.get("/api/recent", headers=AUTH).status_code == 200


def test_mobile_first_requirements(client):
    """No horizontal scroll, touch-sized controls, safe-area support."""
    html = client.get("/").text
    assert "overflow: hidden" in html
    assert "env(safe-area-inset-top" in html
    assert "env(safe-area-inset-bottom" in html
    assert "height: 48px" in html
    # iOS zooms on focus below 16px
    assert "font-size: 16px" in html


def test_graphite_cool_signal_theme_preserves_readable_role_hierarchy(client):
    html = client.get("/").text
    for token in (
        "--bg: #0B0F14", "--panel: #111827", "--field: #111827",
        "--text: #D1D5DB", "--text-user: #9CA3AF", "--signal: #38BDF8",
        "--signal-rail: #E8D58A", "--signal-active: #F0E2A6", "--line: #1F2937",
    ):
        assert token in html
    assert "--font-ui: ui-sans-serif" in html
    assert "--font-mono: ui-monospace" in html
    assert "font: 16px/1.55 var(--font-ui)" in html
    assert ".message-content {" in html and "color: var(--text);" in html
    assert ".message-user .message-content { color: var(--text-user); }" in html
    assert ".message-user {" in html and "border-right: 1px solid var(--signal-rail);" in html
    assert ".speaker {" in html and "color: var(--signal);" in html
    assert ".message-user .speaker { color: var(--text-user); }" not in html
    assert ".message-jarvis.is-streaming" in html
    assert "--signal-active:" in html
    assert "navigator.onLine" in html
    assert "browser network connectivity only, not Jarvis server/auth health" in html
    assert ".send:disabled { opacity: 1; border-color: var(--line); color: var(--muted);" in html
    assert ".composer textarea:focus { border-color: var(--signal);" in html


def test_answer_renderer_uses_safe_dom_nodes_and_retains_stream_behavior(client):
    """Execute the real inline renderer against a tiny DOM shim in Node."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; inline renderer source is checked separately")
    html = client.get("/").text
    match = re.search(
        r"/\* BEGIN_ANSWER_RENDERER \*/(.*?)/\* END_ANSWER_RENDERER \*/",
        html, re.S,
    )
    assert match, "the testable renderer block should be present"
    renderer = match.group(1)
    assert "innerHTML" not in renderer
    harness = renderer + r'''
const assert = require("node:assert/strict");
global.document = {
  createElement(tag) { return { tag, className: "", textContent: "", children: [], appendChild(child) { this.children.push(child); } }; },
  createTextNode(text) { return { tag: "text", textContent: text }; }
};
function render(text) {
  const node = { children: [], replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); } };
  renderAnswerText(node, text);
  return node.children;
}
function visible(children) { return children.map((child) => child.textContent).join(""); }
let children = render("Dial volume fell by 28.5% while collections remained around $80M; ₹22.98 LPA over 2 hours, across 5–9 years.");
assert.deepEqual(children.filter((child) => child.tag === "strong").map((child) => child.textContent), ["28.5%", "$80M", "₹22.98 LPA", "2 hours", "5–9 years"]);
children = render("The total was 28.5% [42]\n\tthen $80M");
assert.equal(visible(children), "The total was 28.5% \n\tthen $80M");
assert(!visible(children).includes("[42]"));
assert.equal(render("Historical 2025; phone 5551234567; ID 12345678901234567890; item 1.").filter((child) => child.tag === "strong").length, 0);
assert.equal(render("Updated on 2026-10-05").filter((child) => child.tag === "strong")[0].textContent, "2026-10-05");
children = render("<img src=x onerror=alert(1)>  $80");
assert.equal(children[0].tag, "text");
assert.equal(visible(children), "<img src=x onerror=alert(1)>  $80");
assert.equal(render("Revenue is $80").filter((child) => child.tag === "strong")[0].textContent, "$80");
assert.equal(render("Revenue is $80M").filter((child) => child.tag === "strong")[0].textContent, "$80M");
assert.equal(render("Revenue is $80").filter((child) => child.tag === "strong")[0].textContent, "$80");
assert.equal(render("Revenue is $80M").filter((child) => child.tag === "strong")[0].textContent, "$80M");
function renderWith(text, links) {
  const node = { children: [], replaceChildren() { this.children = []; }, appendChild(child) { this.children.push(child); } };
  renderAnswerText(node, text, links);
  return node.children;
}
// Final pass with a link map: backed citations become links, others are dropped.
var linked = renderWith("See [W1] and [W9].", { "1": "https://x.example/a" });
assert.deepEqual(linked.filter((child) => child.tag === "a").map((child) => child.textContent), ["[W1]"]);
assert.equal(linked.filter((child) => child.tag === "a")[0].href, "https://x.example/a");
assert.equal(visible(linked), "See [W1] and .");
assert(!visible(linked).includes("[W9]"));
// No link map (streaming): markers stay as plain text.
assert.equal(visible(renderWith("See [W1].", null)), "See [W1].");
// Internal provenance labels are hidden from the user-facing answer.
assert.equal(visible(render("Memory: I do not have it.")), "I do not have it.");
assert.equal(visible(render("Recorded: third party recorded that your flight is on Saturday.")), "Your flight is on Saturday.");
assert.equal(visible(render("Memory: I do not have it. Recorded: your flight is on Saturday.")), "I do not have it. Your flight is on Saturday.");
console.log("renderer ok");
'''
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "renderer ok"


def test_web_meta_reflects_retrieval_status_without_false_success(client):
    """Execute the real web-meta block against a tiny DOM shim in Node."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; source is checked by other tests")
    html = client.get("/").text
    match = re.search(r"/\* BEGIN_WEB_META \*/(.*?)/\* END_WEB_META \*/", html, re.S)
    assert match, "the testable web-meta block should be present"
    block = match.group(1)
    assert "innerHTML" not in block
    harness = block + r'''
const assert = require("node:assert/strict");
global.document = {
  createElement(tag) { return { tag, className: "", textContent: "" }; }
};
function container() {
  const parent = { children: [], appendChild(child) { this.children.push(child); }, querySelector() { return null; } };
  return { parentElement: parent };
}
// Silent when nothing was searched or fields are missing (older responses).
assert.equal(webSearchMeta(null), null);
assert.equal(webSearchMeta(undefined), null);
assert.equal(webSearchMeta({}), null);
assert.equal(webSearchMeta({ web_status: "empty", web_attempted: false }), null);
assert.equal(webSearchMeta({ web_status: "", web_attempted: false }), null);
// Success only when evidence was truly returned.
assert.deepEqual(
  webSearchMeta({ web_status: "ok", web_query: "inflation", web_attempted: true }),
  { text: "Searched the web \u00b7 inflation", warning: false });
assert.deepEqual(
  webSearchMeta({ web_status: "url", web_query: "https://x.example/a", web_attempted: true }),
  { text: "Searched the web \u00b7 https://x.example/a", warning: false });
// Failed / empty / skipped are described accurately, no false success.
assert.equal(webSearchMeta({ web_status: "error", web_attempted: true }).warning, true);
assert.equal(webSearchMeta({ web_status: "url_error", web_attempted: true }).text, "Live sources unavailable");
assert.equal(webSearchMeta({ web_status: "empty", web_attempted: true }).text, "No live sources found");
assert.equal(webSearchMeta({ web_status: "no_query", web_attempted: true }).text, "Live search skipped \u00b7 no usable query");
// Long queries are truncated, but still marked as searched.
const long = webSearchMeta({ web_status: "ok", web_query: "x".repeat(90), web_attempted: true }).text;
assert(long.length < 90 && long.endsWith("\u2026"));
// renderWebMeta appends a single compact node to the message wrapper.
const c = container();
renderWebMeta(c, { web_status: "ok", web_query: "inflation", web_attempted: true });
assert.equal(c.parentElement.children.length, 1);
assert.equal(c.parentElement.children[0].className, "web-meta");
assert.equal(c.parentElement.children[0].textContent, "Searched the web \u00b7 inflation");
// Does not render for non-searched responses.
renderWebMeta(container(), { web_status: "empty", web_attempted: false });
assert.equal(container().parentElement.children.length, 0);
console.log("web-meta ok");
'''
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "web-meta ok"


def test_web_sources_render_clickable_public_links(client):
    """Execute the real sources renderer against a tiny DOM shim in Node."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is not installed; source is checked separately")
    html = client.get("/").text
    match = re.search(r"/\* BEGIN_WEB_META \*/(.*?)/\* END_WEB_META \*/", html, re.S)
    assert match, "the testable web-meta block should be present"
    block = match.group(1)
    assert "innerHTML" not in block
    harness = block + r'''
const assert = require("node:assert/strict");
function makeNode(tag) {
  return { tag: tag, className: "", textContent: "", href: "", target: "", rel: "",
           children: [], appendChild(c) { this.children.push(c); }, remove() {} };
}
global.document = { createElement: function (tag) { return makeNode(tag); } };
assert.deepEqual(
  sourceLinkMap([{ label: "W1", url: "https://a.example/1" }, { label: "W2", url: "https://b.example/2" }]),
  { "1": "https://a.example/1", "2": "https://b.example/2" });
assert.deepEqual(sourceLinkMap([]), {});
assert.deepEqual(sourceLinkMap(null), {});
function makeContainer() {
  var parent = makeNode("div");
  parent.querySelector = function () { return null; };
  return { parentElement: parent };
}
var c = makeContainer();
renderSources(c, [{ label: "W1", title: "CJP protest", url: "https://news.example/trains" }]);
assert.equal(c.parentElement.children.length, 1);
var box = c.parentElement.children[0];
assert.equal(box.className, "web-sources");
assert.equal(box.children[0].textContent, "Sources");
var link = box.children[1].children[0];
assert.equal(link.tag, "a");
assert.equal(link.href, "https://news.example/trains");
assert.equal(link.rel, "noopener noreferrer");
assert.equal(link.textContent, "W1 \u00b7 CJP protest");
// No evidence renders nothing.
renderSources(makeContainer(), []);
console.log("sources ok");
'''
    result = subprocess.run([node, "-e", harness], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "sources ok"


def test_service_worker_only_bumps_shell_version_and_keeps_cache_rules(client):
    sw = client.get("/sw.js").text
    assert "const VERSION = 'v2';" in sw
    assert "if (url.pathname === '/api' || url.pathname.startsWith('/api/')) return true;" in sw
    assert "if (req.mode === 'navigate')" in sw
    assert "if (cached) return cached;" in sw


# --- the three repaired calls, end to end ------------------------------------

def test_add_search_recent_work_with_bearer_auth(client):
    """The exact calls the broken UI used to make, now authenticated."""
    added = client.post("/api/add", json={"text": "pwa group two check"}, headers=AUTH)
    assert added.status_code == 200
    assert "id" in added.json()

    s = client.get("/api/search?q=pwa", headers=AUTH)
    assert s.status_code == 200
    assert "results" in s.json()

    r = client.get("/api/recent", headers=AUTH)
    assert r.status_code == 200
    assert "results" in r.json()


# --- token bootstrap ---------------------------------------------------------

def test_bootstrap_with_valid_token_succeeds(client):
    r = client.get(f"/?token={TOKEN}")
    assert r.status_code == 200


def test_bootstrap_response_never_contains_the_token(client):
    """The whole point: the credential must not reach the HTML."""
    r = client.get(f"/?token={TOKEN}")
    assert TOKEN not in r.text


def test_bootstrap_response_is_not_stored(client):
    r = client.get(f"/?token={TOKEN}")
    assert r.headers.get("cache-control") == "no-store"
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert r.headers.get("pragma") == "no-cache"


def test_bootstrap_page_is_minimal_and_does_not_carry_the_app(client):
    """Must not embed the application payload or any API capability."""
    body = client.get(f"/?token={TOKEN}").text
    assert "/api/search" not in body
    assert "/api/add" not in body
    assert "/api/recent" not in body
    assert "app.js" not in body


def test_bootstrap_stores_token_and_replaces_url(client):
    """The page must persist the token and then clear the URL."""
    body = client.get(f"/?token={TOKEN}").text
    assert 'localStorage.setItem' in body
    assert '"jarvis_token"' in body
    # replace(), not href=, so the token leaves history
    assert 'location.replace("/")' in body
    assert "location.href" not in body


def test_bootstrap_reads_token_from_url_not_from_markup(client):
    """Token arrives at runtime from location.search, never in the source."""
    body = client.get(f"/?token={TOKEN}").text
    assert "location.search" in body


def test_invalid_bootstrap_token_is_rejected(client):
    r = client.get("/?token=wrong-token")
    assert r.status_code == 401
    assert TOKEN not in r.text


def test_bootstrap_prefix_token_is_rejected(client):
    """Truncation must not be accepted."""
    assert client.get(f"/?token={TOKEN[:8]}").status_code == 401
    assert client.get(f"/?token={TOKEN}x").status_code == 401


def test_bootstrap_fails_closed_without_configured_token(monkeypatch, tmp_path):
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "pwa.db"))
    monkeypatch.setenv("JARVIS_RAW_DIR", str(tmp_path / "raw"))
    c = TestClient(app, raise_server_exceptions=False)
    assert c.get("/?token=anything").status_code == 401


def test_bootstrap_does_not_grant_api_access_by_itself(client):
    """Getting the shell must not authenticate subsequent API calls."""
    assert client.get("/?token=" + TOKEN).status_code == 200
    assert client.get("/api/recent").status_code == 401


def test_plain_root_is_the_app_not_the_bootstrap(client):
    body = client.get("/").text
    assert "jarvis_token" not in body or TOKEN not in body
    assert "/manifest.json" in body


def test_api_query_token_is_still_rejected(client):
    """Group 1's rule must survive the bootstrap exception.

    Note the separator: ``&token=`` when the path already carries a query
    string. ``/api/search?q=x?token=T`` would parse as a single ``q`` parameter
    and prove nothing.
    """
    assert client.get(f"/api/recent?token={TOKEN}", headers=AUTH).status_code == 401
    assert client.get(f"/api/search?q=x&token={TOKEN}", headers=AUTH).status_code == 401
    assert client.get(f"/api/stats?token={TOKEN}", headers=AUTH).status_code == 401
    assert client.get(f"/api/show/1?token={TOKEN}", headers=AUTH).status_code == 401
    assert client.post(f"/api/add?token={TOKEN}", json={"text": "x"},
                       headers=AUTH).status_code == 401


def test_query_token_is_rejected_regardless_of_header_validity(client):
    """A token in the URL is refused even with a correct bearer header."""
    assert client.get(f"/api/recent?token={TOKEN}").status_code == 401
    assert client.get(f"/api/recent?token={TOKEN}",
                      headers={"Authorization": "Bearer bogus"}).status_code == 401


def test_query_token_is_only_allowed_on_root(client):
    """The bootstrap exception is scoped to exactly one path."""
    for path in ["/manifest.json", "/sw.js", "/icon-192.png"]:
        r = client.get(f"{path}?token={TOKEN}")
        assert r.status_code == 200, "assets ignore query params entirely"
        assert r.text.find(TOKEN) == -1


# --- service worker rules ----------------------------------------------------

def test_sw_never_caches_api():
    """Personal data must not be able to reach the Cache API."""
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    assert "'/api'" in sw or '"/api"' in sw
    assert "startsWith('/api/')" in sw or 'startsWith("/api/")' in sw
    # Within the fetch handler, the private-request guard must run before any
    # respondWith. Compare inside the handler body only: the word
    # "respondWith" also appears in the file's explanatory comments. The /api
    # test itself lives in the isPrivate() helper, so assert the handler calls
    # that helper first.
    handler = sw[sw.index("self.addEventListener('fetch'"):]
    assert "isPrivate(req, url)" in handler
    assert handler.index("isPrivate(req, url)") < handler.index("respondWith"), \
        "private-request guard must run before respondWith"


def test_sw_api_guard_returns_without_responding():
    """The /api branch must `return`, i.e. never call respondWith for it."""
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    guard = sw[sw.index("function isPrivate"):sw.index("self.addEventListener('fetch'")]
    assert "startsWith('/api/')" in guard
    assert "return true" in guard
    assert "respondWith" not in guard, "isPrivate must only classify, never respond"


def test_sw_refuses_to_cache_authenticated_or_tokenised_requests():
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    assert "searchParams.has('token')" in sw or 'searchParams.has("token")' in sw
    assert "authorization" in sw
    assert "cookie" in sw


def test_sw_cache_name_is_versioned():
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    assert "const VERSION = 'v2'" in sw or 'const VERSION = "v2"' in sw
    assert "jarvis-shell-" in sw
    # old versions must be dropped on activate
    assert "caches.delete" in sw


def test_sw_does_not_precache_api_or_sw_itself():
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    block = sw[sw.index("SHELL_ASSETS = ["):sw.index("];", sw.index("SHELL_ASSETS = ["))]
    assert "/api/" not in block
    assert "/sw.js" not in block


def test_sw_only_stores_successful_basic_responses():
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    assert "res.ok" in sw
    assert "type === 'basic'" in sw or 'type === "basic"' in sw


def test_sw_is_same_origin_only():
    sw = (pwa.STATIC_DIR / "sw.js").read_text(encoding="utf-8")
    assert "self.location.origin" in sw


# --- no cloud / no framework creep -------------------------------------------

def test_no_frontend_framework_or_toolchain():
    """Group 2 stays vanilla: no bundler, no package manifest, no CDN."""
    for name in ("package.json", "package-lock.json", "yarn.lock",
                 "vite.config.js", "webpack.config.js", "tsconfig.json"):
        assert not (config.BASE / name).exists(), f"{name} was introduced"
    html = (pwa.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html.replace("http://www.w3.org", "")
    assert "https://" not in html


def test_no_cors_widened(client):
    """Group 1's closed-CORS posture must be intact."""
    r = client.options("/api/recent", headers={
        "Origin": "https://evil.example",
        "Access-Control-Request-Method": "GET",
    })
    assert r.headers.get("access-control-allow-origin") is None


def test_backend_still_loopback_only():
    """No LAN binding crept in via the PWA work."""
    from app.core import config as cfg_mod
    assert cfg_mod.DEFAULT_HOST == "127.0.0.1"
    for name in ("start_jarvis.sh", "run_server.sh"):
        text = (config.BASE / "scripts" / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))
        assert "0.0.0.0" not in code
