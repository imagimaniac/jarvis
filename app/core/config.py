"""Single source of configuration truth.

Precedence, highest first:

    1. environment variables
    2. config.yaml            (gitignored, local overrides)
    3. config.example.yaml    (committed defaults)
    4. built-in defaults

cli.py and server.py each had their own loader and they disagreed: cli.py
resolved relative to ``Path.cwd()`` while server.py resolved relative to the
repo root, so the CLI silently picked up a different config depending on the
directory it was invoked from. Both now go through here, and relative paths
resolve against BASE.

Secrets are the one documented exception to the precedence chain: the bearer
token is read from the environment only, never from YAML, so it cannot end up
copied into a committed or shared file.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

# app/core/config.py -> app/core -> app -> repo root
BASE = Path(__file__).resolve().parents[2]

DEFAULT_HOST = "127.0.0.1"  # loopback only; never 0.0.0.0
DEFAULT_PORT = 8123
DEFAULT_DB_PATH = "data/jarvis.db"
DEFAULT_RAW_DIR = "memory/raw"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """Read config.yaml, falling back to config.example.yaml, then {}."""
    import yaml

    for name in ("config.yaml", "config.example.yaml"):
        path = BASE / name
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            # A malformed local config must not take the server down; the
            # environment and built-in defaults still apply.
            continue
    return {}


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _yaml(*keys: str, default: Any = None) -> Any:
    node: Any = load_config()
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return node if node not in (None, "") else default


def _under_base(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (BASE / path)


# --- accessors ---------------------------------------------------------------

def host() -> str:
    return _env("JARVIS_HOST") or str(_yaml("core", "host", default=None) or DEFAULT_HOST)


def port() -> int:
    raw = _env("JARVIS_PORT")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if 1 <= value <= 65535:
            return value
    return int(_yaml("server", "port", default=None) or DEFAULT_PORT)


def db_path() -> Path:
    return _under_base(_env("JARVIS_DB_PATH") or str(_yaml("storage", "db_path", default=None) or DEFAULT_DB_PATH))


def raw_dir() -> Path:
    return _under_base(_env("JARVIS_RAW_DIR") or str(_yaml("storage", "raw_dir", default=None) or DEFAULT_RAW_DIR))


def ollama_base_url() -> str:
    return _env("OLLAMA_BASE_URL") or str(_yaml("llm", "base_url", default=None) or DEFAULT_OLLAMA_BASE_URL)


def memory_candidates_enabled() -> bool:
    """Silent deterministic background candidate producer (Phase 1).

    Reads the environment first so the flag can be flipped without touching
    YAML; defaults to disabled to keep the chat path unchanged until opted in.
    """
    value = _env("JARVIS_MEMORY_CANDIDATES")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_yaml("features", "memory_candidates", default=None) or False)


def memory_judge_enabled() -> bool:
    """Gated Memory Judge over stored CANDIDATE memories (Phase 2).

    Even when enabled, the judge only records its decision in memory_history;
    it does not mutate confirmed memories unless ``apply`` is enabled too.
    """
    value = _env("JARVIS_MEMORY_JUDGE")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_yaml("features", "memory_judge", default=None) or False)


def memory_judge_apply() -> bool:
    """Whether the gated Memory Judge may apply CONFIRM/UPDATE/SUPERSEDE/FORGET.

    Defaults to disabled; live autonomous memory mutations stay off until the
    judge has been proven safe in isolation.
    """
    value = _env("JARVIS_MEMORY_JUDGE_APPLY")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_yaml("features", "memory_judge_apply", default=None) or False)


def context_engine_enabled() -> bool:
    """Deterministic cross-rating Context Engine (Phase 4).

    Defaults to enabled; when disabled the chat path falls back to the plain
    ``personal_context`` context string with byte-identical behavior.
    """
    value = _env("JARVIS_CONTEXT_ENGINE")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_yaml("features", "context_engine", default=True))


def web_enabled() -> bool:
    """Opt-in public web retrieval for current-information questions.

    Defaults to disabled so the local-only privacy stance holds unless the user
    explicitly turns it on (env JARVIS_WEB=1 or features.web: true). Only public
    pages are fetched; personal memory is never sent to the web.
    """
    value = _env("JARVIS_WEB")
    if value is not None:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(_yaml("features", "web", default=False))


def web_search_provider() -> str:
    return (_env("JARVIS_WEB_PROVIDER")
            or str(_yaml("web", "search_provider", default=None) or "bing_news")).strip().lower()


def web_max_results() -> int:
    raw = _env("JARVIS_WEB_MAX_RESULTS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if 1 <= value <= 10:
            return value
    try:
        value = int(_yaml("web", "max_results", default=None) or 4)
    except (TypeError, ValueError):
        value = 4
    return max(1, min(10, value))


def token() -> Optional[str]:
    """Bearer token from the environment only. Never read from YAML.

    Read on every call rather than cached so that a test can change the
    environment without reimporting the app.
    """
    return _env("JARVIS_TOKEN")
