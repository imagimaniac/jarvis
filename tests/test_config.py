"""Configuration precedence and loopback defaults (Phase 1A/1C)."""
import os

import pytest

from app.core import config

CONFIG_ENV = ("JARVIS_HOST", "JARVIS_PORT", "JARVIS_DB_PATH", "JARVIS_RAW_DIR",
              "JARVIS_TOKEN", "OLLAMA_BASE_URL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every case starts from a known-empty config environment.

    Without this the suite inherits whatever the developer happens to have
    exported in their shell -- an exported JARVIS_DB_PATH pointed at /tmp made
    these assertions fail even though the code was correct.
    """
    for key in CONFIG_ENV:
        monkeypatch.delenv(key, raising=False)


def test_defaults_are_loopback_and_local(client=None):
    """The safe default must be loopback, never 0.0.0.0."""
    assert config.host() == "127.0.0.1"
    assert config.DEFAULT_HOST == "127.0.0.1"
    assert config.DEFAULT_PORT == 8123


def test_no_default_is_wildcard_address():
    """Guards against someone reintroducing 0.0.0.0 as a default."""
    assert config.DEFAULT_HOST not in ("0.0.0.0", "::", "")
    assert "0.0.0.0" not in (config.DEFAULT_HOST, config.DEFAULT_PORT.__str__())


def test_env_overrides_yaml(monkeypatch):
    monkeypatch.setenv("JARVIS_HOST", "127.0.0.2")
    assert config.host() == "127.0.0.2"
    monkeypatch.setenv("JARVIS_PORT", "9123")
    assert config.port() == 9123


@pytest.mark.parametrize("bad,expected", [
    ("0", config.DEFAULT_PORT),
    ("70000", config.DEFAULT_PORT),
    ("not-a-number", config.DEFAULT_PORT),
    ("", config.DEFAULT_PORT),
])
def test_invalid_port_falls_back_rather_than_crash(monkeypatch, bad, expected):
    monkeypatch.setenv("JARVIS_PORT", bad)
    assert config.port() == expected


def test_paths_resolve_against_repo_root_not_cwd(monkeypatch, tmp_path):
    """The old CLI loader used Path.cwd(), so config changed with the directory."""
    monkeypatch.chdir(tmp_path)
    assert config.BASE not in (tmp_path,)
    assert config.db_path().is_absolute()
    assert config.raw_dir().is_absolute()
    assert str(config.db_path()).startswith(str(config.BASE))


def test_env_paths_win(monkeypatch, tmp_path):
    monkeypatch.setenv("JARVIS_DB_PATH", str(tmp_path / "custom.db"))
    assert config.db_path() == tmp_path / "custom.db"


def test_absolute_env_path_is_not_rebased(monkeypatch, tmp_path):
    target = tmp_path / "abs.db"
    monkeypatch.setenv("JARVIS_DB_PATH", str(target))
    assert config.db_path() == target


def test_ollama_base_url_has_no_cloud_fallback(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    url = config.ollama_base_url()
    assert url.startswith("http://")
    assert "localhost" in url or "127.0.0.1" in url, "must not silently point at a cloud model"


def test_token_is_env_only(monkeypatch):
    """A token in YAML must never become the accepted credential."""
    monkeypatch.setenv("JARVIS_TOKEN", "from-env")
    assert config.token() == "from-env"
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    assert config.token() is None


def test_token_is_not_cached(monkeypatch):
    """Tests change the environment per-case; caching would break them."""
    monkeypatch.setenv("JARVIS_TOKEN", "first")
    assert config.token() == "first"
    monkeypatch.setenv("JARVIS_TOKEN", "second")
    assert config.token() == "second"
    monkeypatch.delenv("JARVIS_TOKEN", raising=False)
    assert config.token() is None


def test_whitespace_token_is_unset(monkeypatch):
    monkeypatch.setenv("JARVIS_TOKEN", "  \t ")
    assert config.token() is None


def test_no_dotenv_reader_is_shipped():
    """Documented in .env.example: values come from the process environment."""
    assert not (config.BASE / ".env").exists(), "a real .env must not be committed"
    src = (config.BASE / "app" / "core" / "config.py").read_text(encoding="utf-8")
    assert "dotenv" not in src.lower() or "load_dotenv" not in src
    assert "load_dotenv" not in src


def test_env_example_documents_every_required_key():
    text = (config.BASE / ".env.example").read_text(encoding="utf-8")
    for key in ("JARVIS_TOKEN", "JARVIS_HOST", "JARVIS_PORT",
                "JARVIS_DB_PATH", "JARVIS_RAW_DIR", "OLLAMA_BASE_URL"):
        assert key in text, f"{key} missing from .env.example"
    assert "JARVIS_PORT=8123" in text


def test_env_example_carries_no_secret():
    """It must document names only: no real token, and no forbidden services."""
    text = (config.BASE / ".env.example").read_text(encoding="utf-8")
    lines = text.splitlines()
    assignments = [ln.split("=", 1)[1].strip()
                   for ln in lines
                   if ln.startswith("JARVIS_TOKEN=")]
    assert assignments == [""], f"JARVIS_TOKEN must ship empty, found {assignments}"
    lowered = text.lower()
    for banned in ("supabase", "neon", "postgres", "render.com", "vercel",
                   "database_url", "0.0.0.0"):
        assert banned not in lowered, f"{banned} must not appear in .env.example"


def test_start_scripts_bind_loopback():
    """Phase 1C: neither launcher may open the API to the LAN."""
    for name in ("start_jarvis.sh", "run_server.sh"):
        text = (config.BASE / "scripts" / name).read_text(encoding="utf-8")
        if name == "start_jarvis.sh":
            # The lifecycle launcher delegates host/port validation to this
            # shared helper rather than duplicating its bind configuration.
            text += (config.BASE / "scripts" / "jarvis_lifecycle_lib.sh").read_text(encoding="utf-8")
        # Strip comments: the files warn against 0.0.0.0 by naming it.
        code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))
        assert "0.0.0.0" not in code, f"{name} still binds all interfaces"
        assert "127.0.0.1" in code, f"{name} does not default to loopback"
        assert "JARVIS_HOST" in code, f"{name} ignores JARVIS_HOST"


def test_config_is_shared_by_cli_and_server():
    """Both entrypoints must resolve config identically."""
    from app import cli, server

    assert cli.load_config() == server._cfg()
    assert cli.get_paths({})[1] == server._paths({})[1]


def test_no_cloud_or_lan_dependencies_were_added():
    """Explicit architectural guard for Group 1."""
    assert "requests" not in os.environ  # sanity: env not used as a smuggling place
    reqs = (config.BASE / "requirements.txt").read_text(encoding="utf-8").lower()
    for banned in ("psycopg", "postgres", "sqlalchemy", "asyncpg", "supabase",
                   "docker", "render", "vercel"):
        assert banned not in reqs, f"{banned} must not be a dependency"
