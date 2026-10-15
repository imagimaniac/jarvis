"""Provider payload behavior for the performance-oriented keep_alive option."""
from app.providers.llm import DEFAULT_MODEL, OllamaCompatible


def test_payload_omits_keep_alive_by_default():
    provider = OllamaCompatible()
    payload = provider._payload([{"role": "user", "content": "hi"}], stream=False, max_tokens=None)
    assert "keep_alive" not in payload
    assert payload["model"] == DEFAULT_MODEL


def test_payload_includes_keep_alive_when_configured():
    provider = OllamaCompatible(keep_alive="30m")
    payload = provider._payload([{"role": "user", "content": "hi"}], stream=True, max_tokens=16)
    assert payload["keep_alive"] == "30m"
    assert payload["stream"] is True


def test_from_config_keep_alive():
    provider = OllamaCompatible.from_config({"llm": {"keep_alive": "45m", "model_chat": "llama3.2"}})
    assert provider.keep_alive == "45m"
    provider = OllamaCompatible.from_config({})
    assert provider.keep_alive is None