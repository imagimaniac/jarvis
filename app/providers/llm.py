"""LLM provider - Ollama or any OpenAI-compatible endpoint. Optional.

Model independence lives here and nowhere else: no module outside this file
knows which provider is in use. Callers pass messages and get text back.
If no provider is reachable, `available()` is False and core stays heuristic.
"""
from __future__ import annotations
import json
import threading
from typing import Any, Dict, Iterator, List, Optional, Sequence

import httpx

Message = Dict[str, str]

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.2"
_DONE = "[DONE]"


class ProviderStreamError(RuntimeError):
    """The provider failed while opening or consuming a streamed completion."""


class OllamaCompatible:
    """Chat + JSON completion against an OpenAI-compatible /chat/completions."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 800,
        timeout: int = 60,
        keep_alive: str | None = None,
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.keep_alive = keep_alive
        self._stream_lock = threading.Lock()
        self._active_stream_response = None
        self._stream_cancel_event: threading.Event | None = None

    def set_stream_cancel_event(self, event: threading.Event) -> None:
        """Register the request's disconnect signal for the active stream."""
        self._stream_cancel_event = event

    def cancel_chat_stream(self) -> None:
        """Close an active HTTP stream when its client disconnects."""
        event = self._stream_cancel_event
        if event is not None:
            event.set()
        with self._stream_lock:
            response = self._active_stream_response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    @classmethod
    def from_config(cls, cfg: dict) -> "OllamaCompatible":
        llm = (cfg or {}).get("llm") or {}
        model = llm.get("model_chat") or llm.get("model_retrieval") or llm.get("model_extraction") or DEFAULT_MODEL
        return cls(
            base_url=llm.get("base_url") or DEFAULT_BASE_URL,
            model=model,
            temperature=float(llm.get("temperature", 0.0) or 0.0),
            max_tokens=int(llm.get("max_tokens", 800) or 800),
            timeout=int(llm.get("timeout_seconds", 60) or 60),
            keep_alive=llm.get("keep_alive"),
        )

    def _payload(self, messages: Sequence[Message], stream: bool, max_tokens: Optional[int]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream,
        }
        # Ollama honors keep_alive on its OpenAI-compatible endpoint; harmless
        # to omit elsewhere.
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        return payload

    def available(self) -> bool:
        """Cheap reachability check. Never raises."""
        try:
            with httpx.Client(timeout=3) as c:
                r = c.get(f"{self.base_url}/models")
                return r.status_code < 500
        except Exception:
            return False

    def chat(self, messages: Sequence[Message], *, max_tokens: Optional[int] = None) -> Optional[str]:
        """Full completion as text. None on any failure."""
        payload = self._payload(messages, stream=False, max_tokens=max_tokens)
        try:
            with httpx.Client(timeout=self.timeout) as c:
                r = c.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    def chat_stream(self, messages: Sequence[Message], *, max_tokens: Optional[int] = None) -> Iterator[str]:
        """Yield text deltas as they arrive.

        Raises ProviderStreamError on a transport/protocol failure so callers
        can distinguish an incomplete response from a successful completion.
        """
        if self._stream_cancel_event is not None and self._stream_cancel_event.is_set():
            return
        payload = self._payload(messages, stream=True, max_tokens=max_tokens)
        try:
            with httpx.Client(timeout=self.timeout) as c:
                with c.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
                    with self._stream_lock:
                        self._active_stream_response = r
                    try:
                        if self._stream_cancel_event is not None and self._stream_cancel_event.is_set():
                            return
                        r.raise_for_status()
                        for line in r.iter_lines():
                            if self._stream_cancel_event is not None and self._stream_cancel_event.is_set():
                                return
                            if not line or not line.startswith("data:"):
                                continue
                            chunk = line[len("data:"):].strip()
                            if not chunk or chunk == _DONE:
                                continue
                            try:
                                data = json.loads(chunk)
                            except json.JSONDecodeError:
                                continue
                            choices = data.get("choices") or []
                            if not choices:
                                continue
                            piece = (choices[0].get("delta") or {}).get("content")
                            if piece:
                                yield piece
                    finally:
                        with self._stream_lock:
                            if self._active_stream_response is r:
                                self._active_stream_response = None
        except Exception as exc:
            raise ProviderStreamError("LLM provider stream failed") from exc

    def complete_json(self, system: str, user: str, *, timeout: int | float | None = None) -> Optional[Dict[str, Any]]:
        """Schema-ish JSON completion for extraction. None on any failure."""
        payload = self._payload(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            stream=False,
            max_tokens=self.max_tokens,
        )
        payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=timeout if timeout is not None else self.timeout) as c:
                r = c.post(f"{self.base_url}/chat/completions", json=payload)
                r.raise_for_status()
                return json.loads(r.json()["choices"][0]["message"]["content"])
        except Exception:
            return None
