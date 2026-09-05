"""Minimal OpenAI-compatible client that speaks token ids.

The client insists on request-level ``prompt_tokens_details`` in the usage
block. Without it we cannot tell how much of a prompt was served from cache,
and every number Tidemark reports about reuse would be a guess.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

try:  # requests is an optional dependency; the simulator never needs it.
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CompletionUsage:
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    e2e_ms: float
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None

    @property
    def uncached_tokens(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    @property
    def cached_ratio(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0


def parse_usage(usage: Dict[str, Any], e2e_ms: float, ttft_ms: Optional[float] = None, tpot_ms: Optional[float] = None) -> CompletionUsage:
    prompt = usage.get("prompt_tokens")
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if not isinstance(prompt, int) or not isinstance(cached, int):
        raise RuntimeError(
            "engine response lacks request-level cached_tokens; "
            "start vLLM with --enable-prefix-caching and a version >= 0.6.3"
        )
    if not 0 <= cached <= prompt:
        raise RuntimeError(f"invalid cached_tokens={cached} for prompt_tokens={prompt}")
    return CompletionUsage(
        prompt_tokens=prompt,
        cached_tokens=cached,
        completion_tokens=int(usage.get("completion_tokens") or 0),
        e2e_ms=e2e_ms,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
    )


class TokenIdClient:
    def __init__(self, base_url: str, model: str, timeout_s: float = 300.0) -> None:
        if requests is None:  # pragma: no cover
            raise RuntimeError("the vLLM client needs the 'requests' package: pip install tidemark[engines]")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self._local = threading.local()

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.trust_env = False
            self._local.session = s
        return s

    def completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        started = time.perf_counter()
        r = self._session().post(f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        body = r.json()
        body["_e2e_ms"] = (time.perf_counter() - started) * 1000.0
        return body

    def foreground(self, prompt_token_ids: Sequence[int], *, max_tokens: int, priority: int = 0, seed: int = 0) -> CompletionUsage:
        """A user-visible request, streamed so TTFT and TPOT can be measured."""
        payload = {
            "model": self.model,
            "prompt": list(prompt_token_ids),
            "max_tokens": max_tokens,
            "temperature": 0,
            "seed": seed,
            "priority": priority,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        r = self._session().post(f"{self.base_url}/v1/completions", json=payload, timeout=self.timeout_s, stream=True)
        r.raise_for_status()
        first: Optional[float] = None
        last: Optional[float] = None
        n = 0
        usage: Dict[str, Any] = {}
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode() if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            import json

            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                if choice.get("text"):
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n += 1
        end = time.perf_counter()
        ttft = None if first is None else (first - started) * 1000.0
        tpot = None if (first is None or last is None or n < 2) else (last - first) * 1000.0 / (n - 1)
        return parse_usage(usage, (end - started) * 1000.0, ttft, tpot)

    def metrics(self) -> str:
        r = self._session().get(f"{self.base_url}/metrics", timeout=10)
        r.raise_for_status()
        return r.text


__all__ = ["TokenIdClient", "CompletionUsage", "parse_usage"]
