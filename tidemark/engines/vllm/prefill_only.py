"""The prefill-only request class.

A background task carries token ids, frontier identity, revision, generation
and a maximum chunk boundary. It enters the normal destination prefill path
but bypasses sampling and user response construction, and its completion
returns the physically cached token count and commit status to the scheduler.

On vLLM this is expressed as an ordinary ``/v1/completions`` call with
``max_tokens=1`` at the lowest priority, plus a metadata block the patched
scheduler recognises. The single generated token is discarded; what we want is
the request-level ``prompt_tokens_details.cached_tokens`` accounting in the
response, which tells us exactly how much of the prefix was found resident and
how much was computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from tidemark.scheduler.ticket import AtomicTicket

BACKGROUND_PRIORITY = 1_000_000  # vLLM: larger value == lower priority


@dataclass(frozen=True)
class PrefillOnlyRequest:
    ticket: AtomicTicket
    delta: int
    model: str

    @property
    def request_id(self) -> str:
        t = self.ticket
        return f"tidemark:{t.model_id}:{t.session_id}:g{t.generation}:f{t.frontier + self.delta}:{t.ticket_id[:8]}"

    @property
    def prompt_token_ids(self) -> Sequence[int]:
        return self.ticket.prompt_token_ids(self.delta)

    def payload(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "prompt": list(self.prompt_token_ids),
            "max_tokens": 1,
            "temperature": 0.0,
            "priority": BACKGROUND_PRIORITY,
            "stream": False,
            "tidemark": {
                **self.ticket.metadata(),
                "admitted_delta": self.delta,
                "request_id": self.request_id,
            },
        }


__all__ = ["PrefillOnlyRequest", "BACKGROUND_PRIORITY"]
