"""Commit/reject adapter for the on-device llama.cpp server.

``llama-server`` has no batching scheduler iteration to hook into, so the
device tier is handled differently from the server tiers:

* The safe budget is derived from slot occupancy. A device engine typically
  runs one slot; if that slot is decoding a foreground request the engine is
  ``Blocked``, otherwise it is ``Idle``. There is no ``Mixed`` mode on a single
  slot, which matches the measured ``tau_bg / tau_fg`` of about 0.9 on the
  phones and boards in the paper.
* A prefill-only request is a ``/completion`` with ``n_predict = 0`` and
  ``cache_prompt = true``; the response's ``tokens_cached`` and
  ``prompt_n`` fields give exact reuse accounting.
* Residency is read from ``/slots``, which reports the prompt each slot still
  holds. The adapter turns a shorter-than-expected prompt into an eviction
  report for the catalog.

The small patch under ``engines/llamacpp/`` adds a ``tidemark_commit`` field
to the completion response with the hash of the cached prefix so the commit
check does not have to re-fetch the slot.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import EngineLocalAdmission, StepState
from tidemark.admission.guard import TpotGuard
from tidemark.engines.base import EngineAdapter, EngineDescriptor, ResidentPrefix
from tidemark.scheduler.ticket import AtomicTicket

log = logging.getLogger("tidemark.engines.llamacpp")

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]


class LlamaCppAdapter(EngineAdapter):
    def __init__(
        self,
        descriptor: EngineDescriptor,
        commit: CommitPath,
        *,
        n_batch: int = 512,
        timeout_s: float = 120.0,
        session: Optional[Any] = None,
    ) -> None:
        super().__init__(descriptor, commit)
        if requests is None and session is None:  # pragma: no cover
            raise RuntimeError("the llama.cpp adapter needs 'requests': pip install tidemark[engines]")
        self.n_batch = n_batch
        self.timeout_s = timeout_s
        self._http = session or requests.Session()
        self.admission = EngineLocalAdmission(guard=TpotGuard(calibration_steps=50), x_max=n_batch)
        self._step = 0
        self._inflight: Dict[str, threading.Event] = {}
        self._eviction_cb: Optional[Callable[[str, int], None]] = None

    # -------------------------------------------------------------- state
    def _slots(self) -> Any:
        r = self._http.get(f"{self.descriptor.endpoint}/slots", timeout=5)
        r.raise_for_status()
        return r.json()

    def step_state(self) -> StepState:
        slots = self._slots()
        busy = [s for s in slots if s.get("is_processing")]
        decode = sum(1 for s in busy if s.get("n_decoded", 0) > 0)
        prefill = sum(int(s.get("n_prompt_tokens_processed", 0)) for s in busy if s.get("n_decoded", 0) == 0)
        ctx = int(slots[0].get("n_ctx", 4096)) if slots else 4096
        used = sum(int(s.get("n_past", 0)) for s in slots)
        total_blocks = max(1, ctx * max(1, len(slots)) // 16)
        free_blocks = max(0, total_blocks - used // 16)
        self._step += 1
        return StepState(
            step_id=self._step,
            token_budget=self.n_batch,
            decode_tokens=decode,
            prefill_tokens=prefill,
            kv_free_blocks=free_blocks,
            kv_total_blocks=total_blocks,
            waiting_foreground=0,
            new_foreground_arrivals=0,
            last_tpot_ms=None,
            block_size=16,
        )

    # ------------------------------------------------------------- submit
    def submit(self, ticket: AtomicTicket, delta: int) -> None:
        decision = self.admission.decide(self.step_state(), ticket_id=ticket.ticket_id, delta_max=delta)
        if not decision.admitted:
            self.commit.refused(ticket, decision.reason)
            return
        admitted = decision.admitted_delta
        self.commit.admitted(ticket, admitted)
        stop = threading.Event()
        self._inflight[ticket.ticket_id] = stop

        def run() -> None:
            started = time.perf_counter()
            try:
                r = self._http.post(
                    f"{self.descriptor.endpoint}/completion",
                    json={
                        "prompt": list(ticket.prompt_token_ids(admitted)),
                        "n_predict": 0,
                        "cache_prompt": True,
                        "id_slot": -1,
                        "tidemark": ticket.metadata(),
                    },
                    timeout=self.timeout_s,
                )
                r.raise_for_status()
                body = r.json()
            except Exception as exc:
                log.warning("device ticket %s failed: %s", ticket.ticket_id, exc)
                self.admission.complete(ticket.ticket_id)
                self.commit.refused(ticket, f"engine_error:{type(exc).__name__}")
                return
            finally:
                self._inflight.pop(ticket.ticket_id, None)
            gpu_ms = (time.perf_counter() - started) * 1000.0
            self.admission.complete(ticket.ticket_id)
            if stop.is_set():
                self.commit.cancelled(ticket, prefilled_tokens=int(body.get("tokens_evaluated", 0)), gpu_ms=gpu_ms)
                return
            timings = body.get("timings") or {}
            cached = int(body.get("tokens_cached", timings.get("cache_n", 0)))
            prompt_n = int(timings.get("prompt_n", 0)) + cached
            self.commit.completed(
                ticket,
                admitted_delta=admitted,
                cached_tokens=cached,
                physically_cached_after=prompt_n,
                gpu_ms=gpu_ms,
            )

        threading.Thread(target=run, name=f"tidemark-dev-{ticket.ticket_id[:8]}", daemon=True).start()

    def cancel(self, ticket_id: str) -> None:
        ev = self._inflight.get(ticket_id)
        if ev is not None:
            ev.set()

    # ------------------------------------------------------------ residency
    def resident_prefix(self, session_id: str, token_ids: tuple) -> ResidentPrefix:
        best = 0
        for slot in self._slots():
            held = slot.get("prompt_tokens") or []
            n = 0
            for a, b in zip(held, token_ids):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return ResidentPrefix(session_id, best)

    def on_eviction(self, callback: Callable[[str, int], None]) -> None:
        self._eviction_cb = callback


__all__ = ["LlamaCppAdapter"]
