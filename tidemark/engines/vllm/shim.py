"""vLLM V1 admission shim.

The patch under ``engines/vllm/`` adds three calls to
``vllm/v1/core/sched/scheduler.py``:

1. ``shim.begin_step(stats)`` right after foreground decode and prefill have
   been scheduled and before the scheduler looks at the waiting queue again.
   ``stats`` carries the token budget, the foreground tokens already selected,
   KV block counts, waiting-queue depth and the last decode step's TPOT.
2. ``shim.size_background(request)`` when the scheduler considers a request
   tagged ``tidemark`` from the waiting queue. It returns the number of new
   tokens the scheduler may compute for it in this iteration, or 0.
3. ``shim.end_step(finished, preempted)`` after the step's outputs are known,
   which turns request completions into ticket results.

The shim is deliberately dumb about vLLM internals; anything engine-specific
is done in the patch and passed in as plain numbers. That keeps this file
testable without a GPU and keeps the patch small (about 120 lines).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple

from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import AdmissionDecision, EngineLocalAdmission, StepState
from tidemark.engines.base import EngineAdapter, EngineDescriptor, ResidentPrefix
from tidemark.engines.vllm.prefill_only import PrefillOnlyRequest
from tidemark.scheduler.ticket import AtomicTicket

log = logging.getLogger("tidemark.engines.vllm")


@dataclass(frozen=True)
class SchedulerStepStats:
    """Plain-number view of one vLLM scheduler iteration."""

    step_id: Hashable
    max_num_batched_tokens: int
    scheduled_decode_tokens: int
    scheduled_prefill_tokens: int
    waiting_foreground: int
    new_foreground_arrivals: int
    kv_free_blocks: int
    kv_total_blocks: int
    block_size: int
    last_tpot_ms: Optional[float] = None


@dataclass
class _Pending:
    ticket: AtomicTicket
    admitted_delta: int = 0
    computed_tokens: int = 0
    started_at: Optional[float] = None
    gpu_ms: float = 0.0


class VllmAdmissionShim:
    """Lives inside the engine process; one per scheduler."""

    def __init__(self, engine_id: str, commit: CommitPath, admission: Optional[EngineLocalAdmission] = None) -> None:
        self.engine_id = engine_id
        self.commit = commit
        self.admission = admission or EngineLocalAdmission()
        self._pending: Dict[str, _Pending] = {}   # ticket_id -> state
        self._by_request: Dict[str, str] = {}      # vLLM request_id -> ticket_id
        self._last_decision: Optional[AdmissionDecision] = None
        self._lock = threading.Lock()
        self.step_log: Optional[Callable[[Mapping[str, Any]], None]] = None

    # ------------------------------------------------------------- tickets
    def enqueue(self, ticket: AtomicTicket, request_id: str) -> None:
        with self._lock:
            self._pending[ticket.ticket_id] = _Pending(ticket)
            self._by_request[request_id] = ticket.ticket_id

    def cancel(self, ticket_id: str) -> None:
        with self._lock:
            p = self._pending.pop(ticket_id, None)
        if p is not None:
            self.admission.complete(ticket_id)
            self.commit.cancelled(p.ticket, prefilled_tokens=p.computed_tokens, gpu_ms=p.gpu_ms)

    # -------------------------------------------------------- step hooks
    def begin_step(self, stats: SchedulerStepStats) -> StepState:
        step = StepState(
            step_id=stats.step_id,
            token_budget=stats.max_num_batched_tokens,
            decode_tokens=stats.scheduled_decode_tokens,
            prefill_tokens=stats.scheduled_prefill_tokens,
            kv_free_blocks=stats.kv_free_blocks,
            kv_total_blocks=stats.kv_total_blocks,
            waiting_foreground=stats.waiting_foreground,
            new_foreground_arrivals=stats.new_foreground_arrivals,
            last_tpot_ms=stats.last_tpot_ms,
            block_size=stats.block_size,
        )
        to_cancel = self.admission.should_cancel(step)
        if to_cancel is not None:
            self.cancel(to_cancel)
        self._step = step
        return step

    def size_background(self, request_id: str) -> int:
        """How many new tokens the scheduler may compute for this request now."""
        with self._lock:
            tid = self._by_request.get(request_id)
            p = self._pending.get(tid) if tid else None
        if p is None:
            return 0
        if p.admitted_delta:
            # Already admitted in an earlier iteration; let it finish its interval.
            return max(0, p.admitted_delta - p.computed_tokens)
        decision = self.admission.decide(self._step, ticket_id=p.ticket.ticket_id, delta_max=p.ticket.delta_max)
        self._last_decision = decision
        if self.step_log is not None:
            self.step_log(decision.as_log_row())
        if not decision.admitted:
            return 0
        p.admitted_delta = decision.admitted_delta
        p.started_at = time.perf_counter()
        self.commit.admitted(p.ticket, decision.admitted_delta)
        return decision.admitted_delta

    def account(self, request_id: str, computed_tokens: int, step_ms: float) -> None:
        with self._lock:
            tid = self._by_request.get(request_id)
            p = self._pending.get(tid) if tid else None
        if p is not None:
            p.computed_tokens += computed_tokens
            p.gpu_ms += step_ms

    def end_step(self, finished: Mapping[str, Tuple[int, int]], preempted: Tuple[str, ...] = ()) -> None:
        """``finished`` maps request_id -> (cached_tokens, resident_prefix_after)."""
        for rid in preempted:
            tid = self._by_request.pop(rid, None)
            if tid:
                self.cancel(tid)
        for rid, (cached, resident_after) in finished.items():
            tid = self._by_request.pop(rid, None)
            if not tid:
                continue
            with self._lock:
                p = self._pending.pop(tid, None)
            if p is None:
                continue
            self.admission.complete(tid)
            if p.admitted_delta == 0:
                self.commit.refused(p.ticket, "never_admitted")
                continue
            self.commit.completed(
                p.ticket,
                admitted_delta=p.admitted_delta,
                cached_tokens=cached,
                physically_cached_after=resident_after,
                gpu_ms=p.gpu_ms,
            )

    def diagnostics(self) -> Dict[str, Any]:
        return {
            "engine": self.engine_id,
            "pending": len(self._pending),
            "counters": dict(self.admission.counters),
            "guard": {
                "tpot_ref_ms": self.admission.guard.tpot_ref_ms,
                "ewma_ms": self.admission.guard.ewma_ms,
                "gamma": self.admission.guard.gamma,
            },
        }


class VllmEngineAdapter(EngineAdapter):
    """Scheduler-side adapter: ships tickets to a vLLM engine over HTTP.

    The engine-side shim does the admission; this adapter only builds the
    prefill-only request and reads back the usage block. When the shim is not
    installed the request still works as a plain low-priority completion, so a
    stock vLLM behaves like the whole-suffix prefetch baseline.
    """

    def __init__(self, descriptor: EngineDescriptor, commit: CommitPath, client: Any) -> None:
        super().__init__(descriptor, commit)
        self.client = client
        self._inflight: Dict[str, threading.Thread] = {}

    def step_state(self) -> StepState:  # pragma: no cover - only meaningful in-process
        raise NotImplementedError("step state is observed inside the engine by VllmAdmissionShim")

    def submit(self, ticket: AtomicTicket, delta: int) -> None:
        req = PrefillOnlyRequest(ticket, delta, self.descriptor.model_id)

        def run() -> None:
            try:
                body = self.client.completion(req.payload())
            except Exception as exc:  # network / engine error: nothing commits
                log.warning("ticket %s failed on %s: %s", ticket.ticket_id, self.engine_id, exc)
                self.commit.refused(ticket, f"engine_error:{type(exc).__name__}")
                return
            usage = body.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            cached = int(details.get("cached_tokens", 0))
            prompt = int(usage.get("prompt_tokens", 0))
            self.commit.completed(
                ticket,
                admitted_delta=delta,
                cached_tokens=cached,
                physically_cached_after=prompt,
                gpu_ms=float(body.get("_e2e_ms", 0.0)),
            )

        t = threading.Thread(target=run, name=f"tidemark-{ticket.ticket_id[:8]}", daemon=True)
        self._inflight[ticket.ticket_id] = t
        t.start()

    def cancel(self, ticket_id: str) -> None:
        # vLLM aborts a request when its client disconnects; the shim reports the cancel.
        self._inflight.pop(ticket_id, None)

    def resident_prefix(self, session_id: str, token_ids: tuple) -> ResidentPrefix:
        # A zero-token completion is not allowed, so probe with max_tokens=1 and
        # read the cached count. Cheap relative to a prefill, and exact.
        body = self.client.completion({"model": self.descriptor.model_id, "prompt": list(token_ids), "max_tokens": 1, "temperature": 0})
        details = (body.get("usage") or {}).get("prompt_tokens_details") or {}
        return ResidentPrefix(session_id, int(details.get("cached_tokens", 0)))


__all__ = ["SchedulerStepStats", "VllmAdmissionShim", "VllmEngineAdapter"]
