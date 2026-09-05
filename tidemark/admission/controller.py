"""Engine-local admission: fit at most one interval per scheduler iteration.

Interval sizes come from a small fixed set, ``D = {256, 512, 1024}``, trading
preemptibility against per-request overhead. ``Idle`` admits the largest
``delta in D`` with ``delta <= min(delta_max, X_t)``; ``Mixed`` admits at most
one interval with the same bound; ``Blocked`` admits nothing. A remaining lag
smaller than the smallest interval is admitted as-is so a frontier can reach
the end of the history.

The controller is re-evaluated on every scheduler iteration, not once per
ticket. An interval in flight when a foreground request lands is stopped at
the next scheduler boundary and reported as cancelled; it does not commit.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Hashable, Optional, Sequence, Tuple

from tidemark.admission.guard import AdmissionMode, TpotGuard, classify_mode
from tidemark.admission.safe_budget import SafeBudgetInputs, safe_budget

INTERVAL_SET: Tuple[int, ...] = (256, 512, 1024)


@dataclass(frozen=True)
class StepState:
    """What the engine knows at the top of one scheduler iteration."""

    step_id: Hashable
    token_budget: int
    decode_tokens: int
    prefill_tokens: int
    kv_free_blocks: int
    kv_total_blocks: int
    waiting_foreground: int = 0
    new_foreground_arrivals: int = 0
    last_tpot_ms: Optional[float] = None
    block_size: int = 16


@dataclass(frozen=True)
class AdmissionDecision:
    step_id: Hashable
    mode: AdmissionMode
    safe_budget_tokens: int
    admitted_delta: int
    reason: str
    guard_ewma_ms: Optional[float]
    guard_threshold_ms: Optional[float]
    kv_utilisation: float

    @property
    def admitted(self) -> bool:
        return self.admitted_delta > 0

    def as_log_row(self) -> Dict[str, object]:
        return {
            "step": self.step_id,
            "mode": self.mode.value,
            "safe_budget": self.safe_budget_tokens,
            "admitted": self.admitted_delta,
            "reason": self.reason,
            "tpot_ewma_ms": self.guard_ewma_ms,
            "tpot_threshold_ms": self.guard_threshold_ms,
            "kv_util": round(self.kv_utilisation, 4),
        }


@dataclass
class _InFlight:
    ticket_id: str
    delta: int
    admitted_step: Hashable
    started_at: float = field(default_factory=time.monotonic)


class EngineLocalAdmission:
    """One instance per engine. Foreground first, then at most one interval."""

    def __init__(
        self,
        *,
        intervals: Sequence[int] = INTERVAL_SET,
        x_max: int = 1024,
        kv_headroom: float = 0.08,
        guard: Optional[TpotGuard] = None,
        cancel_on_arrival: bool = True,
    ) -> None:
        if not intervals or any(d <= 0 for d in intervals):
            raise ValueError("intervals must be positive")
        self.intervals = tuple(sorted(int(d) for d in intervals))
        self.x_max = int(x_max)
        self.kv_headroom = float(kv_headroom)
        self.guard = guard or TpotGuard()
        self.cancel_on_arrival = cancel_on_arrival
        self._inflight: Optional[_InFlight] = None
        self._last_step: Optional[Hashable] = None
        self.counters: Dict[str, int] = {m.value: 0 for m in AdmissionMode}
        self.counters.update(admitted=0, admitted_tokens=0, cancelled=0)

    # ---------------------------------------------------------------- state
    @property
    def inflight_ticket(self) -> Optional[str]:
        return None if self._inflight is None else self._inflight.ticket_id

    def complete(self, ticket_id: str) -> None:
        if self._inflight is not None and self._inflight.ticket_id == ticket_id:
            self._inflight = None

    # -------------------------------------------------------------- iterate
    def observe(self, step: StepState) -> Tuple[AdmissionMode, int]:
        """Update the guard and classify the iteration. Called once per step."""
        if step.last_tpot_ms is not None and step.decode_tokens > 0:
            self.guard.observe(step.last_tpot_ms)
        x_t = safe_budget(
            SafeBudgetInputs(
                token_budget=step.token_budget,
                decode_tokens=step.decode_tokens,
                prefill_tokens=step.prefill_tokens,
                kv_free_blocks=step.kv_free_blocks,
                kv_total_blocks=step.kv_total_blocks,
                block_size=step.block_size,
                kv_headroom=self.kv_headroom,
                x_max=self.x_max,
            )
        )
        mode = classify_mode(
            decode_tokens=step.decode_tokens,
            prefill_tokens=step.prefill_tokens,
            safe_budget_tokens=x_t,
            guard_ok=self.guard.ok,
        )
        self.counters[mode.value] += 1
        self._last_step = step.step_id
        return mode, x_t

    def should_cancel(self, step: StepState) -> Optional[str]:
        """If a foreground request arrived while an interval is in flight, stop it."""
        if self._inflight is None or not self.cancel_on_arrival:
            return None
        if step.new_foreground_arrivals > 0 and step.step_id != self._inflight.admitted_step:
            tid = self._inflight.ticket_id
            self._inflight = None
            self.counters["cancelled"] += 1
            return tid
        return None

    def decide(self, step: StepState, *, ticket_id: Optional[str], delta_max: int) -> AdmissionDecision:
        """Pick the admitted size for the pending ticket, or none."""
        mode, x_t = self.observe(step)
        util = 1.0 - step.kv_free_blocks / step.kv_total_blocks
        base = dict(
            step_id=step.step_id,
            mode=mode,
            safe_budget_tokens=x_t,
            guard_ewma_ms=self.guard.ewma_ms,
            guard_threshold_ms=self.guard.threshold_ms,
            kv_utilisation=util,
        )
        if ticket_id is None or delta_max <= 0:
            return AdmissionDecision(admitted_delta=0, reason="no_ticket", **base)
        if self._inflight is not None:
            return AdmissionDecision(admitted_delta=0, reason="interval_inflight", **base)
        if mode is AdmissionMode.BLOCKED:
            reason = "guard" if not self.guard.ok else "no_safe_budget"
            return AdmissionDecision(admitted_delta=0, reason=reason, **base)
        bound = min(delta_max, x_t)
        if bound <= 0:
            return AdmissionDecision(admitted_delta=0, reason="no_safe_budget", **base)
        delta = self._fit(bound, delta_max)
        if delta <= 0:
            return AdmissionDecision(admitted_delta=0, reason="no_interval_fits", **base)
        self._inflight = _InFlight(ticket_id=ticket_id, delta=delta, admitted_step=step.step_id)
        self.counters["admitted"] += 1
        self.counters["admitted_tokens"] += delta
        return AdmissionDecision(admitted_delta=delta, reason=f"fit_{mode.value}", **base)

    def _fit(self, bound: int, delta_max: int) -> int:
        fitting = [d for d in self.intervals if d <= bound]
        if fitting:
            return max(fitting)
        # The remaining lag is shorter than the smallest interval: take it whole
        # if it fits the budget so the frontier can reach the history's end.
        if delta_max < self.intervals[0] and delta_max <= bound:
            return delta_max
        return 0


__all__ = ["INTERVAL_SET", "StepState", "AdmissionDecision", "EngineLocalAdmission"]
