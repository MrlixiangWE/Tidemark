"""Marginal-value ranking and the cross-tenant ranking epoch (Algorithm 1).

An interval ``a = (s, m, c, delta)`` that advances frontier ``F`` removes
expected latency

    B(a) = p_{s,m} * [ C_m(|H| - F) - C_m(|H| - F - delta) ]

and costs

    R(a) = T_m^{bg}(delta) + lambda_M * M_m(delta)

so its score is ``B(a) / max(R(a), eps)``. Under linear fits both terms are
proportional to ``delta`` and the score is a per-token marginal value that
does not depend on the interval length, which is what lets the engine pick
the admitted size later without disturbing the ranking.

Ranking by score alone would concentrate background work on whichever tenant
has the most active sessions, so each tenant ``u`` is capped on two axes: at
most ``kappa_u`` outstanding tickets, and at most ``beta_u * G_total`` of the
aggregate background budget, where ``G_total`` sums the safe budgets engines
reported over the epoch in ``tau_bg``-weighted compute time. Tenants skipped
while eligible accumulate aging so a stream of high-score arrivals cannot
starve them indefinitely.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tidemark.catalog.frontier import FrontierKey
from tidemark.scheduler.cost_model import EngineRates

EPSILON_MS = 1e-3


def expected_benefit_ms(rates: EngineRates, p_future: float, lag: int, delta: int) -> float:
    """B(a): expected critical-path latency removed by advancing ``delta`` tokens."""
    before = rates.critical_path_ms(lag)
    after = rates.critical_path_ms(max(0, lag - delta))
    return max(0.0, p_future) * (before - after)


def resource_cost_ms(rates: EngineRates, delta: int, lambda_mem_ms_per_gib: float) -> float:
    """R(a): background compute time plus KV occupancy converted to time units."""
    gib = rates.kv_bytes(delta) / float(1 << 30)
    return rates.background_ms(delta) + lambda_mem_ms_per_gib * gib


def score(benefit_ms: float, cost_ms: float, eps: float = EPSILON_MS) -> float:
    return benefit_ms / max(cost_ms, eps)


@dataclass
class Candidate:
    key: FrontierKey
    tenant_id: str
    engine_id: str
    frontier: int
    lag: int
    delta_max: int
    p_future: float
    benefit_ms: float
    cost_ms: float
    score: float
    aging: float = 0.0

    @property
    def effective_score(self) -> float:
        return self.score * (1.0 + self.aging)


@dataclass(frozen=True)
class TenantCaps:
    kappa: int = 2        # outstanding admitted tickets per tenant
    beta: float = 0.35    # share of G_total per tenant

    def __post_init__(self) -> None:
        if self.kappa < 1:
            raise ValueError("kappa must be at least 1")
        if not 0.0 < self.beta <= 1.0:
            raise ValueError("beta must lie in (0, 1]")


@dataclass
class TenantLedger:
    """Per-tenant accounting the epoch consults and charges."""

    caps: TenantCaps
    outstanding: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    charged_ms: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    aging: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    aging_step: float = 0.1
    aging_cap: float = 1.0

    def can_issue(self, tenant_id: str, cost_ms: float, g_total_ms: float) -> Tuple[bool, str]:
        if self.outstanding[tenant_id] >= self.caps.kappa:
            return False, "tenant_ticket_cap"
        if self.charged_ms[tenant_id] + cost_ms > self.caps.beta * g_total_ms + 1e-9:
            return False, "tenant_budget_share"
        return True, ""

    def charge(self, tenant_id: str, cost_ms: float) -> None:
        self.outstanding[tenant_id] += 1
        self.charged_ms[tenant_id] += cost_ms
        self.aging[tenant_id] = 0.0

    def settle(self, tenant_id: str, predicted_ms: float, actual_ms: Optional[float] = None) -> None:
        """A ticket reached a terminal state. Release its slot; keep the charge
        at what was actually spent (or nothing if it never ran)."""
        self.outstanding[tenant_id] = max(0, self.outstanding[tenant_id] - 1)
        if actual_ms is None:
            self.charged_ms[tenant_id] = max(0.0, self.charged_ms[tenant_id] - predicted_ms)
        else:
            self.charged_ms[tenant_id] += actual_ms - predicted_ms
            self.charged_ms[tenant_id] = max(0.0, self.charged_ms[tenant_id])

    def age(self, skipped_tenants: Iterable[str]) -> None:
        for t in skipped_tenants:
            self.aging[t] = min(self.aging_cap, self.aging[t] + self.aging_step)

    def new_epoch(self) -> None:
        """Budget charges are per epoch; outstanding tickets and aging carry over."""
        self.charged_ms.clear()


@dataclass
class EpochOutcome:
    issued: List[Candidate]
    skipped: Dict[FrontierKey, str]
    g_total_ms: float
    epoch: int


class RankingEpoch:
    """One run of Algorithm 1 over a set of candidates."""

    def __init__(self, ledger: TenantLedger) -> None:
        self.ledger = ledger
        self.epoch = 0

    def run(
        self,
        candidates: Sequence[Candidate],
        *,
        engines_busy: Set[str],
        g_total_ms: float,
    ) -> EpochOutcome:
        """Walk candidates in descending score, issue at most one per engine.

        ``engines_busy`` names engines that already carry an in-flight ticket.
        ``g_total_ms`` is the aggregate safe budget in tau_bg-weighted ms.
        """
        self.epoch += 1
        self.ledger.new_epoch()
        for c in candidates:
            c.aging = self.ledger.aging[c.tenant_id]
        ordered = sorted(
            candidates,
            key=lambda c: (-c.effective_score, c.tenant_id, c.key.session_id, c.key.model_id),
        )
        issued: List[Candidate] = []
        skipped: Dict[FrontierKey, str] = {}
        busy = set(engines_busy)
        skipped_eligible: Set[str] = set()
        for c in ordered:
            if not math.isfinite(c.score) or c.score <= 0.0:
                skipped[c.key] = "no_value"
                continue
            if c.engine_id in busy:
                skipped[c.key] = "engine_inflight"
                skipped_eligible.add(c.tenant_id)
                continue
            ok, why = self.ledger.can_issue(c.tenant_id, c.cost_ms, g_total_ms)
            if not ok:
                skipped[c.key] = why
                continue
            issued.append(c)
            busy.add(c.engine_id)
            self.ledger.charge(c.tenant_id, c.cost_ms)
        # Queue aging: tenants skipped while eligible climb next epoch.
        self.ledger.age(t for t in skipped_eligible if self.ledger.outstanding[t] < self.ledger.caps.kappa)
        return EpochOutcome(issued=issued, skipped=skipped, g_total_ms=g_total_ms, epoch=self.epoch)


__all__ = [
    "EPSILON_MS",
    "Candidate",
    "EpochOutcome",
    "RankingEpoch",
    "TenantCaps",
    "TenantLedger",
    "expected_benefit_ms",
    "resource_cost_ms",
    "score",
]
