"""The global frontier scheduler process.

This module ties the catalog, the predictor, the rate table and the ranking
epoch together. It is engine-agnostic: engines are known only by id, tier and
the ``(model, runtime_config)`` they serve, and every interaction with them
goes through :class:`~tidemark.scheduler.ticket.AtomicTicket` and
:class:`~tidemark.scheduler.ticket.TicketResult`.

The scheduler is off the foreground path. A request goes from the router to
its engine directly; the scheduler only learns about it afterwards through
:meth:`GlobalFrontierScheduler.on_turn_served`, and a ticket it issues only
proposes background work the destination engine is free to refuse.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Tuple

from tidemark.catalog.frontier import FrontierKey, Placement, VersionedFrontierCatalog
from tidemark.catalog.history import SessionHistory
from tidemark.scheduler.cost_model import RateTable
from tidemark.scheduler.predictor import DestinationPredictor
from tidemark.scheduler.ranking import (
    Candidate,
    EpochOutcome,
    RankingEpoch,
    TenantCaps,
    TenantLedger,
    expected_benefit_ms,
    resource_cost_ms,
    score,
)
from tidemark.scheduler.ticket import AtomicTicket, TicketResult, TicketStatus, new_ticket_id

log = logging.getLogger("tidemark.scheduler")


@dataclass(frozen=True)
class EngineSpec:
    engine_id: str
    model_id: str
    runtime_config: str
    tier: str


@dataclass
class SchedulerConfig:
    delta_max: int = 1024
    lambda_mem_ms_per_gib: float = 64.0
    alpha: float = 1.0
    tenant_caps: TenantCaps = field(default_factory=TenantCaps)
    ticket_lease_s: float = 30.0
    # Ablation switches. Off in production; the evaluation flips them one at a time.
    ablate_versioned_keys: bool = False
    ablate_bounded_intervals: bool = False
    ablate_tau_bg_weighting: bool = False
    round_robin_ranking: bool = False


TicketSink = Callable[[AtomicTicket], None]


class GlobalFrontierScheduler:
    """Ranks lagging frontiers across tenants and tiers; issues one ticket per engine."""

    def __init__(
        self,
        catalog: VersionedFrontierCatalog,
        rates: RateTable,
        predictor: DestinationPredictor,
        engines: Iterable[EngineSpec],
        config: Optional[SchedulerConfig] = None,
        sink: Optional[TicketSink] = None,
    ) -> None:
        self.catalog = catalog
        self.rates = rates
        self.predictor = predictor
        self.config = config or SchedulerConfig()
        self._engines: Dict[str, EngineSpec] = {e.engine_id: e for e in engines}
        self._engine_for: Dict[Tuple[str, str], str] = {
            (e.model_id, e.runtime_config): e.engine_id for e in self._engines.values()
        }
        self._ledger = TenantLedger(self.config.tenant_caps)
        self._epoch = RankingEpoch(self._ledger)
        self._inflight: Dict[str, AtomicTicket] = {}          # ticket_id -> ticket
        self._inflight_by_engine: Dict[str, str] = {}         # engine_id -> ticket_id
        self._safe_budget_tokens: Dict[str, int] = {}         # engine -> X̄ accumulated this epoch
        self._sink = sink
        self._lock = threading.RLock()
        self.stats = SchedulerStats()

    # ------------------------------------------------------------- wiring
    def set_sink(self, sink: TicketSink) -> None:
        self._sink = sink

    def engine_for(self, model_id: str, runtime_config: str) -> str:
        return self._engine_for[(model_id, runtime_config)]

    def compatible_pairs(self) -> List[Tuple[str, str]]:
        return sorted(self._engine_for)

    # -------------------------------------------------------------- events
    def on_session_start(self, history: SessionHistory) -> None:
        self.catalog.register_session(history, self.compatible_pairs())

    def on_session_end(self, session_id: str) -> None:
        with self._lock:
            for tid, t in list(self._inflight.items()):
                if t.session_id == session_id:
                    self._retire(tid, TicketStatus.CANCELLED)
        self.catalog.forget_session(session_id)
        self.predictor.forget(session_id)

    def on_turn_served(
        self,
        session_id: str,
        *,
        served_model: str,
        runtime_config: str,
        resident_prefix: int,
        engine_id: Optional[str] = None,
        kv_bytes: int = 0,
    ) -> EpochOutcome:
        """The foreground request finished on ``served_model``.

        Advances the serving model's frontier through the same catalog rule as
        a background commit, marks every other model lagging, and runs a
        ranking epoch.
        """
        key = FrontierKey(session_id, served_model, runtime_config)
        engine = engine_id or self.engine_for(served_model, runtime_config)
        self.catalog.commit_foreground(key, resident_prefix=resident_prefix, engine_id=engine, kv_bytes=kv_bytes)
        self.catalog.mark_read(key)
        self.predictor.observe_turn(session_id, served_model)
        return self.run_epoch(exclude_model=served_model, exclude_session=session_id)

    def on_revision_transition(self, session_id: str, lcp_by_model: Mapping[str, int]) -> None:
        with self._lock:
            for tid, t in list(self._inflight.items()):
                if t.session_id == session_id:
                    self._retire(tid, TicketStatus.STALE)
        self.catalog.on_revision_transition(session_id, dict(lcp_by_model))

    def on_eviction(self, key: FrontierKey, resident_prefix: int) -> None:
        self.catalog.on_eviction(key, resident_prefix)

    def on_engine_transition(self, engine_id: str, safe_budget_tokens: int) -> Optional[EpochOutcome]:
        """An engine reported a state transition (idle gap opened, burst ended, ...)."""
        with self._lock:
            self._safe_budget_tokens[engine_id] = self._safe_budget_tokens.get(engine_id, 0) + max(0, safe_budget_tokens)
            if engine_id in self._inflight_by_engine:
                return None
        return self.run_epoch()

    def on_engine_lost(self, engine_id: str) -> None:
        with self._lock:
            tid = self._inflight_by_engine.get(engine_id)
            if tid:
                self._retire(tid, TicketStatus.CANCELLED)
        self.catalog.on_engine_lost(engine_id)

    def on_ticket_result(self, result: TicketResult) -> Optional[EpochOutcome]:
        """Terminal or intermediate report from an engine for one ticket."""
        with self._lock:
            ticket = self._inflight.get(result.ticket_id)
            if ticket is None:
                self.stats.unknown_results += 1
                return None
            if result.status is TicketStatus.ADMITTED:
                self.stats.admitted += 1
                return None
            if result.status is TicketStatus.COMMITTED:
                verdict = self.catalog.commit_background(
                    ticket.key,
                    ticket_id=ticket.ticket_id,
                    ticket_frontier=ticket.frontier,
                    ticket_generation=ticket.generation,
                    ticket_revision=ticket.revision,
                    admitted_delta=result.admitted_delta,
                    requested_delta=result.admitted_delta,
                    snapshot_hash=result.snapshot_hash,
                    engine_id=result.engine_id,
                    kv_bytes=result.kv_bytes,
                )
                if verdict.ok:
                    self.rates.observe_background(result.engine_id, result.admitted_delta, result.gpu_ms)
                    self._retire(ticket.ticket_id, TicketStatus.COMMITTED, actual_ms=result.gpu_ms)
                    self.stats.committed += 1
                    self.stats.committed_tokens += result.admitted_delta
                else:
                    self._retire(ticket.ticket_id, TicketStatus.STALE, actual_ms=result.gpu_ms)
                    self.stats.stale += 1
                    log.debug("ticket %s stale: %s", ticket.ticket_id, verdict.reason)
            else:
                self._retire(ticket.ticket_id, result.status, actual_ms=result.gpu_ms or None)
                if result.status is TicketStatus.CANCELLED:
                    self.stats.cancelled += 1
                elif result.status is TicketStatus.REFUSED:
                    self.stats.refused += 1
                else:
                    self.stats.expired += 1
        return self.run_epoch()

    # --------------------------------------------------------------- epoch
    def run_epoch(self, *, exclude_model: Optional[str] = None, exclude_session: Optional[str] = None) -> EpochOutcome:
        with self._lock:
            candidates = self._build_candidates(exclude_model=exclude_model, exclude_session=exclude_session)
            g_total = self._g_total_ms()
            busy = set(self._inflight_by_engine)
            if self.config.round_robin_ranking:
                for i, c in enumerate(candidates):
                    c.score = 1.0 + (len(candidates) - i) * 1e-6  # keep order, flatten value
            outcome = self._epoch.run(candidates, engines_busy=busy, g_total_ms=g_total)
            for c in outcome.issued:
                self._issue(c, outcome.epoch)
            self._safe_budget_tokens.clear()
            self.stats.epochs += 1
            return outcome

    def _g_total_ms(self) -> float:
        total = 0.0
        for engine_id, tokens in self._safe_budget_tokens.items():
            if engine_id in self.rates:
                total += self.rates.get(engine_id).background_ms(tokens)
        if total <= 0.0:
            # No engine reported a budget this epoch. Use one delta_max per
            # engine as a conservative stand-in so tenant shares stay meaningful.
            for spec in self._engines.values():
                if spec.engine_id in self.rates:
                    total += self.rates.get(spec.engine_id).background_ms(self.config.delta_max)
        return total

    def _build_candidates(self, *, exclude_model: Optional[str], exclude_session: Optional[str]) -> List[Candidate]:
        out: List[Candidate] = []
        for entry, lag in self.catalog.lagging(exclude_model=None):
            key = entry.key
            if exclude_model is not None and exclude_session is not None:
                if key.model_id == exclude_model and key.session_id == exclude_session:
                    continue
            if entry.inflight_target is not None:
                continue
            engine_id = self._engine_for.get((key.model_id, key.runtime_config))
            if engine_id is None or engine_id not in self.rates:
                continue
            rates = self.rates.get(engine_id)
            frontier = entry.frontier if entry.placement is Placement.RESIDENT else 0
            delta_max = lag if self.config.ablate_bounded_intervals else min(self.config.delta_max, lag)
            p = self.predictor.probability(key.session_id, key.model_id)
            benefit = expected_benefit_ms(rates, p, lag, delta_max)
            if self.config.ablate_tau_bg_weighting:
                cost = float(delta_max)  # raw tokens, not commensurable across tiers
            else:
                cost = resource_cost_ms(rates, delta_max, self.config.lambda_mem_ms_per_gib)
            out.append(
                Candidate(
                    key=key,
                    tenant_id=entry.tenant_id,
                    engine_id=engine_id,
                    frontier=frontier,
                    lag=lag,
                    delta_max=delta_max,
                    p_future=p,
                    benefit_ms=benefit,
                    cost_ms=cost,
                    score=score(benefit, cost),
                )
            )
        return out

    def _issue(self, c: Candidate, epoch: int) -> AtomicTicket:
        hist = self.catalog.history(c.key.session_id)
        entry = self.catalog.get(c.key)
        snapshot = hist.snapshot(c.key.model_id)
        ticket = AtomicTicket(
            ticket_id=new_ticket_id(),
            tenant_id=c.tenant_id,
            session_id=c.key.session_id,
            model_id=c.key.model_id,
            runtime_config=c.key.runtime_config,
            engine_id=c.engine_id,
            revision=hist.revision,
            generation=entry.generation,
            frontier=c.frontier,
            delta_max=c.delta_max,
            snapshot=snapshot,
            score=c.score,
            expected_benefit_ms=c.benefit_ms,
            predicted_cost_ms=c.cost_ms,
            p_future=c.p_future,
            epoch=epoch,
        )
        if not self.catalog.reserve(c.key, ticket.ticket_id, c.frontier + c.delta_max):
            self._ledger.settle(c.tenant_id, c.cost_ms)
            return ticket
        self._inflight[ticket.ticket_id] = ticket
        self._inflight_by_engine[c.engine_id] = ticket.ticket_id
        self.stats.issued += 1
        if self._sink is not None:
            self._sink(ticket)
        return ticket

    def _retire(self, ticket_id: str, status: TicketStatus, actual_ms: Optional[float] = None) -> None:
        ticket = self._inflight.pop(ticket_id, None)
        if ticket is None:
            return
        if self._inflight_by_engine.get(ticket.engine_id) == ticket_id:
            del self._inflight_by_engine[ticket.engine_id]
        if status is not TicketStatus.COMMITTED:
            self.catalog.release(ticket.key, ticket_id)
        self._ledger.settle(ticket.tenant_id, ticket.predicted_cost_ms, actual_ms)

    # ------------------------------------------------------------ introspect
    def inflight(self) -> Tuple[AtomicTicket, ...]:
        with self._lock:
            return tuple(self._inflight.values())

    def expire_leases(self, now: float) -> int:
        """Retire tickets whose engine never reported a terminal state."""
        n = 0
        with self._lock:
            for tid, t in list(self._inflight.items()):
                if now - t.issued_at > self.config.ticket_lease_s:
                    self._retire(tid, TicketStatus.EXPIRED)
                    self.stats.expired += 1
                    n += 1
        return n


@dataclass
class SchedulerStats:
    epochs: int = 0
    issued: int = 0
    admitted: int = 0
    committed: int = 0
    committed_tokens: int = 0
    refused: int = 0
    cancelled: int = 0
    stale: int = 0
    expired: int = 0
    unknown_results: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dict(self.__dict__)


__all__ = ["EngineSpec", "SchedulerConfig", "GlobalFrontierScheduler", "SchedulerStats"]
