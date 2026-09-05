"""Global frontier scheduling.

With frontiers visible in the catalog, the scheduler decides, across tenants
and tiers, which frontier to advance with the capacity foreground decode
leaves idle on each engine. It issues that work in bounded intervals that fit
the short safe budgets engines actually have, and it re-ranks after every
committed interval.

A ranking epoch starts whenever a session history grows or an engine reports a
state transition. The epoch enumerates lagging frontiers, scores each one by
the switch latency it would remove per unit of background compute time
(:mod:`tidemark.scheduler.ranking`), applies per-tenant caps and aging, and
issues at most one :class:`~tidemark.scheduler.ticket.AtomicTicket` per engine.
"""

from tidemark.scheduler.cost_model import EngineRates, RateTable, fit_rate
from tidemark.scheduler.global_scheduler import GlobalFrontierScheduler, SchedulerConfig
from tidemark.scheduler.predictor import (
    DestinationPredictor,
    HistoryTransitionEstimator,
    RouterSignal,
    StaticRouterSignal,
)
from tidemark.scheduler.ranking import Candidate, RankingEpoch, TenantCaps, TenantLedger, score
from tidemark.scheduler.ticket import AtomicTicket, TicketResult, TicketStatus, new_ticket_id

__all__ = [
    "AtomicTicket",
    "TicketResult",
    "TicketStatus",
    "new_ticket_id",
    "DestinationPredictor",
    "HistoryTransitionEstimator",
    "RouterSignal",
    "StaticRouterSignal",
    "EngineRates",
    "RateTable",
    "fit_rate",
    "Candidate",
    "RankingEpoch",
    "TenantCaps",
    "TenantLedger",
    "score",
    "GlobalFrontierScheduler",
    "SchedulerConfig",
]
