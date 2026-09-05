"""Atomic tickets.

The unit exchanged between the scheduler and an engine is

    a = <u, s, m, c, e, r, g, [F, F + delta_max)>

where ``u`` is the tenant, ``s`` the session, ``m`` the model, ``c`` the
runtime configuration, ``e`` the destination engine, ``r`` the session
revision, ``g`` a generation number, and ``[F, F + delta_max)`` the largest
frontier interval the ticket may advance. The engine chooses the admitted size
below ``delta_max`` from a small fixed set at admission time.

Atomicity is what lets short windows produce durable progress: a ticket that
is rejected, cancelled or stale leaves the frontier unchanged, and a ticket
that commits does so in one step, after which the scheduler re-ranks.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from tidemark.catalog.frontier import FrontierKey
from tidemark.catalog.history import HistorySnapshot, content_hash


def new_ticket_id() -> str:
    return uuid.uuid4().hex[:16]


class TicketStatus(str, enum.Enum):
    ISSUED = "issued"          # scheduler selected it, engine has not seen it
    ADMITTED = "admitted"      # engine fitted it into a safe budget
    COMMITTED = "committed"    # validated and applied to the catalog
    REFUSED = "refused"        # engine never admitted it (blocked / no budget)
    CANCELLED = "cancelled"    # stopped by a foreground arrival mid-interval
    STALE = "stale"            # completed but failed the validity predicate
    EXPIRED = "expired"        # lease ran out without a terminal report


@dataclass(frozen=True)
class AtomicTicket:
    ticket_id: str
    tenant_id: str
    session_id: str
    model_id: str
    runtime_config: str
    engine_id: str
    revision: int
    generation: int
    frontier: int
    delta_max: int
    snapshot: HistorySnapshot
    score: float
    expected_benefit_ms: float
    predicted_cost_ms: float
    p_future: float
    epoch: int
    issued_at: float = field(default_factory=time.monotonic)

    # -------------------------------------------------------------- helpers
    @property
    def key(self) -> FrontierKey:
        return FrontierKey(self.session_id, self.model_id, self.runtime_config)

    @property
    def interval(self) -> Tuple[int, int]:
        return (self.frontier, self.frontier + self.delta_max)

    def prompt_token_ids(self, delta: Optional[int] = None) -> Tuple[int, ...]:
        """Token ids the engine should prefill for an admitted size ``delta``.

        This is the whole prefix ``H_a[0:F + delta]``, not just the interval:
        the engine reuses ``[0, F)`` from its prefix cache and computes only
        the missing tail, but the request must carry the full prefix so the
        block hashes line up with what a later foreground request will send.
        """
        end = self.frontier + (self.delta_max if delta is None else min(delta, self.delta_max))
        return self.snapshot.token_ids[:end]

    def prefix_hash(self, delta: int) -> str:
        return content_hash(self.prompt_token_ids(delta))

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any], prompt_token_ids: Sequence[int]) -> AtomicTicket:
        """Rebuild a ticket inside an engine process from request metadata.

        The engine never sees the scheduler's objects, only the request; this
        is the inverse of :meth:`metadata` plus the prompt the request carried.
        """
        snapshot = HistorySnapshot(
            session_id=str(meta["session_id"]),
            model_id=str(meta["model_id"]),
            revision=int(meta["revision"]),
            token_ids=tuple(int(t) for t in prompt_token_ids),
        )
        return cls(
            ticket_id=str(meta["ticket_id"]),
            tenant_id=str(meta["tenant_id"]),
            session_id=str(meta["session_id"]),
            model_id=str(meta["model_id"]),
            runtime_config=str(meta["runtime_config"]),
            engine_id=str(meta.get("engine_id", "")),
            revision=int(meta["revision"]),
            generation=int(meta["generation"]),
            frontier=int(meta["frontier"]),
            delta_max=int(meta["delta_max"]),
            snapshot=snapshot,
            score=float(meta.get("score", 0.0)),
            expected_benefit_ms=float(meta.get("expected_benefit_ms", 0.0)),
            predicted_cost_ms=float(meta.get("predicted_cost_ms", 0.0)),
            p_future=float(meta.get("p_future", 0.0)),
            epoch=int(meta.get("epoch", 0)),
        )

    def metadata(self) -> Dict[str, Any]:
        """Fields attached to the prefill-only request so the engine shim can
        recognise, size, and report on the ticket."""
        return {
            "tidemark_ticket": True,
            "ticket_id": self.ticket_id,
            "engine_id": self.engine_id,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "model_id": self.model_id,
            "runtime_config": self.runtime_config,
            "revision": self.revision,
            "generation": self.generation,
            "frontier": self.frontier,
            "delta_max": self.delta_max,
            "expected_benefit_ms": self.expected_benefit_ms,
            "predicted_cost_ms": self.predicted_cost_ms,
            "p_future": self.p_future,
            "score": self.score,
            "epoch": self.epoch,
        }


@dataclass(frozen=True)
class TicketResult:
    """What an engine reports back for a ticket, terminal or not."""

    ticket_id: str
    engine_id: str
    status: TicketStatus
    admitted_delta: int = 0
    cached_tokens: int = 0          # prefix the engine found already resident
    prefilled_tokens: int = 0       # tokens it actually computed
    snapshot_hash: str = ""         # h(H_a[0:F + admitted_delta]) over what it prefilled
    gpu_ms: float = 0.0             # measured background compute time
    kv_bytes: int = 0
    reason: str = ""
    reported_at: float = field(default_factory=time.monotonic)

    @property
    def terminal(self) -> bool:
        return self.status is not TicketStatus.ADMITTED


__all__ = ["AtomicTicket", "TicketResult", "TicketStatus", "new_ticket_id"]
