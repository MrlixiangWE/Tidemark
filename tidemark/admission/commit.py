"""Commit and reuse.

A background task runs the destination model's ordinary prefill path but
bypasses sampling and response construction. When the interval finishes, the
engine hands the result to :class:`CommitPath`, which computes the content hash
over the tokens it actually prefilled and builds the :class:`TicketResult` the
scheduler will validate against the catalog. A valid interval becomes visible
through the engine's ordinary prefix-cache lookup, so future foreground
requests need no special reuse path.
"""

from __future__ import annotations

from typing import Callable

from tidemark.catalog.history import content_hash
from tidemark.scheduler.ticket import AtomicTicket, TicketResult, TicketStatus

ResultSink = Callable[[TicketResult], None]


class CommitPath:
    def __init__(self, engine_id: str, sink: ResultSink) -> None:
        self.engine_id = engine_id
        self._sink = sink

    def admitted(self, ticket: AtomicTicket, delta: int) -> None:
        self._sink(TicketResult(ticket.ticket_id, self.engine_id, TicketStatus.ADMITTED, admitted_delta=delta))

    def refused(self, ticket: AtomicTicket, reason: str) -> None:
        self._sink(TicketResult(ticket.ticket_id, self.engine_id, TicketStatus.REFUSED, reason=reason))

    def cancelled(self, ticket: AtomicTicket, *, prefilled_tokens: int, gpu_ms: float) -> None:
        self._sink(
            TicketResult(
                ticket.ticket_id,
                self.engine_id,
                TicketStatus.CANCELLED,
                prefilled_tokens=prefilled_tokens,
                gpu_ms=gpu_ms,
                reason="foreground_arrival",
            )
        )

    def completed(
        self,
        ticket: AtomicTicket,
        *,
        admitted_delta: int,
        cached_tokens: int,
        physically_cached_after: int,
        gpu_ms: float,
        kv_bytes: int = 0,
    ) -> TicketResult:
        """The prefill finished. Report a commit candidate.

        ``physically_cached_after`` is the prefix the engine reports resident
        once the request completes. If it is shorter than ``F + delta`` the
        engine truncated or partially evicted the interval and we report the
        shortfall, which the validity predicate turns into a rejection.
        """
        end = ticket.frontier + admitted_delta
        effective = min(admitted_delta, max(0, physically_cached_after - ticket.frontier))
        result = TicketResult(
            ticket.ticket_id,
            self.engine_id,
            TicketStatus.COMMITTED,
            admitted_delta=effective,
            cached_tokens=cached_tokens,
            prefilled_tokens=max(0, end - cached_tokens),
            snapshot_hash=content_hash(ticket.snapshot.token_ids[: ticket.frontier + effective]),
            gpu_ms=gpu_ms,
            kv_bytes=kv_bytes,
            reason="prefill_complete" if effective == admitted_delta else "partial_residency",
        )
        self._sink(result)
        return result


__all__ = ["CommitPath", "ResultSink"]
