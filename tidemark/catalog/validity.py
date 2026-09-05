"""Version safety as a validity predicate.

A completed ticket ``a = <u, s, m, c, e, r_a, g_a, [F_a, F_a + delta)>`` is
accepted only if

    Valid(a) = 1[ h(H_a[0:F_a+delta]) == h(H_s[0:F_a+delta]) ]
             * 1[ g_a == g_{s,m,c} ]
             * 1[ F_a == F_{s,m,c} ]

The first term accepts a ticket whose tokens the current history still
reproduces, the second matches the catalog's generation for the key, and the
third requires the interval to continue the committed frontier exactly, with
no gaps and no overlaps. Validity tests the identity of the interval a ticket
produced, not the equality of two whole-history versions: an append after the
ticket was issued does not invalidate it, an edit inside the interval does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CommitVerdict:
    ok: bool
    reason: str
    delta: int = 0

    # Reason codes, kept short because they end up in ticket logs.
    ACCEPTED = "accepted"
    TRUNCATED = "truncated_by_foreground"
    STALE_GENERATION = "stale_generation"
    FRONTIER_MOVED = "frontier_moved"
    HASH_MISMATCH = "hash_mismatch"
    BEYOND_HISTORY = "beyond_history"
    EMPTY = "empty_interval"


def validate_completion(
    *,
    live_revision: int,
    live_generation: int,
    live_frontier: int,
    live_hash_fn: Callable[[int], str],
    live_length: int,
    ticket_frontier: int,
    ticket_generation: int,
    ticket_revision: int,
    admitted_delta: int,
    requested_delta: int,
    snapshot_hash: str,
) -> CommitVerdict:
    """Evaluate ``Valid(a)`` for one completed interval.

    ``snapshot_hash`` is ``h(H_a[0:F_a + admitted_delta])`` computed by the
    engine over the tokens it actually prefilled. ``live_hash_fn(end)`` returns
    the same hash over the live history.
    """
    if admitted_delta <= 0:
        return CommitVerdict(False, CommitVerdict.EMPTY)
    if admitted_delta < requested_delta and requested_delta > 0:
        # Foreground priority: an interval truncated by a foreground arrival
        # does not commit. The scheduler will re-rank and reissue.
        return CommitVerdict(False, CommitVerdict.TRUNCATED)
    if ticket_generation != live_generation:
        return CommitVerdict(False, CommitVerdict.STALE_GENERATION)
    if ticket_frontier != live_frontier:
        return CommitVerdict(False, CommitVerdict.FRONTIER_MOVED)
    end = ticket_frontier + admitted_delta
    if end > live_length:
        return CommitVerdict(False, CommitVerdict.BEYOND_HISTORY)
    if live_hash_fn(end) != snapshot_hash:
        return CommitVerdict(False, CommitVerdict.HASH_MISMATCH)
    # ``ticket_revision`` may lag ``live_revision`` if the session was appended
    # to (not rewritten) since issue; the hash check above already covers that
    # the prefix is unchanged, so revision is informational here.
    del ticket_revision, live_revision
    return CommitVerdict(True, CommitVerdict.ACCEPTED, delta=admitted_delta)


__all__ = ["CommitVerdict", "validate_completion"]
