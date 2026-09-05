"""The versioned KV frontier catalog.

Four invariants are enforced over every entry:

Compatibility
    A frontier can be consumed only by the model and runtime configuration
    that created it. Keys are ``(session, model, runtime_config)``; nothing
    ever reads across keys.
Prefix validity
    A committed interval matches the current model-tokenised session prefix.
    The commit path checks this with :func:`tidemark.catalog.validity.validate_completion`.
Monotonicity
    Absent eviction and revision rewrites, ``F`` only advances.
Foreground priority
    Background work cannot displace a foreground request. That invariant is
    enforced by the engine-local admission path; the catalog's part is to
    refuse an interval truncated by a foreground arrival, which shows up here
    as a completion whose admitted size is smaller than the ticket asked for.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from tidemark.catalog.history import SessionHistory
from tidemark.catalog.validity import CommitVerdict, validate_completion


class Placement(str, enum.Enum):
    """Where the physical blocks behind a frontier live."""

    NONE = "none"        # nothing resident for this key yet
    RESIDENT = "resident"  # blocks resident on the engine named by ``engine_id``
    EVICTED = "evicted"  # engine reclaimed the blocks; logical frontier retracted


@dataclass(frozen=True, order=True)
class FrontierKey:
    session_id: str
    model_id: str
    runtime_config: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.session_id}/{self.model_id}@{self.runtime_config}"


@dataclass
class FrontierEntry:
    """One catalog record, <r, F, L, q, g>, plus bookkeeping."""

    key: FrontierKey
    tenant_id: str
    revision: int = 0            # r: revision that produced the committed state
    frontier: int = 0            # F: committed token position in H_s^{(m)}
    placement: Placement = Placement.NONE  # L
    engine_id: Optional[str] = None
    inflight_target: Optional[int] = None  # q
    inflight_ticket: Optional[str] = None
    generation: int = 0          # g: per-model projection of the session revision
    kv_bytes: int = 0
    committed_at: Optional[float] = None
    last_read_at: Optional[float] = None
    committed_intervals: int = 0
    rejected_intervals: int = 0

    def snapshot(self) -> FrontierEntry:
        return replace(self)


class VersionedFrontierCatalog:
    """Thread-safe map from ``(session, model, runtime_config)`` to a frontier.

    The catalog is soft state. It can be rebuilt after a scheduler restart by
    replaying session logs and asking every engine for the longest prefix it
    still holds (:meth:`rebuild_from_engines`).
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[FrontierKey, FrontierEntry] = {}
        self._sessions: Dict[str, SessionHistory] = {}
        self._compatible: Dict[str, List[Tuple[str, str]]] = {}  # session -> [(model, cfg)]
        self._listeners: List[Callable[[str, FrontierEntry], None]] = []

    # --------------------------------------------------------------- sessions
    def register_session(
        self,
        history: SessionHistory,
        compatible: Iterable[Tuple[str, str]],
    ) -> None:
        """Start tracking a session for the given ``(model, runtime_config)`` pairs."""
        with self._lock:
            self._sessions[history.session_id] = history
            pairs = list(compatible)
            self._compatible[history.session_id] = pairs
            for model_id, cfg in pairs:
                key = FrontierKey(history.session_id, model_id, cfg)
                self._entries.setdefault(key, FrontierEntry(key=key, tenant_id=history.tenant_id))

    def forget_session(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            for model_id, cfg in self._compatible.pop(session_id, []):
                self._entries.pop(FrontierKey(session_id, model_id, cfg), None)

    def history(self, session_id: str) -> SessionHistory:
        with self._lock:
            return self._sessions[session_id]

    def sessions(self) -> Tuple[str, ...]:
        with self._lock:
            return tuple(self._sessions)

    # ---------------------------------------------------------------- lookups
    def get(self, key: FrontierKey) -> FrontierEntry:
        with self._lock:
            return self._entries[key].snapshot()

    def entries(self) -> Iterator[FrontierEntry]:
        with self._lock:
            return iter([e.snapshot() for e in self._entries.values()])

    def lagging(self, exclude_model: Optional[str] = None) -> List[Tuple[FrontierEntry, int]]:
        """Entries whose frontier lags the current history.

        Returns ``(entry, lag_tokens)`` pairs. ``exclude_model`` is the model
        that just served the turn; its frontier is advanced by the foreground
        request itself and is never a background candidate for that turn.
        """
        out: List[Tuple[FrontierEntry, int]] = []
        with self._lock:
            for key, entry in self._entries.items():
                if exclude_model is not None and key.model_id == exclude_model:
                    continue
                hist = self._sessions.get(key.session_id)
                if hist is None:
                    continue
                total = hist.length(key.model_id)
                frontier = entry.frontier if entry.placement is Placement.RESIDENT else 0
                lag = total - frontier
                if lag > 0:
                    out.append((entry.snapshot(), lag))
        return out

    def size_bytes(self) -> int:
        """Approximate in-memory footprint. ~312 B per entry on CPython 3.11."""
        with self._lock:
            return 312 * len(self._entries)

    # ------------------------------------------------------------ reservation
    def reserve(self, key: FrontierKey, ticket_id: str, target: int) -> bool:
        """Record an in-flight target ``q`` for a ticket. One per key."""
        with self._lock:
            entry = self._entries[key]
            if entry.inflight_target is not None:
                return False
            frontier = entry.frontier if entry.placement is Placement.RESIDENT else 0
            if target <= frontier:
                return False
            hist = self._sessions.get(key.session_id)
            if hist is not None and target > hist.length(key.model_id):
                return False
            entry.inflight_target = target
            entry.inflight_ticket = ticket_id
            return True

    def release(self, key: FrontierKey, ticket_id: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.inflight_ticket == ticket_id:
                entry.inflight_target = None
                entry.inflight_ticket = None

    # ---------------------------------------------------------------- commits
    def commit_background(
        self,
        key: FrontierKey,
        *,
        ticket_id: str,
        ticket_frontier: int,
        ticket_generation: int,
        ticket_revision: int,
        admitted_delta: int,
        requested_delta: int,
        snapshot_hash: str,
        engine_id: str,
        kv_bytes: int = 0,
    ) -> CommitVerdict:
        """Apply the validity predicate and advance ``F`` by ``admitted_delta`` if it holds."""
        with self._lock:
            entry = self._entries[key]
            hist = self._sessions[key.session_id]
            verdict = validate_completion(
                live_revision=hist.revision,
                live_generation=entry.generation,
                live_frontier=entry.frontier if entry.placement is Placement.RESIDENT else 0,
                live_hash_fn=lambda end: hist.prefix_hash(key.model_id, end),
                live_length=hist.length(key.model_id),
                ticket_frontier=ticket_frontier,
                ticket_generation=ticket_generation,
                ticket_revision=ticket_revision,
                admitted_delta=admitted_delta,
                requested_delta=requested_delta,
                snapshot_hash=snapshot_hash,
            )
            self.release(key, ticket_id)
            if not verdict.ok:
                entry.rejected_intervals += 1
                return verdict
            entry.frontier = ticket_frontier + admitted_delta
            entry.revision = ticket_revision
            entry.placement = Placement.RESIDENT
            entry.engine_id = engine_id
            entry.kv_bytes = max(entry.kv_bytes, kv_bytes)
            entry.committed_at = time.monotonic()
            entry.committed_intervals += 1
            self._notify("commit", entry)
            return verdict

    def commit_foreground(
        self,
        key: FrontierKey,
        *,
        resident_prefix: int,
        engine_id: str,
        kv_bytes: int = 0,
    ) -> None:
        """A foreground request on ``m`` advanced the same frontier.

        Bumping the generation supersedes any in-flight background target; its
        later completion then fails validation instead of overwriting the
        newer state. ``resident_prefix`` is the prompt prefix that remains
        resident after the request, as reported by the engine.
        """
        with self._lock:
            entry = self._entries[key]
            hist = self._sessions[key.session_id]
            entry.generation += 1
            entry.inflight_target = None
            entry.inflight_ticket = None
            entry.frontier = min(resident_prefix, hist.length(key.model_id))
            entry.revision = hist.revision
            entry.placement = Placement.RESIDENT if entry.frontier > 0 else Placement.NONE
            entry.engine_id = engine_id
            entry.kv_bytes = kv_bytes or entry.kv_bytes
            entry.committed_at = time.monotonic()
            entry.last_read_at = entry.committed_at
            self._notify("foreground", entry)

    # -------------------------------------------------------------- retraction
    def on_revision_transition(self, session_id: str, lcp_by_model: Dict[str, int]) -> None:
        """The history was edited, regenerated or branched.

        Every compatible key gets a new generation, which conservatively
        invalidates every in-flight ticket for the session, and each committed
        frontier retracts to the longest common prefix of the old and new
        tokenised histories. Blocks beyond the retracted frontier are left to
        the engine's ordinary replacement policy.
        """
        with self._lock:
            for model_id, cfg in self._compatible.get(session_id, []):
                entry = self._entries[FrontierKey(session_id, model_id, cfg)]
                entry.generation += 1
                entry.inflight_target = None
                entry.inflight_ticket = None
                lcp = lcp_by_model.get(model_id, 0)
                if entry.frontier > lcp:
                    entry.frontier = lcp
                if entry.frontier == 0:
                    entry.placement = Placement.NONE if entry.placement is Placement.NONE else Placement.EVICTED
                self._notify("revision", entry)

    def on_eviction(self, key: FrontierKey, resident_prefix: int) -> None:
        """The engine reclaimed blocks. Retract ``F`` to what is still resident."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.generation += 1
            entry.inflight_target = None
            entry.inflight_ticket = None
            if resident_prefix < entry.frontier:
                entry.frontier = resident_prefix
            if entry.frontier == 0:
                entry.placement = Placement.EVICTED
                entry.kv_bytes = 0
            self._notify("evict", entry)

    def on_engine_lost(self, engine_id: str) -> None:
        """Engine loss invalidates placement for every key it held."""
        with self._lock:
            for entry in self._entries.values():
                if entry.engine_id == engine_id:
                    entry.generation += 1
                    entry.inflight_target = None
                    entry.inflight_ticket = None
                    entry.frontier = 0
                    entry.placement = Placement.EVICTED
                    entry.kv_bytes = 0
                    entry.engine_id = None

    def mark_read(self, key: FrontierKey) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.last_read_at = time.monotonic()

    # --------------------------------------------------------------- recovery
    def rebuild_from_engines(self, resident: Dict[FrontierKey, Tuple[str, int]]) -> None:
        """Conservative reset after a scheduler restart.

        ``resident`` maps a key to ``(engine_id, longest_resident_prefix)`` as
        reported by each engine. Every frontier is set to no more than that.
        """
        with self._lock:
            for key, entry in self._entries.items():
                engine_id, prefix = resident.get(key, (None, 0))
                entry.generation += 1
                entry.inflight_target = None
                entry.inflight_ticket = None
                entry.frontier = min(entry.frontier, prefix)
                entry.engine_id = engine_id
                entry.placement = Placement.RESIDENT if entry.frontier > 0 else Placement.NONE

    # -------------------------------------------------------------- listeners
    def subscribe(self, fn: Callable[[str, FrontierEntry], None]) -> None:
        self._listeners.append(fn)

    def _notify(self, event: str, entry: FrontierEntry) -> None:
        snap = entry.snapshot()
        for fn in list(self._listeners):
            fn(event, snap)


__all__ = ["Placement", "FrontierKey", "FrontierEntry", "VersionedFrontierCatalog"]
