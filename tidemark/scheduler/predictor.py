"""Future-destination probability.

A candidate is useful only if the destination is likely to serve the session
again. Tidemark assigns each candidate a probability that blends an
application-supplied router signal with a session-history transition estimate:

    p_{s,m} = alpha * p_router_{s,m} + (1 - alpha) * p_hist_{s,m}

``alpha = 1`` recovers a pure router signal, ``alpha = 0`` a router-agnostic
transition prior. In our deployments the router signal is the one that
matters; the prior exists so a deployment with no router probabilities still
has something to rank with, and the paper's sensitivity study shows where it
stops helping.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Sequence


class RouterSignal(Protocol):
    """Anything that can tell us, per session, where the next turn may go."""

    def destination_probabilities(self, session_id: str) -> Optional[Mapping[str, float]]:
        """Return ``{model_id: probability}`` for the next turn, or ``None``
        if the router has no opinion for this session."""


class StaticRouterSignal:
    """A router signal backed by a dict. Useful for replay and tests."""

    def __init__(self, table: Optional[Mapping[str, Mapping[str, float]]] = None) -> None:
        self._table: Dict[str, Dict[str, float]] = {
            s: dict(p) for s, p in (table or {}).items()
        }

    def set(self, session_id: str, probabilities: Mapping[str, float]) -> None:
        self._table[session_id] = dict(probabilities)

    def destination_probabilities(self, session_id: str) -> Optional[Mapping[str, float]]:
        return self._table.get(session_id)


class HistoryTransitionEstimator:
    """First-order model-to-model transition counts, per session.

    Estimated from the session's own history with additive smoothing. Early in
    a session it has seen only a handful of transitions, and the paper is
    explicit that this prior alone is a poor guide; it is here as the
    ``alpha = 0`` end of the blend.
    """

    def __init__(self, models: Sequence[str], smoothing: float = 0.5) -> None:
        if smoothing <= 0:
            raise ValueError("smoothing must be positive")
        self.models = tuple(models)
        self.smoothing = float(smoothing)
        self._counts: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        self._last: Dict[str, str] = {}

    def observe(self, session_id: str, served_model: str) -> None:
        prev = self._last.get(session_id)
        if prev is not None:
            self._counts[session_id][prev][served_model] += 1.0
        self._last[session_id] = served_model

    def forget(self, session_id: str) -> None:
        self._counts.pop(session_id, None)
        self._last.pop(session_id, None)

    def probabilities(self, session_id: str) -> Dict[str, float]:
        current = self._last.get(session_id)
        row = self._counts[session_id][current] if current is not None else {}
        total = sum(row.get(m, 0.0) for m in self.models) + self.smoothing * len(self.models)
        return {m: (row.get(m, 0.0) + self.smoothing) / total for m in self.models}


@dataclass
class DestinationPredictor:
    """Blend of router signal and history prior (Equation 3 in the paper)."""

    models: Sequence[str]
    router: Optional[RouterSignal] = None
    alpha: float = 1.0
    smoothing: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        self.models = tuple(self.models)
        self._hist = HistoryTransitionEstimator(self.models, self.smoothing)

    def observe_turn(self, session_id: str, served_model: str) -> None:
        self._hist.observe(session_id, served_model)

    def forget(self, session_id: str) -> None:
        self._hist.forget(session_id)

    def probabilities(self, session_id: str) -> Dict[str, float]:
        p_hist = self._hist.probabilities(session_id)
        p_router = self.router.destination_probabilities(session_id) if self.router else None
        if p_router is None:
            # No router opinion: fall back to the prior regardless of alpha.
            return p_hist
        out: Dict[str, float] = {}
        for m in self.models:
            out[m] = self.alpha * float(p_router.get(m, 0.0)) + (1.0 - self.alpha) * p_hist[m]
        z = sum(out.values())
        if z > 0:
            out = {m: v / z for m, v in out.items()}
        return out

    def probability(self, session_id: str, model_id: str) -> float:
        return self.probabilities(session_id).get(model_id, 0.0)


__all__ = [
    "RouterSignal",
    "StaticRouterSignal",
    "HistoryTransitionEstimator",
    "DestinationPredictor",
]
