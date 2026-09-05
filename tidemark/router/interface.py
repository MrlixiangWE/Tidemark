from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Protocol


@dataclass(frozen=True)
class RouteDecision:
    session_id: str
    turn_index: int
    model_id: str
    runtime_config: str
    # Optional: the router's belief about where the *next* turn will go.
    next_destination: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


class Router(Protocol):
    def route(self, session_id: str, turn_index: int, prompt: str, context: Mapping[str, object]) -> RouteDecision: ...


class DifficultyRouter:
    """A reference router: difficulty tier -> tier, with constraint fallback.

    ``constraints`` is a callable returning the set of tiers currently feasible
    for the session (link up, battery above floor, cost cap not hit). When the
    difficulty-selected tier is infeasible the route falls back to the highest
    feasible one and returns when the constraint clears, which is the
    "mobility replay" behaviour described in the paper.
    """

    ORDER = ("device", "edge", "cloud")

    def __init__(
        self,
        tier_models: Mapping[str, str],
        runtime_config: str,
        difficulty_fn,
        constraints_fn=None,
        stickiness: float = 0.6,
    ) -> None:
        self.tier_models = dict(tier_models)
        self.runtime_config = runtime_config
        self.difficulty_fn = difficulty_fn
        self.constraints_fn = constraints_fn or (lambda session_id: set(self.ORDER))
        self.stickiness = stickiness
        self._last: Dict[str, str] = {}

    def route(self, session_id: str, turn_index: int, prompt: str, context: Mapping[str, object]) -> RouteDecision:
        wanted = self.difficulty_fn(prompt, context)  # "device" | "edge" | "cloud"
        feasible = set(self.constraints_fn(session_id)) & set(self.tier_models)
        if not feasible:
            raise RuntimeError(f"no feasible tier for session {session_id}")
        tier = wanted if wanted in feasible else max(feasible, key=self.ORDER.index)
        model = self.tier_models[tier]
        self._last[session_id] = tier
        return RouteDecision(
            session_id=session_id,
            turn_index=turn_index,
            model_id=model,
            runtime_config=self.runtime_config,
            next_destination=self._next_belief(tier, feasible),
            reason="task" if tier == wanted else "constraint_fallback",
        )

    def _next_belief(self, tier: str, feasible: set) -> Dict[str, float]:
        others = [t for t in self.tier_models if t != tier]
        rest = (1.0 - self.stickiness) / max(1, len(others))
        return {self.tier_models[t]: (self.stickiness if t == tier else rest) for t in self.tier_models}

    # RouterSignal protocol -------------------------------------------------
    def destination_probabilities(self, session_id: str) -> Optional[Mapping[str, float]]:
        tier = self._last.get(session_id)
        if tier is None:
            return None
        return self._next_belief(tier, set(self.tier_models))


__all__ = ["Router", "RouteDecision", "DifficultyRouter"]
