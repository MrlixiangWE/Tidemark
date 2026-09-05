"""Router interface.

Tidemark does not route. The application router remains the only component
that decides which model serves a turn; Tidemark takes that choice as input,
marks every other model lagging, and optionally consumes the router's
probabilities over future destinations. This module defines the small contract
a router implements to plug in.
"""

from tidemark.router.interface import DifficultyRouter, RouteDecision, Router

__all__ = ["Router", "RouteDecision", "DifficultyRouter"]
