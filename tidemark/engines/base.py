from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Callable, Optional

from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import StepState
from tidemark.scheduler.ticket import AtomicTicket


@dataclass(frozen=True)
class EngineDescriptor:
    engine_id: str
    model_id: str
    runtime_config: str
    tier: str            # "device" | "edge" | "cloud"
    endpoint: str        # base URL of the engine's HTTP API
    block_size: int = 16


@dataclass(frozen=True)
class ResidentPrefix:
    session_id: str
    tokens: int
    kv_bytes: int = 0


class EngineAdapter(abc.ABC):
    """Interface every engine integration implements."""

    def __init__(self, descriptor: EngineDescriptor, commit: CommitPath) -> None:
        self.descriptor = descriptor
        self.commit = commit

    @property
    def engine_id(self) -> str:
        return self.descriptor.engine_id

    @abc.abstractmethod
    def step_state(self) -> StepState:
        """Current scheduler-iteration state, for the local admission controller."""

    @abc.abstractmethod
    def submit(self, ticket: AtomicTicket, delta: int) -> None:
        """Run a prefill-only request for ``[F, F + delta)``; report through ``commit``."""

    @abc.abstractmethod
    def cancel(self, ticket_id: str) -> None:
        """Stop an in-flight interval at the next scheduler boundary."""

    @abc.abstractmethod
    def resident_prefix(self, session_id: str, token_ids: tuple) -> ResidentPrefix:
        """Longest prefix of ``token_ids`` the engine still has resident."""

    def on_eviction(self, callback: Callable[[str, int], None]) -> None:  # pragma: no cover - optional
        """Register ``callback(session_id, resident_tokens)`` for eviction reports."""
        self._eviction_cb: Optional[Callable[[str, int], None]] = callback


__all__ = ["EngineAdapter", "EngineDescriptor", "ResidentPrefix"]
