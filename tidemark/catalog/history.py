"""Session histories, revisions, and per-model tokenised views.

A conversation grows in two ways. Appending a turn extends the history without
disturbing any earlier token, so a prefix computed before the append remains a
prefix after it. Editing, regenerating or branching rewrites tokens at some
position, and every prefix spanning that position becomes stale. We call the
first kind of change an *append* and the second a *revision transition*; only
the latter bumps ``revision``.

Because the tiers of a deployment usually run models with different
tokenizers, the same text maps to a different offset on every model. The
history therefore keeps one tokenised view per model, H_s^{(m)}, and every
frontier position in the catalog is expressed in that model's token space.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Tokenizer = Callable[[str], Sequence[int]]


def content_hash(token_ids: Sequence[int]) -> str:
    """Content hash over a token range.

    Engines already maintain a hash of this kind for block-level prefix lookup;
    we use the same idea at ticket granularity so the validity check can tell
    whether the tokens a ticket prefilled are still the tokens of the history.
    """
    h = hashlib.blake2b(digest_size=16)
    for tok in token_ids:
        h.update(int(tok).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class TokenizerRegistry:
    """Maps a model id to the tokenizer that produces its view of the history."""

    def __init__(self) -> None:
        self._tokenizers: Dict[str, Tokenizer] = {}

    def register(self, model_id: str, tokenizer: Tokenizer) -> None:
        self._tokenizers[model_id] = tokenizer

    def get(self, model_id: str) -> Tokenizer:
        try:
            return self._tokenizers[model_id]
        except KeyError:
            raise KeyError(f"no tokenizer registered for model {model_id!r}") from None

    def models(self) -> Tuple[str, ...]:
        return tuple(self._tokenizers)

    def __contains__(self, model_id: object) -> bool:
        return model_id in self._tokenizers


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class _View:
    """Tokenised history under one model's tokenizer."""

    token_ids: List[int] = field(default_factory=list)
    turn_offsets: List[int] = field(default_factory=list)  # token offset where turn i starts


class SessionHistory:
    """The authoritative text of one conversation plus its tokenised views.

    Thread-safe. ``append`` never changes ``revision``; ``rewrite`` and
    ``truncate`` do, and return the longest common prefix of the old and new
    views for every model so the catalog can retract frontiers accordingly.
    """

    def __init__(self, session_id: str, tenant_id: str, tokenizers: TokenizerRegistry) -> None:
        self.session_id = session_id
        self.tenant_id = tenant_id
        self._tokenizers = tokenizers
        self._turns: List[Turn] = []
        self._views: Dict[str, _View] = {}
        self._revision = 0
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ props
    @property
    def revision(self) -> int:
        return self._revision

    @property
    def turns(self) -> Tuple[Turn, ...]:
        with self._lock:
            return tuple(self._turns)

    def models(self) -> Tuple[str, ...]:
        return self._tokenizers.models()

    # ------------------------------------------------------------------ views
    def _view(self, model_id: str) -> _View:
        view = self._views.get(model_id)
        if view is None:
            view = _View()
            tok = self._tokenizers.get(model_id)
            for turn in self._turns:
                view.turn_offsets.append(len(view.token_ids))
                view.token_ids.extend(tok(self._render(turn)))
            self._views[model_id] = view
        return view

    @staticmethod
    def _render(turn: Turn) -> str:
        # A minimal chat template. Real deployments plug the engine's own
        # template in through the tokenizer callable; what matters here is
        # that every model sees the same text and tokenises it its own way.
        return f"<|{turn.role}|>\n{turn.text}\n"

    def token_ids(self, model_id: str) -> Tuple[int, ...]:
        with self._lock:
            return tuple(self._view(model_id).token_ids)

    def length(self, model_id: str) -> int:
        with self._lock:
            return len(self._view(model_id).token_ids)

    def prefix_hash(self, model_id: str, end: int) -> str:
        with self._lock:
            ids = self._view(model_id).token_ids
            if end > len(ids):
                raise ValueError(f"prefix end {end} exceeds history length {len(ids)}")
            return content_hash(ids[:end])

    # -------------------------------------------------------------- mutation
    def append(self, role: str, text: str) -> int:
        """Append a turn. Returns the turn index. Does not bump the revision."""
        with self._lock:
            turn = Turn(role, text)
            self._turns.append(turn)
            for model_id, view in self._views.items():
                tok = self._tokenizers.get(model_id)
                view.turn_offsets.append(len(view.token_ids))
                view.token_ids.extend(tok(self._render(turn)))
            return len(self._turns) - 1

    def rewrite(self, turn_index: int, text: str, role: Optional[str] = None) -> Dict[str, int]:
        """Rewrite one turn (edit / regenerate) and drop everything after it.

        Returns ``{model_id: lcp}`` where ``lcp`` is the longest common prefix
        between the old and new tokenised views for that model.
        """
        with self._lock:
            if not 0 <= turn_index < len(self._turns):
                raise IndexError(turn_index)
            old_views = {m: list(v.token_ids) for m, v in self._views.items()}
            keep = self._turns[:turn_index]
            new_turn = Turn(role or self._turns[turn_index].role, text)
            self._turns = keep + [new_turn]
            self._views.clear()
            self._revision += 1
            lcp: Dict[str, int] = {}
            for model_id, old in old_views.items():
                new = self._view(model_id).token_ids
                lcp[model_id] = longest_common_prefix(old, new)
            return lcp

    def truncate(self, turn_count: int) -> Dict[str, int]:
        """Branch: keep the first ``turn_count`` turns. Bumps the revision."""
        with self._lock:
            if turn_count > len(self._turns):
                raise ValueError("cannot truncate beyond the current history")
            old_views = {m: list(v.token_ids) for m, v in self._views.items()}
            self._turns = self._turns[:turn_count]
            self._views.clear()
            self._revision += 1
            return {
                m: longest_common_prefix(old, self._view(m).token_ids)
                for m, old in old_views.items()
            }

    # -------------------------------------------------------------- snapshot
    def snapshot(self, model_id: str) -> HistorySnapshot:
        with self._lock:
            ids = tuple(self._view(model_id).token_ids)
            return HistorySnapshot(
                session_id=self.session_id,
                model_id=model_id,
                revision=self._revision,
                token_ids=ids,
            )


@dataclass(frozen=True)
class HistorySnapshot:
    """An immutable copy of H_s^{(m)} at a given revision.

    Tickets carry a snapshot so that the engine can prefill exactly the tokens
    the scheduler ranked, and so the commit path can compare the hash of what
    was prefilled against the live history.
    """

    session_id: str
    model_id: str
    revision: int
    token_ids: Tuple[int, ...]

    def __len__(self) -> int:
        return len(self.token_ids)

    def prefix_hash(self, end: int) -> str:
        return content_hash(self.token_ids[:end])

    def slice(self, start: int, end: int) -> Tuple[int, ...]:
        return self.token_ids[start:end]


__all__ = [
    "Tokenizer",
    "TokenizerRegistry",
    "SessionHistory",
    "HistorySnapshot",
    "Turn",
    "content_hash",
    "longest_common_prefix",
]
