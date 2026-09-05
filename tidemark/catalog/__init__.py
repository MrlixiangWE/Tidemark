"""Versioned KV frontier catalog.

The catalog is the abstraction that makes the progress of non-serving models
visible to a scheduler. For every active session it stores the current text
revision, and for every compatible ``(model, runtime_config)`` pair it stores

    F : (s, m, c) -> <r, F, L, q, g>

where ``r`` is the revision that produced the state, ``F`` the committed token
position in the model's own tokenisation of the history, ``L`` the physical
engine placement, ``q`` an optional in-flight target, and ``g`` the current
generation, the per-model projection of the session revision.

The entry separates logical progress from physical allocation. The engine
keeps ownership of block hashing, placement and reclamation; the frontier only
advances after the engine confirms a physical commit that still passes the
validity predicate in :mod:`tidemark.catalog.validity`.
"""

from tidemark.catalog.frontier import (
    FrontierEntry,
    FrontierKey,
    Placement,
    VersionedFrontierCatalog,
)
from tidemark.catalog.history import SessionHistory, Tokenizer, TokenizerRegistry, content_hash
from tidemark.catalog.validity import CommitVerdict, validate_completion

__all__ = [
    "SessionHistory",
    "Tokenizer",
    "TokenizerRegistry",
    "content_hash",
    "FrontierEntry",
    "FrontierKey",
    "Placement",
    "VersionedFrontierCatalog",
    "CommitVerdict",
    "validate_completion",
]
