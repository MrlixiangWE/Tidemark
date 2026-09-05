from __future__ import annotations

import pytest

from tidemark.catalog.history import SessionHistory, TokenizerRegistry
from tidemark.scheduler.cost_model import EngineRates, RateTable


def _tok(scale: int):
    def tok(text: str):
        out = []
        for w in text.split():
            out.extend([hash(w) & 0xFFFF] * scale)
        return out

    return tok


@pytest.fixture
def tokenizers() -> TokenizerRegistry:
    reg = TokenizerRegistry()
    reg.register("device-1b", _tok(1))
    reg.register("edge-7b", _tok(2))
    reg.register("cloud-14b", _tok(3))
    return reg


@pytest.fixture
def history(tokenizers: TokenizerRegistry) -> SessionHistory:
    h = SessionHistory("s1", "tenant-a", tokenizers)
    h.append("user", "hello there how are you")
    h.append("assistant", "fine thanks and you")
    return h


@pytest.fixture
def rates() -> RateTable:
    return RateTable(
        [
            EngineRates("device-0", "device-1b", "device", 2.56, 2.33, 32768, tpot_ref_ms=48.0),
            EngineRates("edge-0", "edge-7b", "edge", 0.732, 0.344, 57344, tpot_ref_ms=31.0),
            EngineRates("cloud-0", "cloud-14b", "cloud", 0.191, 0.073, 98304, tpot_ref_ms=9.0),
        ]
    )
