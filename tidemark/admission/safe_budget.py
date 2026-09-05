"""Safe budget of a scheduler iteration.

Let ``B_t`` be the engine token budget of iteration ``t`` and let ``D_t`` and
``P_t`` be the foreground decode and prefill tokens already selected. Then

    X_t = max(0, min(B_t - D_t - P_t, X_max, X_t^KV))

where ``X_max`` caps a single admission and ``X_t^KV`` is the number of tokens
that fit below the configured KV headroom. It is recomputed on every iteration
rather than once per ticket; that is what lets bounded intervals occupy the
short windows a decode-heavy batch leaves without charging the foreground.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SafeBudgetInputs:
    token_budget: int          # B_t, e.g. vLLM's max_num_batched_tokens
    decode_tokens: int         # D_t
    prefill_tokens: int        # P_t
    kv_free_blocks: int
    kv_total_blocks: int
    block_size: int = 16
    kv_headroom: float = 0.08  # keep this fraction of KV blocks free
    x_max: int = 1024

    def __post_init__(self) -> None:
        if self.token_budget < 0 or self.decode_tokens < 0 or self.prefill_tokens < 0:
            raise ValueError("token counts must be non-negative")
        if self.kv_total_blocks <= 0 or not 0 <= self.kv_free_blocks <= self.kv_total_blocks:
            raise ValueError("invalid KV block counts")
        if not 0.0 <= self.kv_headroom < 1.0:
            raise ValueError("kv_headroom must lie in [0, 1)")
        if self.block_size <= 0 or self.x_max <= 0:
            raise ValueError("block_size and x_max must be positive")

    @property
    def kv_headroom_tokens(self) -> int:
        reserve_blocks = int(self.kv_total_blocks * self.kv_headroom + 0.999999)
        usable = self.kv_free_blocks - reserve_blocks
        return max(0, usable) * self.block_size

    @property
    def kv_utilisation(self) -> float:
        return 1.0 - self.kv_free_blocks / self.kv_total_blocks


def safe_budget(inputs: SafeBudgetInputs) -> int:
    """``X_t`` for one iteration."""
    residual = inputs.token_budget - inputs.decode_tokens - inputs.prefill_tokens
    return max(0, min(residual, inputs.x_max, inputs.kv_headroom_tokens))


__all__ = ["SafeBudgetInputs", "safe_budget"]
