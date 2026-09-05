# Engine-local admission and commit

The scheduler proposes; the engine decides. Each scheduling iteration first
admits foreground decode and foreground prefill, so that background work never
delays them, and then fits at most one interval into what is left. This page
describes that decision and what happens after the interval finishes.

## The safe budget

Let `B_t` be the engine token budget of iteration `t` (vLLM's
`max_num_batched_tokens`) and let `D_t` and `P_t` be the foreground decode and
prefill tokens already selected. The safe budget is

```
X_t = max(0, min(B_t − D_t − P_t,  X_max,  X_t^KV))
```

`X_max` caps a single admission (1024 by default) and `X_t^KV` is the number of
tokens that fit below the configured KV headroom (8 % of blocks kept free).
`X_t` is recomputed on every iteration. Under medium load on the cloud engine
its median non-zero value is about 430 tokens against a median missing suffix
of 1,174; whole-suffix admission would discard most of those windows.

Implementation: `tidemark/admission/safe_budget.py`.

## The three-mode decision

```
          ⎧ Idle,     D_t = P_t = 0
Mode(t) = ⎨ Mixed,    D_t + P_t > 0  ∧  X_t > 0  ∧  ok_t
          ⎩ Blocked,  otherwise
```

`ok_t` holds while the exponentially weighted moving average of the engine's
foreground TPOT stays within a configured multiple of a reference value:

```
TPOT_t^ewma ≤ (1 + γ) · TPOT_ref          γ = 0.03
```

`TPOT_ref` is fixed to the median of 200 foreground-only calibration steps
taken when the engine starts. Before calibration the guard is conservative and
admits nothing while foreground work is present.

Implementation: `tidemark/admission/guard.py`.

## Choosing the interval

Interval sizes come from a small fixed set, `D = {256, 512, 1024}`, trading
preemptibility against per-request overhead.

- **Idle** admits the largest `Δ ∈ D` with `Δ ≤ min(Δmax, X_t)`.
- **Mixed** admits at most one interval with the same bound.
- **Blocked** admits nothing.

A remaining lag shorter than 256 tokens is admitted whole if it fits, so a
frontier can actually reach the end of the history.

A foreground request that arrives while an interval is in flight stops it at
the next scheduler boundary. The engine reports `cancelled` with the tokens it
computed; those blocks stay in the prefix cache and may be found by a later
request, but the *logical* frontier does not move, because the scheduler cannot
know how much of the interval survived eviction. The next epoch re-ranks and
reissues from the committed frontier.

Implementation: `tidemark/admission/controller.py`.

## Commit and reuse

A background task runs the destination model's ordinary prefill path but
bypasses sampling and response construction. When the interval finishes the
engine reports:

- the admitted size `Δ`,
- the cached prefix it found on arrival,
- the prefix physically resident after completion, and
- the content hash over the tokens it prefilled.

The scheduler evaluates `Valid(a)` against the catalog. A valid interval
advances `F` and becomes visible through the engine's ordinary prefix-cache
lookup, so a future foreground request needs no special reuse path. A stale or
partial completion produces no logical advance. Tidemark reuses each engine's
existing cache management and replacement instead of introducing a second
physical cache manager.

Implementation: `tidemark/admission/commit.py`, `tidemark/catalog/validity.py`.

## What we measured

On the cloud engine over a 0.6 s medium-load episode: admitted sizes follow
the safe-budget envelope, 1024 tokens while the engine is lightly loaded and
256 as a burst consumes the budget. Single foreground arrivals dent the
envelope and stop the in-flight interval at the next boundary; admission
resumes at the following iteration. Removing the guard entirely moves the
switch tail the least of our ablations but inflates TPOT by 65.7 %,
which is the trade whole-suffix prefetch makes at high load.

## Device tier

`llama-server` has a single slot per device and no batching iteration. The
adapter derives the mode from slot state: a slot decoding a foreground request
is `Blocked`, an idle slot is `Idle`, and there is no `Mixed`. This matches the
measured `τ_bg / τ_fg ≈ 0.9` on phones and boards, where there is little spare
width to interleave into anyway.
