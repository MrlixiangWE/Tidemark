# Global frontier scheduling

With frontiers visible in the catalog, the scheduler decides, across tenants
and tiers, which one to advance with the capacity foreground decode leaves idle
on each engine. It issues that work in bounded intervals that fit the short
windows engines actually have, and re-ranks after every commit.

## When an epoch runs

A ranking epoch starts whenever

- a session history grows (`on_turn_served`), or
- an engine reports a state transition (`on_engine_transition`), or
- a ticket reaches a terminal state (`on_ticket_result`).

Epochs are cheap (median 0.4 ms for 60 sessions on the edge module) because the
candidate set is just the lagging entries of the catalog.

## Candidates

Every lagging frontier `F_{s,m,c} < |H_s^(m)|` for a non-serving model is a
candidate, carrying the largest interval it may advance:

```
Δmax = min(1024, |H_s^(m)| − F_{s,m,c})
```

The destination engine picks the admitted size below `Δmax` from `{256, 512,
1024}` at admission time (see [engine-admission.md](engine-admission.md)).

## Future-use probability

A candidate is only useful if the destination is likely to serve the session
again. Tidemark blends an application-supplied router signal with a
session-history transition estimate:

```
p_{s,m} = α · p̂_router_{s,m} + (1 − α) · p̂_hist_{s,m}
```

`α = 1` (the default) is a pure router signal. The history prior on its own is
a poor guide early in a session and our sensitivity study shows a
deployment with no router signal does better disabling preparation than
trusting the prior; it is here as the `α = 0` end of the blend and as a
fallback when the router has no opinion for a session.

## Marginal-value ranking

Let `C_m(x) ≈ τ_fg · x` be the critical-path prefill time of model `m` for an
uncached suffix of `x` tokens. Advancing frontier `F` by `Δ` removes expected
latency

```
B(a) = p_{s,m} · [ C_m(|H| − F) − C_m(|H| − F − Δ) ]
```

at a resource cost of

```
R(a) = T_m^bg(Δ) + λ_M · M_m(Δ)         T_m^bg(Δ) ≈ τ_bg · Δ
```

where `M_m(Δ)` is the KV bytes the interval adds and `λ_M` (64 ms/GiB by
default) converts occupancy into time units. The score is

```
Score(a) = B(a) / max(R(a), ε)
```

Under linear fits both `B` and `R` are proportional to `Δ`, so the score is a
per-token marginal value that does not depend on the interval length. That is
the property that lets the engine choose the admitted size later without
disturbing the ranking.

### Why two rates

`τ_fg` and `τ_bg` measure different things. `τ_fg` is wall time per token on
the critical path of a switch. `τ_bg` is the incremental scheduled compute time
an interval adds when it is batched alongside foreground decode. Their ratio
on our testbed:

| tier | τ_fg (ms/Ktok) | τ_bg / τ_fg |
|---|---|---|
| cloud (Qwen2.5-14B, 2× PCIe GPU) | 191 | 0.38 |
| edge (Qwen2.5-7B, AGX Orin) | 732 | 0.47 |
| device (Llama-3.2-1B, phone SoC) | 2560 | 0.91 |

A background token on the cloud engine fills capacity a decode-heavy batch
leaves unused; a device engine has little spare width to absorb it. Tiers
therefore differ both in how much latency a prepared token removes and in how
much compute it takes to prepare, and ranking in raw tokens would get the order
wrong. Removing the `τ_bg` weighting is one of our ablations.

## Tenant isolation

Ranking by score alone concentrates background work on whichever tenant has
the most active sessions. Tidemark caps each tenant `u` on two axes:

- at most `κ_u` outstanding admitted tickets (default 2), and
- at most `β_u · G_total` of the aggregate background budget (default 0.35),
  where `G_total = Σ_e τ_bg(e) · X̄(e)` sums the safe budgets engines reported
  over the epoch in `τ_bg`-weighted compute time.

A single issue point is what makes these caps bind: caps applied per tier would
let one tenant hold `κ_u` tickets on each tier. Tenants skipped while eligible
accumulate aging so a stream of high-score arrivals cannot starve them.

## One ranking epoch

```
A ← ∅
for each F_{s,m,c} with m non-serving and F < |H_s^(m)|:
    A ← A ∪ {(s, m, c, Δmax)}
compute Score(a) for a ∈ A
for a ∈ A in descending Score:
    if tickets(u_a) ≥ κ_u  or  bg(u_a) + τ_bg·Δmax > β_u·G_total:  skip
    else if engine e_a has an in-flight ticket:                      skip
    else issue a; charge u_a
age tenants skipped while eligible
```

Implementation: `tidemark/scheduler/ranking.py` (`RankingEpoch.run`) and
`tidemark/scheduler/global_scheduler.py` (candidate construction, ticket
issue, result handling).

## Ablation switches

`SchedulerConfig` exposes the switches our ablations use:

| switch | effect |
|---|---|
| `ablate_versioned_keys` | collapse `(s, m, c)` to a session-level key |
| `ablate_bounded_intervals` | `Δmax = lag` (whole-suffix tickets) |
| `ablate_tau_bg_weighting` | `R(a) = Δ` (rank in raw tokens) |
| `round_robin_ranking` | ignore score, rotate over sessions (the CPP baseline) |

All default to off.
