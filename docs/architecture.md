# Architecture

Tidemark is a runtime layer between an application-level model router and the
inference engines of a device-edge-cloud deployment. Its job is narrow: keep
the KV state of the models that are *not* serving a session close to the
conversation, so that when the router switches to one of them the destination
has most of the prefix resident and the first token comes back quickly.

Nothing about the foreground path changes. A request goes from the router to
its engine directly. Tidemark learns about it afterwards, decides which lagging
frontier is worth advancing with the capacity foreground decode leaves idle,
and proposes bounded background work that each engine is free to refuse.

```
                          application router
                                 │  route(turn) → model
                                 │  p(next destination)      (optional)
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │  Tidemark scheduler process                       (runs on an edge box) │
 │                                                                          │
 │   ❶ versioned KV frontier catalog     F : (s, m, c) ↦ ⟨r, F, L, q, g⟩    │
 │        history grows → mark non-serving models lagging                   │
 │        edit / eviction → retract, bump generation                        │
 │                                                                          │
 │   ❷ global frontier scheduler         one ranking epoch per event        │
 │        candidates = lagging frontiers, Δmax = min(1024, lag)             │
 │        score = B(a) / R(a)   (latency removed per unit bg compute)       │
 │        per-tenant caps κ, β · G_total; aging; ≤ 1 ticket per engine      │
 └──────────────┬──────────────────────┬──────────────────────┬────────────┘
                │ atomic ticket        │                      │
                ▼                      ▼                      ▼
   ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐
   │ device · llama.cpp │  │ edge · vLLM        │  │ cloud · vLLM       │
   │ ❸ commit/reject    │  │ ❸ admission shim   │  │ ❸ admission shim   │
   │   adapter          │  │   X_t, Mode(t)     │  │   X_t, Mode(t)     │
   │   Idle / Blocked   │  │   ≤1 interval/iter │  │   ≤1 interval/iter │
   └────────────────────┘  └────────────────────┘  └────────────────────┘
                ▲ ticket result: admitted / committed / cancelled / stale / refused
```

## The three components

### ❶ Versioned KV frontier catalog — `tidemark/catalog/`

For every session `s` the catalog stores the current text revision `r_s`. For
every compatible `(model m, runtime configuration c)` it stores

```
F : (s, m, c) ↦ ⟨r, F, L, q, g⟩
```

`F` is the committed token position in that model's own tokenisation of the
history, `L` the physical placement, `q` an optional in-flight target, and `g`
a generation number. The entry separates *logical* progress from *physical*
allocation: the engine keeps ownership of block hashing, placement and
reclamation, and the frontier only advances after the engine confirms a
physical commit.

A completed interval is accepted only if the validity predicate holds:

```
Valid(a) = 1[h(H_a[0:F+Δ]) = h(H_s[0:F+Δ])] · 1[g_a = g_{s,m,c}] · 1[F_a = F_{s,m,c}]
```

The three terms mean: the tokens the ticket prefilled are still the tokens of
the history; nothing (a foreground request, an edit, an eviction) bumped the
generation since the ticket was issued; and the interval continues the
committed frontier exactly. Appends never invalidate a ticket. Edits,
regenerations and branches do, and also retract each committed frontier to the
longest common prefix of the old and new histories.

→ [design/versioned-frontier.md](design/versioned-frontier.md)

### ❷ Global frontier scheduler — `tidemark/scheduler/`

A ranking epoch starts whenever a history grows or an engine reports a state
transition. It enumerates lagging frontiers, gives each a bounded interval
`Δmax = min(1024, |H| − F)`, and scores it by the expected switch latency it
removes per unit of background compute time:

```
B(a) = p_{s,m} · [C_m(|H| − F) − C_m(|H| − F − Δ)]        expected latency removed
R(a) = T_m^bg(Δ) + λ_M · M_m(Δ)                            bg compute + KV occupancy
Score(a) = B(a) / max(R(a), ε)
```

`p_{s,m}` blends the router's destination probabilities with a per-session
transition prior (`α`). `C_m` and `T_m^bg` are linear in tokens with rates
fitted per engine, and those rates differ a lot: a background token on the
cloud engine costs about 0.38 of a foreground token's time, on the device
about 0.91. Ranking in raw tokens would therefore be wrong across tiers.

Per-tenant caps (at most `κ` outstanding tickets, at most `β · G_total` of the
aggregate background budget) and queue aging keep one busy tenant from taking
every idle window. At most one ticket is issued per engine per epoch.

→ [design/global-scheduler.md](design/global-scheduler.md)

### ❸ Engine-local admission and commit — `tidemark/admission/`, `tidemark/engines/`

Each engine serves foreground decode and prefill first, then fits at most one
interval into what is left:

```
X_t = max(0, min(B_t − D_t − P_t, X_max, X_t^KV))                       safe budget
Mode(t) = Idle    if D_t = P_t = 0
          Mixed   if D_t + P_t > 0 ∧ X_t > 0 ∧ TPOT_ewma ≤ (1+γ)·TPOT_ref
          Blocked otherwise
```

The admitted size comes from `D = {256, 512, 1024}`: the largest that fits in
`min(Δmax, X_t)`. The decision is re-evaluated on every scheduler iteration,
not once per ticket, which is what lets short windows be used at all. A
foreground arrival stops the in-flight interval at the next scheduler boundary
and it does not commit.

On vLLM this is a ~120-line patch to the V1 scheduler that calls into the shim
in `tidemark/engines/vllm/shim.py`. On llama.cpp, which has no batching
iteration to hook, an adapter drives the server from outside and a small server
patch adds commit accounting to the completion response.

→ [design/engine-admission.md](design/engine-admission.md),
[deployment/vllm.md](deployment/vllm.md), [deployment/llamacpp.md](deployment/llamacpp.md)

## Ticket lifecycle

```
 scheduler                          engine
 ─────────                          ──────
 issue a=⟨u,s,m,c,e,r,g,[F,F+Δmax)⟩ ─────▶ enqueue prefill-only request
                                            each iteration: X_t, Mode(t)
                                   ◀─────  admitted(Δ)          Δ ∈ D, Δ ≤ min(Δmax, X_t)
                                            ... prefill [F, F+Δ) ...
                                   ◀─────  committed(Δ, h)   │ cancelled  │ refused
 Valid(a)? ── yes ─▶ F ← F+Δ, re-rank      │                 │ (fg arrival)│ (blocked)
          └─ no  ─▶ stale, F unchanged      ▼
```

Every ticket ends in exactly one of `committed`, `refused`, `cancelled`,
`stale` or `expired`, and all of them are written to `telemetry/tickets.jsonl`.

## Failure model

- **A ticket fails or is rejected.** Nothing changes in the catalog; the
  interval is re-ranked in the next epoch.
- **A foreground request advances the same frontier.** It bumps the
  generation; the in-flight ticket's later completion fails validation.
- **An engine evicts blocks.** It reports the longest prefix still resident and
  the catalog retracts to it.
- **An engine disappears.** Its placements are invalidated and every frontier
  it held drops to zero.
- **The scheduler restarts.** The catalog is soft state: it replays session logs
  and conservatively resets each frontier to what each engine reports resident.
- **The scheduler is gone entirely.** Ticket issue stops. Every engine keeps
  serving foreground work with its stock scheduler.

## Where the numbers come from

`configs/testbed/rates.yaml` holds the fitted `τ_fg` and `τ_bg` for our
testbed. `scripts/calibrate_rates.py` regenerates them for new hardware; the
scheduler also refines `τ_bg` online from the measured compute time of each
committed interval.
