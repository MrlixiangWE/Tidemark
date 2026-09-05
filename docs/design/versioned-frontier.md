# The versioned KV frontier

## Why a residency bit is not enough

Prefix caches and session-retention policies describe KV state as *what is
resident now, for the model that produced it*. That is a session-level bit:
either the engine still has this session's blocks or it does not. It answers
"can the request that just arrived reuse something?" and nothing else.

Background preparation asks different questions. An interval may complete
after the history it was computed against has been edited, after the engine
has evicted part of the prefix it extends, or after a foreground request has
advanced the same frontier. To tell a completed interval that is still useful
from one that is not, the catalog has to record progress as a *versioned
position*, and it has to do so per model: a device model and a cloud model
tokenise the same text differently, so "1,174 tokens" means a different place
in the conversation on each of them.

## What is stored

```
r_s                                  current text revision of session s
F : (s, m, c) ↦ ⟨r, F, L, q, g⟩      one entry per compatible (model, runtime config)
```

| field | meaning |
|---|---|
| `r` | revision that produced the committed state |
| `F` | committed token position in `H_s^(m)`, the history under `m`'s tokenizer |
| `L` | physical placement: `none`, `resident` on an engine, or `evicted` |
| `q` | optional in-flight target, the end of the interval a ticket is working on |
| `g` | generation, the per-model projection of `r_s`; bumped by anything that could make in-flight work stale |

The entry costs about 312 bytes in CPython. Sixty sessions across three models
is under 60 KB, four orders of magnitude below the KV memory it describes.

## Invariants

1. **Compatibility.** A frontier is consumed only by the model and runtime
   class that created it. Keys are `(s, m, c)`; `c` captures quantisation,
   attention backend and block size, because a prefix cached under one runtime
   configuration is not reusable under another even for the same weights.
2. **Prefix validity.** A committed interval matches the current
   model-tokenised prefix. Checked by the validity predicate at commit.
3. **Monotonicity.** Absent eviction and revision rewrites, `F` only advances.
4. **Foreground priority.** Background work never displaces a foreground
   request, and an interval truncated by one does not commit.

## The validity predicate

A completed ticket `a = ⟨u, s, m, c, e, r_a, g_a, [F_a, F_a+Δ)⟩` is accepted iff

```
Valid(a) = 1[h(H_a[0:F_a+Δ]) = h(H_s[0:F_a+Δ])]   the tokens are still the history's tokens
         · 1[g_a = g_{s,m,c}]                      nobody touched this key since issue
         · 1[F_a = F_{s,m,c}]                      the interval continues F exactly
```

`h` is a content hash over a token range. Engines already keep such hashes for
block-level prefix lookup; the catalog computes the same thing at interval
granularity. Validity tests the identity of *the interval a ticket produced*,
not the equality of two whole-history versions. That is why an **append** after
issue does not invalidate a ticket (the prefix is unchanged) while an **edit**
inside the interval does.

Implementation: `tidemark/catalog/validity.py`. The predicate is a pure
function so the same code runs in unit tests and in the scheduler.

## Revision transitions

When a turn is edited, regenerated or branched, the catalog increments `r_s`
and the generation of every compatible key, conservatively invalidating every
in-flight ticket for the session. Each committed frontier then retracts:

```
F_{s,m,c} ← min(F_{s,m,c}, LCP(H_old^(m), H_new^(m)))
```

Blocks beyond the retracted frontier are left to the engine's ordinary
replacement policy. There is no second physical cache manager.

## Eviction as retraction

When an engine reclaims blocks backing part of a committed suffix, it reports
the longest prefix still resident and the catalog retracts `F` to it. Engine
loss is the degenerate case: placement is invalidated and `F` drops to zero.

## How foreground requests use it

A foreground request on model `m` advances `F_{s,m,c}` through the same rule as
a background commit, with `Δ` equal to the prompt prefix that remains resident
after the request. It also bumps the generation, so a background ticket for the
same key that was in flight cannot land on top of newer state.

## Recovery

The catalog is soft state. After a scheduler restart it replays the session
logs and resets each frontier to no more than the longest prefix the
corresponding engine reports resident. Updates are idempotent, so a retry after
a network failure is harmless.
