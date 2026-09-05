# FAQ

**Does Tidemark route requests?**
No. The application router remains the only component that decides which
model serves a turn. Tidemark takes that choice as input, marks every other
model lagging, and optionally consumes the router's probabilities over future
destinations. See `tidemark/router/` for the small contract a router
implements.

**Does it move KV state between models?**
No. Models provisioned independently for each tier do not share KV state, and
even two sizes of one family produce tensors of different shapes. Tidemark
schedules each destination's *own* prefill of the suffix it is missing, in
bounded intervals, when that engine has idle capacity.

**How is this different from prefetching the whole suffix?**
A whole-suffix prefill treats the frontier lag as one unit of work. Under load
it either competes with in-flight decode or waits for a window large enough to
hold it, and windows that large are rare. Bounded intervals fit the short safe
budgets engines actually have and commit on their own, so a burst that cancels
one interval does not discard the ones before it. The versioned catalog is what
makes that partial progress safe to continue later.

**What if the router is wrong about the next destination?**
The interval was computed for nothing, at a cost the local guard kept out of
the foreground's way. The tenant caps bound how much of the background budget
one tenant's bad predictions can consume. The paper's `α` sweep shows the
router signal is what matters; if you have no router probabilities at all you
are better off disabling preparation than trusting the history prior.

**Can a background interval ever produce a wrong answer?**
No. Background work runs the destination model's ordinary prefill path and
becomes visible through the ordinary prefix cache; a foreground request that
finds it reuses it exactly as it would reuse its own earlier prefill. The
validity predicate ensures only state that matches the current history is ever
counted, and over 240 switches per tier in the paper a committed frontier and a
clean full prefill yielded identical next-token ids.

**What does Tidemark cost on the engines?**
On the paper's cloud engine the scheduler-facing overhead is a JSON post per
ticket state transition and about 120 lines in the scheduler iteration. The
scheduler process itself uses about 2 % of one CPU core on the edge module and
56 KB of catalog for 60 sessions.

**What happens if the scheduler dies?**
Ticket issue stops. Every engine keeps serving foreground work with its stock
scheduler. When the scheduler comes back it rebuilds the catalog from session
logs and what each engine reports resident.

**Why Python for the scheduler?**
It is off the foreground path. A ranking epoch takes 0.4 ms at the median for
60 sessions and a ticket only proposes work; nothing a user is waiting on goes
through the scheduler process.

**Which vLLM versions work?**
The patch targets vLLM 0.10.x with the V1 engine. CI dry-runs it against
`v0.10.1`. Older releases lack request-level `cached_tokens`, which the commit
path depends on.

**Can I use it with SGLang / TensorRT-LLM / MLC?**
Not yet. The adapter interface in `tidemark/engines/base.py` is small; an
engine needs to expose per-iteration foreground load and KV occupancy to the
local admission controller, accept a prefill-only request, and report the
resident prefix after it. Contributions welcome.
