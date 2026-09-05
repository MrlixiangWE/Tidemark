#!/usr/bin/env python3
"""Generate a small synthetic route trace for the quick-start replay.

The trace has the shape of our mobility replay: a handful of tenants,
sessions of 6-14 turns, a difficulty router that sends turns to device / edge /
cloud, and reversible constraint events (link drop, battery floor) that force a
temporary fallback and a later return. Prompts are filler text; only their
lengths matter to the scheduler.

    python scripts/make_demo_trace.py --sessions 24 --seed 3 > examples/traces/demo.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys

MODELS = {"device": "llama-3.2-1b", "edge": "qwen2.5-7b", "cloud": "qwen2.5-14b"}
ORDER = ("device", "edge", "cloud")
WORDS = (
    "please summarise the previous discussion and list the open questions we still need to settle "
    "before the review; include the trade offs between latency, energy and cost that were raised, "
    "and propose a concrete schedule for the next three iterations of the prototype"
).split()


def filler(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--tenants", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    events = []
    for s in range(args.sessions):
        sid = f"s{s:03d}"
        tenant = f"tenant-{rng.randrange(args.tenants):02d}"
        turns = rng.randint(6, 14)
        t = rng.uniform(0, 30)
        # Each session leans toward one tier but wanders with task difficulty.
        home = rng.choices(ORDER, weights=(0.35, 0.4, 0.25))[0]
        link_down_until = -1.0
        low_battery = rng.random() < 0.2
        for i in range(turns):
            difficulty = rng.random()
            wanted = home
            if difficulty > 0.8:
                wanted = "cloud"
            elif difficulty < 0.25:
                wanted = "device"
            feasible = set(ORDER)
            if rng.random() < 0.08:
                link_down_until = t + rng.uniform(2, 6)
            if t < link_down_until:
                feasible -= {"edge", "cloud"}
            if low_battery and i > turns // 2 and len(feasible) > 1:
                feasible -= {"device"}  # the device stays reachable as a last resort
            tier = wanted if wanted in feasible else max(feasible, key=ORDER.index)
            p_next = {MODELS[x]: 0.15 for x in ORDER}
            p_next[MODELS[tier]] = 0.7
            events.append(
                {
                    "t": round(t, 3),
                    "session": sid,
                    "tenant": tenant,
                    "turn": i,
                    "model": MODELS[tier],
                    "reason": "task" if tier == wanted else "constraint",
                    "text": filler(rng, rng.randint(40, 260)),
                    "p_next": p_next,
                }
            )
            t += rng.uniform(1.5, 6.0)
    events.sort(key=lambda e: e["t"])
    out = sys.stdout
    out.write("# synthetic mobility-replay trace for `tidemark replay`; fields: t, session, tenant, turn, model, reason, text, p_next\n")
    for e in events:
        out.write(json.dumps(e) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
