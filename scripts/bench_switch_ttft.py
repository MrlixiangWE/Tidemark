#!/usr/bin/env python3
"""Measure model-switch TTFT as a function of frontier lag on one engine.

Reproduces the "switch cost scales linearly with the lag" measurement: fix the
history at N tokens, cache a prefix of length N - lag on the engine, then issue
the full prompt and time the first token.

    python scripts/bench_switch_ttft.py --endpoint http://10.0.1.10:8000 \
        --model Qwen/Qwen2.5-14B-Instruct --history 8192 --lags 0 1024 2048 4096 8192
"""

from __future__ import annotations

import argparse
import statistics

from tidemark.engines.vllm.client import TokenIdClient


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--history", type=int, default=8192)
    ap.add_argument("--lags", type=int, nargs="+", default=[0, 1024, 2048, 4096, 8192])
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    client = TokenIdClient(args.endpoint, args.model)
    print(f"{'lag':>6} {'cached':>7} {'ttft_ms (median)':>18} {'ms/Ktok':>9}")
    for lag in args.lags:
        ttfts = []
        for r in range(args.repeats):
            ids = [(r * 104729 + i * 31) % 32000 + 100 for i in range(args.history)]
            warm = ids[: args.history - lag]
            if warm:
                client.completion({"model": args.model, "prompt": warm, "max_tokens": 1, "temperature": 0})
            u = client.foreground(ids, max_tokens=8)
            ttfts.append(u.ttft_ms or u.e2e_ms)
        med = statistics.median(ttfts)
        rate = (med / lag * 1000.0) if lag else float("nan")
        print(f"{lag:>6} {args.history - lag:>7} {med:>18.1f} {rate:>9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
