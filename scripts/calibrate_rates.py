#!/usr/bin/env python3
"""Fit tau_fg and tau_bg for every engine in a config.

For each engine the script

1. clears the prefix cache,
2. prefills prompts of 512, 1K, 2K, 4K and 8K tokens with nothing cached and
   records TTFT; the slope through the origin is ``tau_fg``,
3. starts a steady decode load (``--decode-clients`` streaming requests),
   submits prefill-only requests of 256, 512 and 1024 tokens, and records the
   extra scheduler time each adds; that slope is ``tau_bg``,
4. records the median decode step time under the same load as ``TPOT_ref``.

Output is a rates.yaml the scheduler loads directly.

    python scripts/calibrate_rates.py --config configs/testbed/device_edge_cloud.yaml \
        --out configs/testbed/rates.yaml --repeats 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

import yaml

from tidemark.runtime.config import load_config
from tidemark.scheduler.cost_model import EngineRates, RateTable, fit_rate

FG_SIZES = (512, 1024, 2048, 4096, 8192)
BG_SIZES = (256, 512, 1024)


def _client(engine):
    if engine.backend == "llamacpp":
        from tidemark.engines.llamacpp.adapter import LlamaCppAdapter  # noqa: F401  (import check)

        raise SystemExit("llama.cpp calibration uses scripts/calibrate_device.sh on the device itself")
    from tidemark.engines.vllm.client import TokenIdClient

    return TokenIdClient(engine.endpoint, engine.model)


def _synthetic_prompt(n: int, seed: int) -> List[int]:
    # Distinct ids per size and repeat so nothing is served from cache.
    base = 1000 + 7919 * seed
    return [(base + i * 31) % 32000 + 100 for i in range(n)]


def fit_foreground(client, repeats: int) -> Tuple[float, List[Tuple[int, float]]]:
    samples: List[Tuple[int, float]] = []
    for r in range(repeats):
        for n in FG_SIZES:
            usage = client.foreground(_synthetic_prompt(n, r * 10 + len(samples)), max_tokens=4)
            if usage.cached_tokens:
                print(f"  warning: {usage.cached_tokens} tokens were cached; result may be optimistic", file=sys.stderr)
            samples.append((usage.uncached_tokens, usage.ttft_ms or usage.e2e_ms))
    return fit_rate(samples), samples


def fit_background(client, repeats: int, decode_clients: int) -> Tuple[float, float]:
    """Return (tau_bg, tpot_ref_ms) under a steady decode load."""
    stop = False
    tpots: List[float] = []

    def decoder(seed: int) -> None:
        while not stop:
            u = client.foreground(_synthetic_prompt(256, seed), max_tokens=256, seed=seed)
            if u.tpot_ms:
                tpots.append(u.tpot_ms)

    with cf.ThreadPoolExecutor(max_workers=decode_clients) as pool:
        futs = [pool.submit(decoder, 500 + i) for i in range(decode_clients)]
        time.sleep(5.0)  # let decode settle
        baseline = statistics.median(tpots[-50:]) if len(tpots) >= 10 else float("nan")
        samples: List[Tuple[int, float]] = []
        for r in range(repeats):
            for n in BG_SIZES:
                t0 = time.perf_counter()
                client.completion({"model": client.model, "prompt": _synthetic_prompt(n, 900 + r * 10 + n), "max_tokens": 1, "priority": 1_000_000})
                samples.append((n, (time.perf_counter() - t0) * 1000.0))
                time.sleep(0.5)
        stop = True
        for f in futs:
            f.cancel()
    return fit_rate(samples), baseline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-clients", type=int, default=8)
    ap.add_argument("--only", action="append", help="engine id(s) to calibrate")
    args = ap.parse_args()

    cfg = load_config(args.config)
    table = RateTable.load(cfg.rates_file) if cfg.rates_file and Path(cfg.rates_file).exists() else RateTable()
    for e in cfg.engines:
        if args.only and e.engine_id not in args.only:
            continue
        print(f"== {e.engine_id} ({e.tier}, {e.model})")
        client = _client(e)
        tau_fg, fg = fit_foreground(client, args.repeats)
        print(f"  tau_fg = {tau_fg * 1000:.1f} ms/Ktok over {len(fg)} points")
        tau_bg, tpot_ref = fit_background(client, args.repeats, args.decode_clients)
        print(f"  tau_bg = {tau_bg * 1000:.1f} ms/Ktok, TPOT_ref = {tpot_ref:.1f} ms, bg/fg = {tau_bg / tau_fg:.2f}")
        kv = e.kv_bytes_per_token or (table.get(e.engine_id).kv_bytes_per_token if e.engine_id in table else 65536)
        table.put(EngineRates(e.engine_id, e.model, e.tier, tau_fg, tau_bg, kv, tpot_ref_ms=tpot_ref, samples_fg=len(fg), samples_bg=args.repeats * len(BG_SIZES)))
    out = Path(args.out)
    payload = {"engines": {}}
    for eid in table.engines():
        r = table.get(eid)
        payload["engines"][eid] = {
            "model": r.model_id,
            "tier": r.tier,
            "tau_fg_ms_per_ktok": round(r.tau_fg_ms_per_token * 1000, 1),
            "tau_bg_ms_per_ktok": round(r.tau_bg_ms_per_token * 1000, 1),
            "kv_bytes_per_token": r.kv_bytes_per_token,
            "tpot_ref_ms": None if r.tpot_ref_ms is None else round(r.tpot_ref_ms, 1),
        }
    out.write_text("# generated by scripts/calibrate_rates.py\n" + yaml.safe_dump(payload, sort_keys=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
