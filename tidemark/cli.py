"""``tidemark`` command line.

    tidemark replay  --trace examples/traces/demo.jsonl [--policy tidemark|apc|full-prefetch] [--load 0.6]
    tidemark serve   --config configs/testbed/device_edge_cloud.yaml
    tidemark inspect --telemetry ./telemetry
    tidemark rates   --show configs/testbed/rates.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tidemark.version import __version__


def _cmd_replay(args: argparse.Namespace) -> int:
    from tidemark.sim.replay import Policy, load_trace, replay_trace

    trace = load_trace(args.trace)
    policies = [Policy(p) for p in args.policy] if args.policy else [Policy.APC, Policy.FULL_PREFETCH, Policy.TIDEMARK]
    rows = []
    for p in policies:
        rep = replay_trace(trace, policy=p, load=args.load, seed=args.seed)
        rows.append(rep.summary())
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    cols = ["policy", "switches", "p50_switch_ttft_ms", "p95_switch_ttft_ms", "cached_token_ratio", "background_compute_s", "tpot_inflation_ms", "committed_intervals"]
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from tidemark.runtime.config import load_config
    from tidemark.runtime.server import serve

    cfg = load_config(args.config)
    return serve(cfg, listen=args.listen or cfg.scheduler.listen)


def _cmd_inspect(args: argparse.Namespace) -> int:
    d = Path(args.telemetry)
    tickets = d / "tickets.jsonl"
    if not tickets.exists():
        print(f"no ticket log under {d}", file=sys.stderr)
        return 1
    counts = {}
    tokens = 0
    for line in tickets.read_text().splitlines():
        row = json.loads(line)
        ev = row.get("event")
        counts[ev] = counts.get(ev, 0) + 1
        if ev == "committed":
            tokens += int(row.get("delta", 0))
    print("ticket lifecycle")
    for k in ("issued", "admitted", "committed", "refused", "cancelled", "stale", "expired"):
        if k in counts:
            print(f"  {k:<10} {counts[k]}")
    print(f"  committed tokens: {tokens}")
    return 0


def _cmd_rates(args: argparse.Namespace) -> int:
    from tidemark.scheduler.cost_model import RateTable

    table = RateTable.load(args.show)
    print(f"{'engine':<12}{'tier':<8}{'model':<18}{'tau_fg ms/Ktok':>16}{'tau_bg ms/Ktok':>16}{'bg/fg':>8}")
    for e in table.engines():
        r = table.get(e)
        print(f"{e:<12}{r.tier:<8}{r.model_id:<18}{r.tau_fg_ms_per_token*1000:>16.1f}{r.tau_bg_ms_per_token*1000:>16.1f}{r.bg_fg_ratio:>8.2f}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="tidemark", description="Model switching with versioned KV frontiers.")
    p.add_argument("--version", action="version", version=f"tidemark {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("replay", help="replay a route trace through the scheduler without GPUs")
    r.add_argument("--trace", required=True)
    r.add_argument("--policy", action="append", choices=["apc", "full-prefetch", "tidemark"])
    r.add_argument("--load", type=float, default=0.6, help="foreground utilisation of each engine")
    r.add_argument("--seed", type=int, default=7)
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_cmd_replay)

    s = sub.add_parser("serve", help="run the global scheduler process")
    s.add_argument("--config", "-c", required=True)
    s.add_argument("--listen")
    s.set_defaults(fn=_cmd_serve)

    i = sub.add_parser("inspect", help="summarise a telemetry directory")
    i.add_argument("--telemetry", default="./telemetry")
    i.set_defaults(fn=_cmd_inspect)

    ra = sub.add_parser("rates", help="show fitted per-engine rates")
    ra.add_argument("--show", required=True)
    ra.set_defaults(fn=_cmd_rates)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
