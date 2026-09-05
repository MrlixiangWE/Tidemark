import json
import subprocess
import sys
from pathlib import Path

import pytest

from tidemark.sim.replay import Policy, replay_trace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def trace():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_trace.py"), "--sessions", "12", "--seed", "1"], check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line and not line.startswith("#")]


def test_replay_runs_all_policies(trace):
    reports = {p: replay_trace(trace, policy=p, load=0.6, seed=1) for p in Policy}
    for rep in reports.values():
        assert rep.switches > 0
        assert len(rep.switch_ttft_ms) == rep.switches
    assert reports[Policy.APC].background_ms == 0.0
    assert reports[Policy.TIDEMARK].committed_intervals > 0


def test_tidemark_lowers_switch_tail_versus_reactive(trace):
    apc = replay_trace(trace, policy=Policy.APC, load=0.6, seed=1)
    tm = replay_trace(trace, policy=Policy.TIDEMARK, load=0.6, seed=1)
    assert tm.p95() < apc.p95()


def test_tidemark_spends_less_than_full_prefetch(trace):
    fp = replay_trace(trace, policy=Policy.FULL_PREFETCH, load=0.6, seed=1)
    tm = replay_trace(trace, policy=Policy.TIDEMARK, load=0.6, seed=1)
    assert tm.background_ms < fp.background_ms
    assert tm.tpot_inflation_ms <= fp.tpot_inflation_ms


def test_cli_replay_smoke(tmp_path, trace):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in trace))
    out = subprocess.run([sys.executable, "-m", "tidemark.cli", "replay", "--trace", str(p), "--policy", "tidemark", "--json"], check=True, capture_output=True, text=True, cwd=ROOT).stdout
    rows = json.loads(out)
    assert rows[0]["policy"] == "tidemark"
