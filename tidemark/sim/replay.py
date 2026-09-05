from __future__ import annotations

import enum
import json
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import EngineLocalAdmission, StepState
from tidemark.admission.guard import TpotGuard
from tidemark.catalog.history import TokenizerRegistry
from tidemark.engines.base import EngineAdapter, EngineDescriptor, ResidentPrefix
from tidemark.runtime.config import EngineConfig, TidemarkConfig
from tidemark.runtime.service import TidemarkRuntime
from tidemark.scheduler.predictor import StaticRouterSignal
from tidemark.scheduler.ticket import AtomicTicket


class Policy(str, enum.Enum):
    APC = "apc"                    # reactive prefix caching: nothing prepared
    FULL_PREFETCH = "full-prefetch"  # whole missing suffix, unbounded, no guard
    TIDEMARK = "tidemark"


def _word_tokenizer(scale: float):
    """A deterministic stand-in tokenizer: ~scale tokens per word."""

    def tok(text: str) -> Sequence[int]:
        out: List[int] = []
        for w in text.split():
            n = max(1, round(scale * (1 + len(w) // 6)))
            out.extend(hash((w, j)) & 0x7FFFFFFF for j in range(n))
        return out

    return tok


class ReplayEngine(EngineAdapter):
    """An engine that advances a clock instead of running a model.

    Foreground load is modelled as a per-engine utilisation ``rho``: with
    probability ``rho`` an iteration carries foreground decode, otherwise it is
    idle. That is enough to reproduce the qualitative behaviour we see on real engines (short
    safe budgets under load, whole-suffix prefetch inflating TPOT) without
    pretending to be a GPU.
    """

    def __init__(self, descriptor: EngineDescriptor, commit: CommitPath, *, rates, rho: float, rng: random.Random, policy: Policy) -> None:
        super().__init__(descriptor, commit)
        self.rates = rates
        self.rho = rho
        self.rng = rng
        self.policy = policy
        self.admission = EngineLocalAdmission(guard=TpotGuard(calibration_steps=1))
        self.admission.guard.calibrate([rates.tpot_ref_ms or 20.0])
        self.resident: Dict[str, Tuple[int, ...]] = {}  # session -> resident token ids
        self.bg_ms = 0.0
        self.fg_tpot_inflation_ms = 0.0
        self._step = 0
        self.pending: List[Tuple[AtomicTicket, int]] = []

    def step_state(self) -> StepState:
        self._step += 1
        busy = self.rng.random() < self.rho
        decode = self.rng.randint(8, 64) if busy else 0
        prefill = self.rng.choice([0, 0, 0, 256, 512]) if busy else 0
        tpot = (self.rates.tpot_ref_ms or 20.0) * (1.0 + (0.08 if busy and self.rng.random() < 0.3 else 0.0))
        return StepState(
            step_id=self._step,
            token_budget=2048,
            decode_tokens=decode,
            prefill_tokens=prefill,
            kv_free_blocks=900,
            kv_total_blocks=1000,
            new_foreground_arrivals=1 if (busy and self.rng.random() < 0.15) else 0,
            last_tpot_ms=tpot if decode else None,
        )

    def submit(self, ticket: AtomicTicket, delta: int) -> None:
        self.pending.append((ticket, delta))

    def cancel(self, ticket_id: str) -> None:
        self.pending = [(t, d) for t, d in self.pending if t.ticket_id != ticket_id]

    def resident_prefix(self, session_id: str, token_ids: tuple) -> ResidentPrefix:
        held = self.resident.get(session_id, ())
        n = 0
        for a, b in zip(held, token_ids):
            if a != b:
                break
            n += 1
        return ResidentPrefix(session_id, n)

    def tick(self) -> None:
        """Advance one scheduler iteration."""
        if not self.pending:
            return
        step = self.step_state()
        ticket, delta_max = self.pending[0]
        if self.policy is Policy.FULL_PREFETCH:
            # Whole suffix, no guard: compute it all and charge foreground.
            delta = delta_max
            self.bg_ms += self.rates.tau_fg_ms_per_token * delta
            if step.decode_tokens:
                self.fg_tpot_inflation_ms += self.rates.tau_fg_ms_per_token * delta * 0.5
            self._finish(ticket, delta, cached=ticket.frontier)
            return
        decision = self.admission.decide(step, ticket_id=ticket.ticket_id, delta_max=delta_max)
        if not decision.admitted:
            return
        delta = decision.admitted_delta
        self.commit.admitted(ticket, delta)
        # Foreground arrival during the interval cancels it (Mixed mode only).
        if step.decode_tokens and self.rng.random() < 0.1:
            self.admission.complete(ticket.ticket_id)
            self.pending.pop(0)
            self.bg_ms += self.rates.tau_bg_ms_per_token * delta * 0.5
            self.commit.cancelled(ticket, prefilled_tokens=delta // 2, gpu_ms=self.rates.tau_bg_ms_per_token * delta * 0.5)
            return
        self.bg_ms += self.rates.tau_bg_ms_per_token * delta
        self.admission.complete(ticket.ticket_id)
        self._finish(ticket, delta, cached=ticket.frontier)

    def _finish(self, ticket: AtomicTicket, delta: int, cached: int) -> None:
        self.pending.pop(0)
        self.resident[ticket.session_id] = ticket.prompt_token_ids(delta)
        self.commit.completed(
            ticket,
            admitted_delta=delta,
            cached_tokens=cached,
            physically_cached_after=ticket.frontier + delta,
            gpu_ms=self.rates.tau_bg_ms_per_token * delta,
        )


@dataclass
class ReplayReport:
    policy: str
    switches: int
    switch_ttft_ms: List[float] = field(default_factory=list)
    cached_ratio: List[float] = field(default_factory=list)
    background_ms: float = 0.0
    tpot_inflation_ms: float = 0.0
    committed_intervals: int = 0
    stale_intervals: int = 0

    def p50(self) -> float:
        return statistics.median(self.switch_ttft_ms) if self.switch_ttft_ms else 0.0

    def p95(self) -> float:
        if not self.switch_ttft_ms:
            return 0.0
        xs = sorted(self.switch_ttft_ms)
        return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]

    def summary(self) -> Dict[str, float]:
        return {
            "policy": self.policy,
            "switches": self.switches,
            "p50_switch_ttft_ms": round(self.p50(), 1),
            "p95_switch_ttft_ms": round(self.p95(), 1),
            "cached_token_ratio": round(statistics.fmean(self.cached_ratio), 3) if self.cached_ratio else 0.0,
            "background_compute_s": round(self.background_ms / 1000.0, 2),
            "tpot_inflation_ms": round(self.tpot_inflation_ms, 1),
            "committed_intervals": self.committed_intervals,
            "stale_intervals": self.stale_intervals,
        }


def _default_engines() -> List[EngineConfig]:
    # The three tiers of our testbed with their fitted rates.
    return [
        EngineConfig("device-0", "llama-3.2-1b", "device", "http://device:8080", "llamacpp", tau_fg_ms_per_ktok=2560, tau_bg_ms_per_ktok=2330, kv_bytes_per_token=32768, tpot_ref_ms=48.0),
        EngineConfig("edge-0", "qwen2.5-7b", "edge", "http://edge:8000", "vllm", tau_fg_ms_per_ktok=732, tau_bg_ms_per_ktok=344, kv_bytes_per_token=57344, tpot_ref_ms=31.0),
        EngineConfig("cloud-0", "qwen2.5-14b", "cloud", "http://cloud:8000", "vllm", tau_fg_ms_per_ktok=191, tau_bg_ms_per_ktok=73, kv_bytes_per_token=98304, tpot_ref_ms=9.0),
    ]


def replay_trace(
    trace: Iterable[Mapping[str, object]],
    *,
    policy: Policy = Policy.TIDEMARK,
    engines: Optional[Sequence[EngineConfig]] = None,
    load: float = 0.6,
    iterations_between_turns: int = 40,
    seed: int = 7,
) -> ReplayReport:
    """Replay a route trace under one policy.

    Each trace row is ``{"session": s, "tenant": u, "turn": i, "model": m,
    "text": prompt, "p_next": {model: p}}``. Rows must be ordered by time.
    """
    rng = random.Random(seed)
    engines = list(engines or _default_engines())
    cfg = TidemarkConfig(engines=engines)
    if policy is Policy.FULL_PREFETCH:
        cfg.scheduler.delta_max = 1 << 20
    tokenizers = TokenizerRegistry()
    for i, e in enumerate(engines):
        tokenizers.register(e.model, _word_tokenizer(1.0 + 0.15 * i))
    router = StaticRouterSignal()
    rt = TidemarkRuntime(cfg, tokenizers, router=router)
    sims: Dict[str, ReplayEngine] = {}
    for e in engines:
        desc = EngineDescriptor(e.engine_id, e.model, e.runtime_config, e.tier, e.endpoint)
        sims[e.engine_id] = ReplayEngine(desc, CommitPath(e.engine_id, rt._on_result), rates=rt.rates.get(e.engine_id), rho=load, rng=rng, policy=policy)
        rt.adapters[e.engine_id] = sims[e.engine_id]
    if policy is Policy.APC:
        rt.scheduler.set_sink(lambda t: None)  # never dispatch background work

    report = ReplayReport(policy=policy.value, switches=0)
    last_model: Dict[str, str] = {}
    for row in trace:
        sid, tenant, model, text = str(row["session"]), str(row.get("tenant", "t0")), str(row["model"]), str(row["text"])
        if sid not in rt._histories:
            rt.open_session(sid, tenant)
        if row.get("p_next"):
            router.set(sid, dict(row["p_next"]))  # type: ignore[arg-type]
        hist = rt.history(sid)
        hist.append("user", text)
        engine_id = rt.scheduler.engine_for(model, "default")
        eng = sims[engine_id]
        ids = hist.token_ids(model)
        cached = eng.resident_prefix(sid, ids).tokens
        missing = len(ids) - cached
        is_switch = sid in last_model and last_model[sid] != model
        if is_switch:
            ttft = 35.0 + eng.rates.critical_path_ms(missing) + rng.uniform(0, 8)
            report.switches += 1
            report.switch_ttft_ms.append(ttft)
            report.cached_ratio.append(cached / len(ids) if ids else 0.0)
        last_model[sid] = model
        eng.resident[sid] = ids
        hist.append("assistant", " ".join(["reply"] * rng.randint(20, 80)))
        rt.turn_served(sid, model_id=model, resident_prefix=hist.length(model), engine_id=engine_id, is_switch=is_switch)
        for _ in range(iterations_between_turns):
            for e in sims.values():
                e.tick()
    for e in sims.values():
        report.background_ms += e.bg_ms
        report.tpot_inflation_ms += e.fg_tpot_inflation_ms
    report.committed_intervals = rt.scheduler.stats.committed
    report.stale_intervals = rt.scheduler.stats.stale
    return report


def load_trace(path: Union[str, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


__all__ = ["Policy", "ReplayEngine", "ReplayReport", "replay_trace", "load_trace"]
