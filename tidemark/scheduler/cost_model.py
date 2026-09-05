"""Per-engine prefill rate model.

Switch TTFT grows approximately linearly with the missing suffix on every tier,
so we fit two rates per engine and keep them apart because they measure
different things:

``tau_fg``
    foreground wall time per uncached token on the critical path of a switch.
    ``C_m(x) = tau_fg * x`` estimates the prefill time a switch with ``x``
    missing tokens would pay.
``tau_bg``
    incremental scheduled compute time a background interval adds when it is
    batched alongside foreground decode. ``T_m(delta) = tau_bg * delta``.

The ratio ``tau_bg / tau_fg`` is well below one on a datacenter GPU, where a
background interval fills capacity a decode-heavy batch leaves unused, and
close to one on a device engine that has little spare width. Neither rate is
monotone in model size, so they are measured per engine rather than inferred
from the model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union


def fit_rate(samples: Iterable[Tuple[int, float]]) -> float:
    """Least-squares slope through the origin for ``(tokens, milliseconds)`` pairs."""
    num = 0.0
    den = 0.0
    for tokens, ms in samples:
        num += float(tokens) * float(ms)
        den += float(tokens) ** 2
    if den == 0.0:
        raise ValueError("need at least one sample with tokens > 0")
    return num / den


@dataclass
class EngineRates:
    engine_id: str
    model_id: str
    tier: str
    tau_fg_ms_per_token: float
    tau_bg_ms_per_token: float
    kv_bytes_per_token: int
    tpot_ref_ms: Optional[float] = None
    samples_fg: int = 0
    samples_bg: int = 0

    def __post_init__(self) -> None:
        if self.tau_fg_ms_per_token <= 0 or self.tau_bg_ms_per_token <= 0:
            raise ValueError("prefill rates must be positive")
        if self.kv_bytes_per_token <= 0:
            raise ValueError("kv_bytes_per_token must be positive")

    # C_m(x): critical-path prefill time for an uncached suffix of x tokens.
    def critical_path_ms(self, missing_tokens: int) -> float:
        return self.tau_fg_ms_per_token * max(0, missing_tokens)

    # T_m(delta): background compute time an interval adds.
    def background_ms(self, delta: int) -> float:
        return self.tau_bg_ms_per_token * max(0, delta)

    # M_m(delta): KV bytes an interval adds.
    def kv_bytes(self, delta: int) -> int:
        return self.kv_bytes_per_token * max(0, delta)

    @property
    def bg_fg_ratio(self) -> float:
        return self.tau_bg_ms_per_token / self.tau_fg_ms_per_token

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class RateTable:
    """Fitted rates for every engine in the deployment."""

    def __init__(self, rates: Iterable[EngineRates] = ()) -> None:
        self._by_engine: Dict[str, EngineRates] = {}
        for r in rates:
            self.put(r)

    def put(self, rates: EngineRates) -> None:
        self._by_engine[rates.engine_id] = rates

    def get(self, engine_id: str) -> EngineRates:
        try:
            return self._by_engine[engine_id]
        except KeyError:
            raise KeyError(f"no fitted rates for engine {engine_id!r}; run scripts/calibrate_rates.py") from None

    def __contains__(self, engine_id: object) -> bool:
        return engine_id in self._by_engine

    def engines(self) -> Tuple[str, ...]:
        return tuple(self._by_engine)

    # ------------------------------------------------------------- updates
    def observe_background(self, engine_id: str, delta: int, gpu_ms: float, ewma: float = 0.1) -> None:
        """Refine ``tau_bg`` online from a committed interval's measured compute time."""
        if delta <= 0 or gpu_ms <= 0:
            return
        r = self._by_engine[engine_id]
        sample = gpu_ms / delta
        r.tau_bg_ms_per_token = (1.0 - ewma) * r.tau_bg_ms_per_token + ewma * sample
        r.samples_bg += 1

    def observe_foreground(self, engine_id: str, uncached_tokens: int, prefill_ms: float, ewma: float = 0.1) -> None:
        if uncached_tokens <= 0 or prefill_ms <= 0:
            return
        r = self._by_engine[engine_id]
        sample = prefill_ms / uncached_tokens
        r.tau_fg_ms_per_token = (1.0 - ewma) * r.tau_fg_ms_per_token + ewma * sample
        r.samples_fg += 1

    # ---------------------------------------------------------------- I/O
    @classmethod
    def load(cls, path: Union[str, Path]) -> RateTable:
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml  # optional dependency, only needed for YAML configs

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        engines = data["engines"] if isinstance(data, dict) and "engines" in data else data
        table = cls()
        for engine_id, spec in engines.items():
            table.put(
                EngineRates(
                    engine_id=engine_id,
                    model_id=spec["model"],
                    tier=spec.get("tier", "unknown"),
                    tau_fg_ms_per_token=float(spec["tau_fg_ms_per_ktok"]) / 1000.0,
                    tau_bg_ms_per_token=float(spec["tau_bg_ms_per_ktok"]) / 1000.0,
                    kv_bytes_per_token=int(spec["kv_bytes_per_token"]),
                    tpot_ref_ms=spec.get("tpot_ref_ms"),
                )
            )
        return table

    def dump(self, path: Union[str, Path]) -> None:
        payload = {
            "engines": {
                e: {
                    "model": r.model_id,
                    "tier": r.tier,
                    "tau_fg_ms_per_ktok": round(r.tau_fg_ms_per_token * 1000.0, 3),
                    "tau_bg_ms_per_ktok": round(r.tau_bg_ms_per_token * 1000.0, 3),
                    "kv_bytes_per_token": r.kv_bytes_per_token,
                    "tpot_ref_ms": r.tpot_ref_ms,
                }
                for e, r in self._by_engine.items()
            }
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


__all__ = ["EngineRates", "RateTable", "fit_rate"]
