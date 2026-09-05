"""Configuration.

Everything our evaluation fixes is a field here with the same default:
interval set ``{256, 512, 1024}``, foreground TPOT guard ``gamma = 0.03``,
predictor weight ``alpha = 1``, per-tenant caps ``kappa = 2`` and
``beta = 0.35``, memory weight ``lambda_M = 64 ms/GiB``, and ``delta_max = 1024``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from tidemark.scheduler.ranking import TenantCaps


@dataclass
class EngineConfig:
    engine_id: str
    model: str
    tier: str
    endpoint: str
    backend: str = "vllm"              # "vllm" | "llamacpp"
    runtime_config: str = "default"
    block_size: int = 16
    tau_fg_ms_per_ktok: Optional[float] = None
    tau_bg_ms_per_ktok: Optional[float] = None
    kv_bytes_per_token: Optional[int] = None
    tpot_ref_ms: Optional[float] = None


@dataclass
class SchedulerSection:
    delta_max: int = 1024
    alpha: float = 1.0
    lambda_mem_ms_per_gib: float = 64.0
    kappa: int = 2
    beta: float = 0.35
    ticket_lease_s: float = 30.0
    listen: str = "0.0.0.0:7420"

    def tenant_caps(self) -> TenantCaps:
        return TenantCaps(kappa=self.kappa, beta=self.beta)


@dataclass
class AdmissionSection:
    intervals: List[int] = field(default_factory=lambda: [256, 512, 1024])
    x_max: int = 1024
    kv_headroom: float = 0.08
    gamma: float = 0.03
    tpot_ewma_alpha: float = 0.2
    calibration_steps: int = 200


@dataclass
class TelemetrySection:
    directory: str = "./telemetry"
    request_log: bool = True
    step_log: bool = True
    ticket_log: bool = True


@dataclass
class TidemarkConfig:
    engines: List[EngineConfig] = field(default_factory=list)
    scheduler: SchedulerSection = field(default_factory=SchedulerSection)
    admission: AdmissionSection = field(default_factory=AdmissionSection)
    telemetry: TelemetrySection = field(default_factory=TelemetrySection)
    rates_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TidemarkConfig:
        engines = [EngineConfig(**e) for e in data.get("engines", [])]
        sched = SchedulerSection(**data.get("scheduler", {}))
        adm = AdmissionSection(**data.get("admission", {}))
        tel = TelemetrySection(**data.get("telemetry", {}))
        return cls(engines=engines, scheduler=sched, admission=adm, telemetry=tel, rates_file=data.get("rates_file"))


def load_config(path: Union[str, Path]) -> TidemarkConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        import json

        data = json.loads(text)
    cfg = TidemarkConfig.from_dict(data)
    if cfg.rates_file and not Path(cfg.rates_file).is_absolute():
        cfg.rates_file = str((path.parent / cfg.rates_file).resolve())
    return cfg


__all__ = ["EngineConfig", "SchedulerSection", "AdmissionSection", "TelemetrySection", "TidemarkConfig", "load_config"]
