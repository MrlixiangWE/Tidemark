"""The decode-TPOT guard and the three-mode decision function (Equation 8).

                 | Idle,     D_t = P_t = 0
    Mode(t) =    | Mixed,    D_t + P_t > 0  and  X_t > 0  and  ok_t
                 | Blocked,  otherwise

``ok_t`` holds while the exponentially weighted moving average of the engine's
foreground TPOT stays within a configured multiple of a reference value:

    TPOT_t^ewma <= (1 + gamma) * TPOT_ref

``TPOT_ref`` is the median of a run of foreground-only calibration steps taken
when the engine starts (200 by default), and ``gamma`` is 0.03 in the paper.
"""

from __future__ import annotations

import enum
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


class AdmissionMode(str, enum.Enum):
    IDLE = "idle"
    MIXED = "mixed"
    BLOCKED = "blocked"


@dataclass
class TpotGuard:
    gamma: float = 0.03
    ewma_alpha: float = 0.2
    calibration_steps: int = 200
    tpot_ref_ms: Optional[float] = None
    ewma_ms: Optional[float] = None
    _calib: List[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must lie in (0, 1]")

    @property
    def calibrated(self) -> bool:
        return self.tpot_ref_ms is not None

    @property
    def threshold_ms(self) -> Optional[float]:
        return None if self.tpot_ref_ms is None else (1.0 + self.gamma) * self.tpot_ref_ms

    def calibrate(self, samples: Sequence[float]) -> float:
        """Fix ``TPOT_ref`` from foreground-only steps."""
        if not samples:
            raise ValueError("need at least one calibration sample")
        self.tpot_ref_ms = float(statistics.median(samples))
        return self.tpot_ref_ms

    def observe(self, tpot_ms: float) -> None:
        """Feed one foreground decode step. Calibrates itself for the first
        ``calibration_steps`` observations if no reference was supplied."""
        if tpot_ms <= 0:
            return
        if self.tpot_ref_ms is None:
            self._calib.append(float(tpot_ms))
            if len(self._calib) >= self.calibration_steps:
                self.calibrate(self._calib)
                self._calib.clear()
        if self.ewma_ms is None:
            self.ewma_ms = float(tpot_ms)
        else:
            self.ewma_ms = self.ewma_alpha * float(tpot_ms) + (1.0 - self.ewma_alpha) * self.ewma_ms

    @property
    def ok(self) -> bool:
        """``ok_t``. Conservative before calibration: no reference, no admission
        while foreground work is present."""
        if self.tpot_ref_ms is None:
            return False
        if self.ewma_ms is None:
            return True
        return self.ewma_ms <= self.threshold_ms  # type: ignore[operator]

    def headroom(self) -> Optional[float]:
        if self.threshold_ms is None or self.ewma_ms is None:
            return None
        return self.threshold_ms - self.ewma_ms


def classify_mode(*, decode_tokens: int, prefill_tokens: int, safe_budget_tokens: int, guard_ok: bool) -> AdmissionMode:
    foreground = decode_tokens + prefill_tokens
    if foreground == 0:
        return AdmissionMode.IDLE
    if safe_budget_tokens > 0 and guard_ok:
        return AdmissionMode.MIXED
    return AdmissionMode.BLOCKED


__all__ = ["AdmissionMode", "TpotGuard", "classify_mode"]
