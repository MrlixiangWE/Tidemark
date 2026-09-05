"""Engine-local admission and commit.

Each scheduling iteration first admits foreground decode and foreground
prefill, so that background work never delays them, and then fits at most one
interval into the budget they leave. The pieces are:

* :func:`safe_budget` -- ``X_t`` (Equation 7): how many background tokens fit
  after foreground work while preserving KV headroom.
* :class:`TpotGuard` and :func:`classify_mode` -- the three-mode decision
  function (Equation 8): ``Idle``, ``Mixed`` or ``Blocked``.
* :class:`EngineLocalAdmission` -- picks the admitted size from the fixed
  interval set, one interval per iteration, and cancels it at the next
  scheduler boundary if a foreground request lands.
* :class:`CommitPath` -- runs the validity predicate against a catalog
  snapshot and reports the terminal ticket state back to the scheduler.

Everything here is independent of any particular engine so the same source can
be vendored into a vLLM patch or driven by a llama.cpp adapter and still be
unit-tested on a laptop.
"""

from tidemark.admission.commit import CommitPath
from tidemark.admission.controller import (
    INTERVAL_SET,
    AdmissionDecision,
    EngineLocalAdmission,
    StepState,
)
from tidemark.admission.guard import AdmissionMode, TpotGuard, classify_mode
from tidemark.admission.safe_budget import SafeBudgetInputs, safe_budget

__all__ = [
    "SafeBudgetInputs",
    "safe_budget",
    "AdmissionMode",
    "TpotGuard",
    "classify_mode",
    "INTERVAL_SET",
    "AdmissionDecision",
    "EngineLocalAdmission",
    "StepState",
    "CommitPath",
]
