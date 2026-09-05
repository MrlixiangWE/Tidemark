"""Tidemark: model switching with versioned KV frontiers.

Tidemark is a runtime layer that sits between an application-level model
router and the inference engines of a device-edge-cloud deployment. It keeps
the KV state of the models that are *not* serving a session close to the
conversation, so that a later switch to one of them finds most of the prefix
already resident.

The package is organised around three components:

* :mod:`tidemark.catalog` -- the versioned KV frontier catalog, which records
  for every ``(session, model, runtime_config)`` how far a reusable prefix has
  reached and which history revision it is valid against.
* :mod:`tidemark.scheduler` -- the global frontier scheduler, which ranks
  bounded frontier advances across tenants and tiers by the switch latency
  they remove per unit of background compute time and issues at most one
  atomic ticket per engine.
* :mod:`tidemark.admission` -- the engine-local admission and commit path,
  which serves foreground work first, fits at most one interval into the safe
  budget of the current scheduler iteration, and validates the result before
  the catalog advances.
"""

from tidemark.admission import AdmissionMode, EngineLocalAdmission, safe_budget
from tidemark.catalog import FrontierEntry, FrontierKey, SessionHistory, VersionedFrontierCatalog
from tidemark.runtime import TidemarkConfig, TidemarkRuntime
from tidemark.scheduler import AtomicTicket, GlobalFrontierScheduler, TicketResult, TicketStatus
from tidemark.version import __version__

__all__ = [
    "__version__",
    "FrontierKey",
    "FrontierEntry",
    "VersionedFrontierCatalog",
    "SessionHistory",
    "AtomicTicket",
    "TicketResult",
    "TicketStatus",
    "GlobalFrontierScheduler",
    "EngineLocalAdmission",
    "AdmissionMode",
    "safe_budget",
    "TidemarkRuntime",
    "TidemarkConfig",
]
