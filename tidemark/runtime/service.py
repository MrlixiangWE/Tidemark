"""The Tidemark runtime.

``TidemarkRuntime`` owns one catalog, one global scheduler, and one adapter per
engine. The application calls :meth:`turn_served` after every foreground
request; everything else happens in the background. Losing the runtime stops
ticket issue and leaves every engine serving foreground work with its stock
scheduler, which is the failure mode we want.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

from tidemark.admission.commit import CommitPath
from tidemark.catalog.frontier import FrontierKey, VersionedFrontierCatalog
from tidemark.catalog.history import SessionHistory, TokenizerRegistry
from tidemark.engines.base import EngineAdapter, EngineDescriptor
from tidemark.runtime.config import TidemarkConfig
from tidemark.runtime.telemetry import Telemetry
from tidemark.scheduler.cost_model import EngineRates, RateTable
from tidemark.scheduler.global_scheduler import EngineSpec, GlobalFrontierScheduler, SchedulerConfig
from tidemark.scheduler.predictor import DestinationPredictor, RouterSignal
from tidemark.scheduler.ticket import AtomicTicket, TicketResult

log = logging.getLogger("tidemark.runtime")

AdapterFactory = Callable[[EngineDescriptor, CommitPath], EngineAdapter]


class TidemarkRuntime:
    def __init__(
        self,
        config: TidemarkConfig,
        tokenizers: TokenizerRegistry,
        *,
        router: Optional[RouterSignal] = None,
        adapter_factory: Optional[AdapterFactory] = None,
        rates: Optional[RateTable] = None,
        telemetry: Optional[Telemetry] = None,
    ) -> None:
        self.config = config
        self.tokenizers = tokenizers
        self.telemetry = telemetry
        self.catalog = VersionedFrontierCatalog()
        self.rates = rates or self._rates_from_config(config)
        models = sorted({e.model for e in config.engines})
        self.predictor = DestinationPredictor(models=models, router=router, alpha=config.scheduler.alpha)
        specs = [EngineSpec(e.engine_id, e.model, e.runtime_config, e.tier) for e in config.engines]
        self.scheduler = GlobalFrontierScheduler(
            self.catalog,
            self.rates,
            self.predictor,
            specs,
            SchedulerConfig(
                delta_max=config.scheduler.delta_max,
                lambda_mem_ms_per_gib=config.scheduler.lambda_mem_ms_per_gib,
                alpha=config.scheduler.alpha,
                tenant_caps=config.scheduler.tenant_caps(),
                ticket_lease_s=config.scheduler.ticket_lease_s,
            ),
            sink=self._dispatch,
        )
        self.adapters: Dict[str, EngineAdapter] = {}
        if adapter_factory is not None:
            for e in config.engines:
                desc = EngineDescriptor(e.engine_id, e.model, e.runtime_config, e.tier, e.endpoint, e.block_size)
                self.adapters[e.engine_id] = adapter_factory(desc, CommitPath(e.engine_id, self._on_result))
        self._histories: Dict[str, SessionHistory] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._lease_thread: Optional[threading.Thread] = None
        if telemetry is not None:
            self.catalog.subscribe(self._log_catalog_event)

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _rates_from_config(config: TidemarkConfig) -> RateTable:
        if config.rates_file:
            return RateTable.load(config.rates_file)
        table = RateTable()
        for e in config.engines:
            if e.tau_fg_ms_per_ktok is None or e.tau_bg_ms_per_ktok is None or e.kv_bytes_per_token is None:
                raise ValueError(f"engine {e.engine_id}: provide rates inline or through rates_file")
            table.put(
                EngineRates(
                    engine_id=e.engine_id,
                    model_id=e.model,
                    tier=e.tier,
                    tau_fg_ms_per_token=e.tau_fg_ms_per_ktok / 1000.0,
                    tau_bg_ms_per_token=e.tau_bg_ms_per_ktok / 1000.0,
                    kv_bytes_per_token=e.kv_bytes_per_token,
                    tpot_ref_ms=e.tpot_ref_ms,
                )
            )
        return table

    # ------------------------------------------------------------ sessions
    def open_session(self, session_id: str, tenant_id: str) -> SessionHistory:
        with self._lock:
            hist = SessionHistory(session_id, tenant_id, self.tokenizers)
            self._histories[session_id] = hist
            self.scheduler.on_session_start(hist)
            return hist

    def close_session(self, session_id: str) -> None:
        with self._lock:
            self._histories.pop(session_id, None)
            self.scheduler.on_session_end(session_id)

    def history(self, session_id: str) -> SessionHistory:
        return self._histories[session_id]

    # --------------------------------------------------------------- turns
    def append_turn(self, session_id: str, role: str, text: str) -> int:
        return self._histories[session_id].append(role, text)

    def rewrite_turn(self, session_id: str, turn_index: int, text: str) -> None:
        lcp = self._histories[session_id].rewrite(turn_index, text)
        self.scheduler.on_revision_transition(session_id, lcp)

    def turn_served(
        self,
        session_id: str,
        *,
        model_id: str,
        runtime_config: str = "default",
        resident_prefix: int,
        engine_id: Optional[str] = None,
        cached_tokens: Optional[int] = None,
        uncached_tokens: Optional[int] = None,
        ttft_ms: Optional[float] = None,
        is_switch: Optional[bool] = None,
    ) -> None:
        """Report a finished foreground request and trigger a ranking epoch."""
        outcome = self.scheduler.on_turn_served(
            session_id,
            served_model=model_id,
            runtime_config=runtime_config,
            resident_prefix=resident_prefix,
            engine_id=engine_id,
        )
        if self.telemetry is not None:
            self.telemetry.requests.write(
                {
                    "session": session_id,
                    "model": model_id,
                    "engine": engine_id or self.scheduler.engine_for(model_id, runtime_config),
                    "cached_tokens": cached_tokens,
                    "uncached_tokens": uncached_tokens,
                    "ttft_ms": ttft_ms,
                    "switch": is_switch,
                    "epoch": outcome.epoch,
                    "issued": len(outcome.issued),
                }
            )

    # --------------------------------------------------------------- engine
    def engine_transition(self, engine_id: str, safe_budget_tokens: int) -> None:
        self.scheduler.on_engine_transition(engine_id, safe_budget_tokens)

    def eviction(self, engine_id: str, session_id: str, resident_prefix: int) -> None:
        spec = next(a for a in self.config.engines if a.engine_id == engine_id)
        self.scheduler.on_eviction(FrontierKey(session_id, spec.model, spec.runtime_config), resident_prefix)

    def _dispatch(self, ticket: AtomicTicket) -> None:
        if self.telemetry is not None:
            self.telemetry.tickets.write({"event": "issued", **ticket.metadata(), "score": ticket.score})
        adapter = self.adapters.get(ticket.engine_id)
        if adapter is None:
            return
        adapter.submit(ticket, ticket.delta_max)

    def _on_result(self, result: TicketResult) -> None:
        if self.telemetry is not None and result.terminal:
            self.telemetry.tickets.write(
                {
                    "event": result.status.value,
                    "ticket_id": result.ticket_id,
                    "engine": result.engine_id,
                    "delta": result.admitted_delta,
                    "cached": result.cached_tokens,
                    "gpu_ms": round(result.gpu_ms, 3),
                    "reason": result.reason,
                }
            )
        self.scheduler.on_ticket_result(result)

    def _log_catalog_event(self, event: str, entry) -> None:
        if self.telemetry is not None:
            self.telemetry.tickets.write(
                {
                    "event": f"catalog_{event}",
                    "session": entry.key.session_id,
                    "model": entry.key.model_id,
                    "frontier": entry.frontier,
                    "generation": entry.generation,
                    "placement": entry.placement.value,
                }
            )

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._lease_thread is not None:
            return

        def loop() -> None:
            while not self._stop.wait(1.0):
                n = self.scheduler.expire_leases(time.monotonic())
                if n:
                    log.debug("expired %d ticket leases", n)

        self._lease_thread = threading.Thread(target=loop, name="tidemark-leases", daemon=True)
        self._lease_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._lease_thread is not None:
            self._lease_thread.join(timeout=2.0)
            self._lease_thread = None
        if self.telemetry is not None:
            self.telemetry.close()

    def __enter__(self) -> TidemarkRuntime:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # --------------------------------------------------------------- status
    def status(self) -> Dict[str, object]:
        return {
            "sessions": len(self._histories),
            "catalog_entries": sum(1 for _ in self.catalog.entries()),
            "catalog_bytes": self.catalog.size_bytes(),
            "inflight": [t.ticket_id for t in self.scheduler.inflight()],
            "scheduler": self.scheduler.stats.as_dict(),
            "rates": {e: self.rates.get(e).to_dict() for e in self.rates.engines()},
        }


__all__ = ["TidemarkRuntime", "AdapterFactory"]
