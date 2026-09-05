"""HTTP front for the scheduler process.

A deliberately small JSON-over-HTTP surface, served from the standard library
so the scheduler has no framework dependency on the edge module it runs on.

    POST /v1/sessions                    {session, tenant}
    DELETE /v1/sessions/{id}
    POST /v1/sessions/{id}/turns         {role, text}
    POST /v1/sessions/{id}/served        {model, runtime_config, resident_prefix, engine, ...}
    POST /v1/sessions/{id}/rewrite       {turn_index, text}
    POST /v1/engines/{id}/transition     {safe_budget_tokens}
    POST /v1/engines/{id}/eviction       {session, resident_prefix}
    POST /v1/tickets/result              TicketResult as JSON (from engine shims)
    GET  /v1/status

Engines report ticket results here; the scheduler pushes tickets to engines
through their adapters. Every update is idempotent, which is what allows a
retry after a network or engine failure to be harmless.
"""

from __future__ import annotations

import json
import logging
import signal
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from tidemark.catalog.history import TokenizerRegistry
from tidemark.runtime.config import TidemarkConfig
from tidemark.runtime.service import TidemarkRuntime
from tidemark.runtime.telemetry import Telemetry
from tidemark.scheduler.ticket import TicketResult, TicketStatus

log = logging.getLogger("tidemark.server")


def _default_tokenizers(cfg: TidemarkConfig) -> TokenizerRegistry:
    """Load a HF tokenizer per model when available; otherwise a byte tokenizer.

    Production deployments register the engine's own tokenizer here so token
    offsets line up exactly with what the engine hashes.
    """
    reg = TokenizerRegistry()
    for e in cfg.engines:
        reg.register(e.model, _tokenizer_for(e.model))
    return reg


def _tokenizer_for(model: str):
    try:
        from transformers import AutoTokenizer  # type: ignore

        hf = AutoTokenizer.from_pretrained(model)

        def hf_tokenize(text: str):
            return hf(text, add_special_tokens=False)["input_ids"]

        return hf_tokenize
    except Exception:
        log.warning("no HF tokenizer for %s; falling back to bytes (offsets will not match the engine)", model)

        def byte_tokenize(text: str):
            return list(text.encode("utf-8"))

        return byte_tokenize


def _adapter_factory(cfg: TidemarkConfig):
    def make(desc, commit):
        spec = next(e for e in cfg.engines if e.engine_id == desc.engine_id)
        if spec.backend == "llamacpp":
            from tidemark.engines.llamacpp.adapter import LlamaCppAdapter

            return LlamaCppAdapter(desc, commit)
        from tidemark.engines.vllm.client import TokenIdClient
        from tidemark.engines.vllm.shim import VllmEngineAdapter

        return VllmEngineAdapter(desc, commit, TokenIdClient(desc.endpoint, desc.model_id))

    return make


class _Handler(BaseHTTPRequestHandler):
    runtime: TidemarkRuntime  # set by serve()

    def _json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _send(self, code: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        log.debug(fmt, *args)

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/v1/status":
            self._send(200, self.runtime.status())
        else:
            self._send(404, {"error": "not found"})

    def do_DELETE(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["v1", "sessions"]:
            self.runtime.close_session(parts[2])
            self._send(200, {"ok": True})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        body = self._json()
        rt = self.runtime
        try:
            if parts == ["v1", "sessions"]:
                rt.open_session(body["session"], body.get("tenant", "default"))
                return self._send(201, {"ok": True})
            if len(parts) == 4 and parts[1] == "sessions":
                sid, action = parts[2], parts[3]
                if action == "turns":
                    idx = rt.append_turn(sid, body.get("role", "user"), body["text"])
                    return self._send(200, {"turn_index": idx})
                if action == "served":
                    rt.turn_served(
                        sid,
                        model_id=body["model"],
                        runtime_config=body.get("runtime_config", "default"),
                        resident_prefix=int(body["resident_prefix"]),
                        engine_id=body.get("engine"),
                        cached_tokens=body.get("cached_tokens"),
                        uncached_tokens=body.get("uncached_tokens"),
                        ttft_ms=body.get("ttft_ms"),
                        is_switch=body.get("switch"),
                    )
                    return self._send(200, {"ok": True})
                if action == "rewrite":
                    rt.rewrite_turn(sid, int(body["turn_index"]), body["text"])
                    return self._send(200, {"ok": True})
            if len(parts) == 4 and parts[1] == "engines":
                eid, action = parts[2], parts[3]
                if action == "transition":
                    rt.engine_transition(eid, int(body.get("safe_budget_tokens", 0)))
                    return self._send(200, {"ok": True})
                if action == "eviction":
                    rt.eviction(eid, body["session"], int(body["resident_prefix"]))
                    return self._send(200, {"ok": True})
            if parts == ["v1", "tickets", "result"]:
                rt._on_result(
                    TicketResult(
                        ticket_id=body["ticket_id"],
                        engine_id=body["engine_id"],
                        status=TicketStatus(body["status"]),
                        admitted_delta=int(body.get("admitted_delta", 0)),
                        cached_tokens=int(body.get("cached_tokens", 0)),
                        prefilled_tokens=int(body.get("prefilled_tokens", 0)),
                        snapshot_hash=str(body.get("snapshot_hash", "")),
                        gpu_ms=float(body.get("gpu_ms", 0.0)),
                        kv_bytes=int(body.get("kv_bytes", 0)),
                        reason=str(body.get("reason", "")),
                    )
                )
                return self._send(200, {"ok": True})
            self._send(404, {"error": "not found"})
        except KeyError as exc:
            self._send(400, {"error": f"missing field {exc}"})
        except Exception as exc:  # keep the scheduler up; log and report
            log.exception("request failed")
            self._send(500, {"error": str(exc)})


def serve(cfg: TidemarkConfig, *, listen: str) -> int:
    host, _, port = listen.rpartition(":")
    telemetry = Telemetry(cfg.telemetry.directory, request=cfg.telemetry.request_log, step=cfg.telemetry.step_log, ticket=cfg.telemetry.ticket_log)
    runtime = TidemarkRuntime(cfg, _default_tokenizers(cfg), adapter_factory=_adapter_factory(cfg), telemetry=telemetry)
    _Handler.runtime = runtime
    httpd = ThreadingHTTPServer((host or "0.0.0.0", int(port)), _Handler)
    stop = threading.Event()

    def _sig(*_: Any) -> None:
        stop.set()
        httpd.shutdown()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)
    log.info("tidemark scheduler listening on %s with %d engines", listen, len(cfg.engines))
    with runtime:
        httpd.serve_forever()
    return 0


__all__ = ["serve"]
