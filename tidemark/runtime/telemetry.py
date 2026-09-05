"""Request, scheduler-step and ticket logs.

Three JSONL streams separate predicted progress from physically reusable state:
cached versus uncached prompt tokens per request, the pressure reason at every
scheduler iteration, and a terminal state for every selected ticket. The
evaluation scripts consume nothing else.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Union


class JsonlLog:
    def __init__(self, path: Union[str, Path], enabled: bool = True) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()
        self._fh = None
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", buffering=1)

    def write(self, row: Mapping[str, Any]) -> None:
        if not self.enabled or self._fh is None:
            return
        record = {"ts": time.time(), **row}
        line = json.dumps(record, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class Telemetry:
    def __init__(self, directory: Union[str, Path], *, request: bool = True, step: bool = True, ticket: bool = True) -> None:
        d = Path(directory)
        self.requests = JsonlLog(d / "requests.jsonl", request)
        self.steps = JsonlLog(d / "scheduler_steps.jsonl", step)
        self.tickets = JsonlLog(d / "tickets.jsonl", ticket)

    def close(self) -> None:
        for log in (self.requests, self.steps, self.tickets):
            log.close()


__all__ = ["JsonlLog", "Telemetry"]
