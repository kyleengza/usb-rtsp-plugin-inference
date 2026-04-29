"""Worker-to-panel event channel.

Each worker writes JSON-line events to a per-job file under
~/.cache/usb-rtsp/inference-events/<job>.jsonl. The plugin's API
process tails the file (last-N lines) on each /events poll.

Why not a unix socket: a file is simpler (no dual-listener wiring on
the FastAPI side, no race during worker restart), and "last 500 lines"
is exactly what the UI wants. Worker truncation keeps the file
bounded.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

EVENTS_DIR = Path.home() / ".cache" / "usb-rtsp" / "inference-events"
MAX_LINES = 500


class JobEventLog:
    """Append-only per-job event log with rolling truncation. Worker side."""

    def __init__(self, job_name: str, max_lines: int = MAX_LINES) -> None:
        self.path = EVENTS_DIR / f"{job_name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.max_lines = max_lines
        self._line_count = 0
        if self.path.exists():
            try:
                with self.path.open("rb") as f:
                    self._line_count = sum(1 for _ in f)
            except OSError:
                self._line_count = 0

    def emit(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, separators=(",", ":")) + "\n"
        try:
            with self.path.open("a") as f:
                f.write(line)
        except OSError:
            return
        self._line_count += 1
        if self._line_count > self.max_lines * 2:
            self._truncate()

    def _truncate(self) -> None:
        try:
            with self.path.open("rb") as f:
                lines = f.readlines()
            keep = lines[-self.max_lines:]
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp.open("wb") as f:
                f.writelines(keep)
            tmp.replace(self.path)
            self._line_count = len(keep)
        except OSError:
            pass


def read_recent(job_name: str, n: int = 100) -> list[dict[str, Any]]:
    """Read the most recent ``n`` events for ``job_name``. Reader side
    (FastAPI). Returns oldest-first."""
    path = EVENTS_DIR / f"{job_name}.jsonl"
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            lines = f.readlines()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in lines[-n:]:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out
