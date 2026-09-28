"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record an input event and remember when the request started."""
        key = str(request_id or user_id)
        self._open[key] = time.perf_counter()
        self.logs.append(
            {
                "event": "input",
                "request_id": request_id,
                "user_id": user_id,
                "input": text,
                "timestamp": utc_now_iso(),
            }
        )

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Record an output event, its decision, and elapsed request time."""
        key = str(request_id or user_id)
        started = self._open.pop(key, None)
        latency_ms = (
            round((time.perf_counter() - started) * 1000, 3)
            if started is not None
            else None
        )
        self.logs.append(
            {
                "event": "output",
                "request_id": request_id,
                "user_id": user_id,
                "output": text,
                "blocked": blocked,
                "layer": layer,
                "latency_ms": latency_ms,
                "timestamp": utc_now_iso(),
            }
        )

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as output_file:
            json.dump(self.logs, output_file, indent=2, ensure_ascii=False)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
