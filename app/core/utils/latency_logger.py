"""
Latency Logger - Tracks timing throughout request processing.

Writes detailed timing data to a file for latency analysis.
Also persists request traces to PostgreSQL for the trace visualizer.
"""

import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

logger = logging.getLogger("uvicorn")


@dataclass
class TimingEntry:
    """A single timing measurement (span)."""
    label: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    service: str = "cc"
    status: str = "ok"
    metadata: Optional[dict[str, Any]] = None


@dataclass
class RequestTiming:
    """Timing data for a single request."""
    request_id: str
    request_type: str  # "warmup", "voice_command", "voice_command_stream", "mobile_chat"
    start_time: float = field(default_factory=time.perf_counter)
    entries: list[TimingEntry] = field(default_factory=list)
    _current_entry: Optional[TimingEntry] = field(default=None, repr=False)

    # Trace metadata — set by the endpoint that creates the timing.
    source: str = "unknown"            # "node" or "mobile"
    node_id: Optional[str] = None
    household_id: Optional[str] = None
    user_command: Optional[str] = None
    trace_status: str = "ok"           # overall trace status
    error_message: Optional[str] = None

    def checkpoint(self, label: str, service: str = "cc") -> None:
        """Record a checkpoint with elapsed time from request start."""
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.entries.append(TimingEntry(
            label=label,
            start_time=elapsed,
            end_time=elapsed,
            duration_ms=0,
            service=service,
        ))

    @contextmanager
    def measure(self, label: str, service: str = "cc", metadata: dict[str, Any] | None = None):
        """Context manager to measure a block of code.

        Args:
            label: Name for this span (e.g. "llm_call_iter_1").
            service: Which service this span represents ("cc", "llm_proxy", "tts", "node", "whisper").
            metadata: Arbitrary key-value pairs (tokens, tool_name, error_msg, etc.).
        """
        entry = TimingEntry(
            label=label,
            start_time=0,
            service=service,
            metadata=metadata,
        )
        start = time.perf_counter()
        try:
            yield entry
        except Exception:
            entry.status = "error"
            raise
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            elapsed_start = (start - self.start_time) * 1000
            elapsed_end = (end - self.start_time) * 1000
            entry.start_time = elapsed_start
            entry.end_time = elapsed_end
            entry.duration_ms = duration_ms
            self.entries.append(entry)

    def total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000

    def to_spans(self) -> list[dict[str, Any]]:
        """Export timing entries as a list of span dicts for storage."""
        return [
            {
                "name": e.label,
                "service": e.service,
                "start_ms": round(e.start_time, 1),
                "end_ms": round(e.end_time, 1) if e.end_time is not None else round(e.start_time, 1),
                "duration_ms": round(e.duration_ms, 1) if e.duration_ms else 0,
                "status": e.status,
                "metadata": e.metadata or {},
            }
            for e in sorted(self.entries, key=lambda x: x.start_time)
        ]

    def to_trace_summary(self) -> dict[str, Any]:
        """Build a service-level trace summary for SSE delivery.

        Groups leaf spans into sequential *service hops* — consecutive
        spans on the same service are merged into one entry.  The result
        reads like a request flow: CC → LLM Proxy → CC → Node → …
        """
        spans = self.to_spans()
        nonzero = [s for s in spans if s["duration_ms"] > 0]

        # Filter to leaf spans: exclude any span that fully contains another.
        leaf_spans: list[dict[str, Any]] = []
        for s in nonzero:
            is_parent = any(
                other is not s
                and s["start_ms"] <= other["start_ms"]
                and s["end_ms"] >= other["end_ms"]
                and other["duration_ms"] > 0
                for other in nonzero
            )
            if not is_parent:
                leaf_spans.append(s)

        # Sort chronologically and merge consecutive same-service spans
        # into service hops.
        leaf_spans.sort(key=lambda s: s["start_ms"])
        service_hops: list[dict[str, Any]] = []
        for span in leaf_spans:
            svc = span["service"]
            if service_hops and service_hops[-1]["service"] == svc:
                hop = service_hops[-1]
                hop["duration_ms"] = round(hop["duration_ms"] + span["duration_ms"], 1)
                hop["steps"].append(span["name"])
                if span["status"] == "error":
                    hop["status"] = "error"
            else:
                service_hops.append({
                    "service": svc,
                    "duration_ms": span["duration_ms"],
                    "status": span["status"],
                    "steps": [span["name"]],
                })

        return {
            "total_duration_ms": round(self.total_ms(), 1),
            "span_count": len(spans),
            "status": self.trace_status,
            "service_hops": service_hops,
        }


class LatencyLogger:
    """Thread-safe latency logger that writes to a file and persists traces."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            # Default to temp/latency.log in repo root
            repo_root = Path(__file__).resolve().parents[3]
            log_path = repo_root / "temp" / "latency.log"
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._requests: dict[str, RequestTiming] = {}

    def start_request(self, request_id: str, request_type: str) -> RequestTiming:
        """Start tracking a new request."""
        timing = RequestTiming(request_id=request_id, request_type=request_type)
        self._requests[request_id] = timing
        return timing

    def get_request(self, request_id: str) -> Optional[RequestTiming]:
        """Get timing data for an existing request."""
        return self._requests.get(request_id)

    def end_request(self, request_id: str) -> None:
        """End tracking, write timing data to file, and persist trace to DB."""
        timing = self._requests.pop(request_id, None)
        if timing is None:
            return

        total_ms = timing.total_ms()

        # Build log output
        lines = [
            f"\n{'='*60}",
            f"[{timing.request_type.upper()}] {timing.request_id[:8]}  Total: {total_ms:.1f}ms",
            f"{'='*60}",
        ]

        # Sort entries by start time
        sorted_entries = sorted(timing.entries, key=lambda e: e.start_time)

        for entry in sorted_entries:
            if entry.duration_ms and entry.duration_ms > 0:
                lines.append(
                    f"  [{entry.start_time:7.1f}ms] {entry.label}: {entry.duration_ms:.1f}ms"
                )
            else:
                lines.append(
                    f"  [{entry.start_time:7.1f}ms] {entry.label}"
                )

        lines.append(f"{'='*60}\n")

        # Write to file
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write("\n".join(lines) + "\n")

        # Also log summary to console
        top_entries = sorted(
            [e for e in timing.entries if e.duration_ms and e.duration_ms > 10],
            key=lambda e: e.duration_ms or 0,
            reverse=True
        )[:5]
        if top_entries:
            summary_parts = [f"{e.label}={e.duration_ms:.0f}ms" for e in top_entries]
            logger.info(
                f"⏱️  [{timing.request_type.upper()}] {timing.request_id[:8]} "
                f"Total={total_ms:.0f}ms | {' | '.join(summary_parts)}"
            )

        # Persist trace to PostgreSQL (fire-and-forget)
        if timing.source != "unknown":
            self._persist_trace(timing, total_ms)

    def _persist_trace(self, timing: RequestTiming, total_ms: float) -> None:
        """Write a RequestTrace row to PostgreSQL in a background thread."""
        trace_id = str(uuid4())
        spans_json = json.dumps(timing.to_spans())

        def _write() -> None:
            try:
                from app.db import get_session_local
                from app.models import RequestTrace

                Session = get_session_local()
                db = Session()
                try:
                    trace = RequestTrace(
                        id=trace_id,
                        conversation_id=timing.request_id,
                        request_type=timing.request_type,
                        source=timing.source,
                        node_id=timing.node_id,
                        household_id=timing.household_id,
                        user_command=timing.user_command,
                        status=timing.trace_status,
                        error_message=timing.error_message,
                        total_duration_ms=round(total_ms, 1),
                        spans_json=spans_json,
                    )
                    db.add(trace)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.warning("Failed to persist request trace: %s", exc)
                finally:
                    db.close()
            except Exception as exc:
                logger.warning("Failed to persist request trace (session): %s", exc)

        threading.Thread(target=_write, daemon=True).start()


# Global instance
latency_logger = LatencyLogger()
