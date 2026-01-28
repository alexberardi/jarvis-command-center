"""
Latency Logger - Tracks timing throughout request processing.

Writes detailed timing data to a file for latency analysis.
"""

import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("uvicorn")


@dataclass
class TimingEntry:
    """A single timing measurement."""
    label: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None


@dataclass
class RequestTiming:
    """Timing data for a single request."""
    request_id: str
    request_type: str  # "warmup" or "voice_command"
    start_time: float = field(default_factory=time.perf_counter)
    entries: list[TimingEntry] = field(default_factory=list)
    _current_entry: Optional[TimingEntry] = field(default=None, repr=False)

    def checkpoint(self, label: str) -> None:
        """Record a checkpoint with elapsed time from request start."""
        elapsed = (time.perf_counter() - self.start_time) * 1000
        self.entries.append(TimingEntry(
            label=label,
            start_time=elapsed,
            end_time=elapsed,
            duration_ms=0
        ))

    @contextmanager
    def measure(self, label: str):
        """Context manager to measure a block of code."""
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            duration_ms = (end - start) * 1000
            elapsed_start = (start - self.start_time) * 1000
            elapsed_end = (end - self.start_time) * 1000
            self.entries.append(TimingEntry(
                label=label,
                start_time=elapsed_start,
                end_time=elapsed_end,
                duration_ms=duration_ms
            ))

    def total_ms(self) -> float:
        """Get total elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000


class LatencyLogger:
    """Thread-safe latency logger that writes to a file."""

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
        """End tracking and write timing data to file."""
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


# Global instance
latency_logger = LatencyLogger()
