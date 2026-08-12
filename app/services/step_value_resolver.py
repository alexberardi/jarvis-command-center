"""Inter-step value resolver for workflow/errand steps.

The engine's ``_resolve_node_args`` only resolves date-KEYS; it has no way for one
step's arg to depend on an earlier step's RESULT. This adds that — deterministically,
in code — so an autonomous multi-step plan (e.g. drive_time → reminder) can flow a
computed value between steps without asking a small model to do arithmetic.

An arg value may be a ``$``-directive dict; the resolver replaces it with a concrete
value, or returns a ``skip_reason`` meaning "this step no longer makes sense, skip
it" (e.g. the leave-by time has already passed). The LLM composes the plan SHAPE
(which steps, which directive); this module owns the numbers.

Directives (M1):
  * ``{"$leave_by": {"event_start": <iso>, "drive_from_step": <i>,
        "drive_field": "duration_minutes", "buffer_minutes": <int>}}``
      → relative minutes from NOW until (event_start − drive − buffer). ``skip`` if
        that instant is already past.
  * ``{"$from_step": {"step": <i>, "field": <name>}}``
      → the named field from an earlier step's ``data`` (or top-level).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("uvicorn")


def _is_directive(value: Any) -> bool:
    return isinstance(value, dict) and len(value) == 1 and next(iter(value)).startswith("$")


def _drive_minutes(spec: dict[str, Any], prior: list[dict[str, Any]]) -> tuple[int | None, str | None]:
    idx = int(spec.get("drive_from_step", 0))
    field = spec.get("drive_field", "duration_minutes")
    if idx < 0 or idx >= len(prior):
        return None, f"no result for step {idx}"
    result = prior[idx] or {}
    if not result.get("success"):
        return None, f"step {idx} ({result.get('command')}) did not succeed"
    data = result.get("data") or {}
    raw = data.get(field, result.get(field))
    try:
        return int(raw), None
    except (TypeError, ValueError):
        return None, f"step {idx} has no numeric '{field}'"


def _parse_dt(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    # A naive timestamp (calendar sources should be tz-aware, but be defensive) is
    # treated as UTC so it can be compared against a tz-aware ``now``.
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _leave_by(spec: dict[str, Any], prior: list[dict[str, Any]], now: datetime) -> tuple[Any, str | None]:
    drive, skip = _drive_minutes(spec, prior)
    if skip is not None or drive is None:
        return None, skip or "no drive time"
    start = _parse_dt(spec.get("event_start"))
    if start is None:
        return None, "invalid event_start"
    buffer_minutes = int(spec.get("buffer_minutes", 5))
    leave_at = start - timedelta(minutes=drive + buffer_minutes)
    rel = round((leave_at - now).total_seconds() / 60)
    if rel <= 0:
        return None, f"departure time already passed ({rel} min)"
    return rel, None


def _from_step(spec: dict[str, Any], prior: list[dict[str, Any]]) -> tuple[Any, str | None]:
    idx = int(spec.get("step", 0))
    field = spec.get("field")
    if idx < 0 or idx >= len(prior):
        return None, f"no result for step {idx}"
    result = prior[idx] or {}
    data = result.get("data") or {}
    if field in data:
        return data[field], None
    if field in result:
        return result[field], None
    return None, f"step {idx} has no '{field}'"


def _resolve_directive(
    directive: dict[str, Any], prior: list[dict[str, Any]], now: datetime
) -> tuple[Any, str | None]:
    key = next(iter(directive))
    spec = directive[key] if isinstance(directive[key], dict) else {}
    if key == "$leave_by":
        return _leave_by(spec, prior, now)
    if key == "$from_step":
        return _from_step(spec, prior)
    return None, f"unknown directive {key}"


def resolve_step_args(
    args: dict[str, Any], prior_results: list[dict[str, Any]], now: datetime | None = None
) -> tuple[dict[str, Any], str | None]:
    """Resolve any ``$``-directive values in ``args`` against ``prior_results``.

    Returns ``(resolved_args, skip_reason)``. When ``skip_reason`` is not None the
    caller should SKIP the step (its arithmetic no longer makes sense — e.g. the
    leave-by time already passed) rather than dispatch it. Plain args pass through
    untouched.
    """
    now = now or datetime.now(timezone.utc)
    resolved: dict[str, Any] = {}
    for key, value in args.items():
        if _is_directive(value):
            out, skip = _resolve_directive(value, prior_results, now)
            if skip is not None:
                logger.info("step_value_resolver: skipping step — %s", skip)
                return args, skip
            resolved[key] = out
        else:
            resolved[key] = value
    return resolved, None
