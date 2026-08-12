"""SignalReactionBridge — the signal → EXECUTE entrypoint (the missing third mode).

A Signal edge could previously only PROPOSE (situation matcher, tap-gated) or RENDER
(reactive ambient block). This adds AUTORUN: a qualifying signal runs a low-blast
multi-step plan with no tap — the generic Signal-Bus payoff, where the assistant
composes existing node capabilities in response to the world.

First reaction: ``appt.upcoming`` → ``get_drive_time`` → ``reminder`` (leave-by).
Two independent gates, both fail-closed:
  * ``errands.autonomous_enabled`` (household setting) — off ⇒ no autorun at all.
  * the plan-start blast gate (:mod:`autorun_gate`) — only allowlisted, non-risky,
    no-outbound, few-step plans autorun; anything else is declined.
For M1 a declined/ineligible reaction is a no-op; M2 will fall back to a proposal
card. The plan is a fixed template for M1 (deterministic); M2 swaps in the LLM
planner (``create_errand_plan``) so the LLM genuinely composes from the open signal.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Awaitable, Callable

from app.services.autorun_gate import is_autorun_eligible

logger = logging.getLogger("uvicorn")

DEFAULT_LEAVE_BUFFER_MINUTES = 5

_main_loop: asyncio.AbstractEventLoop | None = None
# In-process dedup so the calendar agent's per-5-min re-emit of the same upcoming
# event doesn't spawn duplicate runs. Keyed on (household, event). Resets on restart
# — acceptable for M1 (the signal source_key upsert + agent-side keys back it up);
# M2/M3 add DB-backed dedup + cooldown.
_fired: set[str] = set()


# ── gates + helpers ───────────────────────────────────────────────────────────
def _autonomous_enabled(household_id: str) -> bool:
    """Fail-closed read of ``errands.autonomous_enabled`` (default False)."""
    try:
        from app.services.settings_service import get_settings_service

        value = get_settings_service().get("errands.autonomous_enabled", household_id=str(household_id))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False
    except Exception:  # noqa: BLE001 — fail closed
        return False


def _claim(household_id: str, event_id: str) -> bool:
    """Claim (household, event) for reaction; False if already claimed."""
    key = f"{household_id}:{event_id}"
    if key in _fired:
        return False
    _fired.add(key)
    return True


async def _node_command_names(node_id: str) -> set[str]:
    """The command names the node currently advertises (short-timeout probe)."""
    from app.api.node_tools import _request_tools_from_node

    report = await _request_tools_from_node(node_id, timeout=4.0)
    if not report:
        return set()
    return {c.get("command_name") for c in (report.get("available_commands") or []) if c.get("command_name")}


def _build_leave_by_plan(facts: dict[str, Any]) -> list[dict[str, Any]]:
    """The fixed M1 template: check drive time to the event, then set a reminder to
    leave — its ``relative_minutes`` a ``$leave_by`` directive the executor resolves
    from the drive-time result + the event start (see step_value_resolver)."""
    title = facts.get("title") or "your appointment"
    location = facts.get("location")
    start_iso = facts.get("start_iso") or facts.get("start")
    return [
        {"command": "get_drive_time", "args": {"destination": location},
         "label": f"Check drive time to {location}", "is_risky": False},
        {"command": "reminder",
         "args": {"action": "set", "text": f"Leave for {title}",
                  "relative_minutes": {"$leave_by": {
                      "event_start": start_iso, "drive_from_step": 0,
                      "buffer_minutes": DEFAULT_LEAVE_BUFFER_MINUTES}}},
         "label": f"Set leave reminder for {title}", "is_risky": False},
    ]


async def _spawn_and_run_errand(
    *, household_id: str, node_id: str | None, user_id: int | None,
    goal: str, steps: list[dict[str, Any]], title: str | None = None,
) -> str:
    """Spawn a durable Workflow RUN from the steps and drive it headless — the
    autorun analogue of the plan card's Run tap (``_handle_approve_errand_plan``),
    minus the draft/plan-card (there is no approval to gather)."""
    from app.db import get_session_local
    from app.models import Workflow
    from app.services.workflow_engine import run_workflow

    db = get_session_local()()
    try:
        workflow = Workflow(
            kind="errand", household_id=household_id, user_id=user_id, node_id=node_id,
            goal=goal, title=title or goal, steps=json.dumps(steps),
            cursor=0, results_json="[]", state="running",
        )
        db.add(workflow)
        db.flush()
        workflow_id = workflow.id
        db.commit()
    finally:
        db.close()

    await run_workflow(workflow_id)  # owns terminal state + the one completion card
    return workflow_id


# ── the reaction ──────────────────────────────────────────────────────────────
async def react_to_appt_upcoming(
    household_id: str,
    facts: dict[str, Any],
    node_id: str | None,
    user_id: int | None,
    *,
    enabled_check: Callable[[str], bool] | None = None,
    menu_fetch: Callable[[str], Awaitable[set[str]]] | None = None,
    runner: Callable[..., Awaitable[str]] | None = None,
) -> str:
    """React to an ``appt.upcoming`` signal by autorunning the leave-by plan.

    Returns a short status string (for logging/tests): one of ``disabled``,
    ``no_location``/``no_start``/``no_user``/``no_node``, ``duplicate``,
    ``no_drive_time``, ``not_eligible``, ``autorun_started``. Never raises.
    """
    enabled_check = enabled_check or _autonomous_enabled
    if not enabled_check(household_id):
        return "disabled"
    if not node_id:
        return "no_node"
    if not facts.get("location"):
        return "no_location"          # leave-by is meaningless without a place
    if not (facts.get("start_iso") or facts.get("start")):
        return "no_start"
    if user_id is None:
        return "no_user"              # reminder(set) needs a concrete speaker

    event_id = str(facts.get("event_id") or facts.get("id") or facts.get("start_iso") or facts.get("start"))
    if not _claim(household_id, event_id):
        return "duplicate"

    try:
        menu_fetch = menu_fetch or _node_command_names
        names = await menu_fetch(node_id)
        if "get_drive_time" not in names or "reminder" not in names:
            logger.info("leave-by: node %s lacks get_drive_time/reminder — skipping", node_id)
            return "no_drive_time"

        steps = _build_leave_by_plan(facts)
        eligible, reason = is_autorun_eligible(steps)
        if not eligible:
            logger.info("leave-by: plan not autorun-eligible (%s)", reason)
            return "not_eligible"

        runner = runner or _spawn_and_run_errand
        await runner(
            household_id=household_id, node_id=node_id, user_id=user_id,
            goal=f"Leave on time for {facts.get('title') or 'your appointment'}",
            steps=steps, title="Leave-by reminder",
        )
        return "autorun_started"
    except Exception:  # noqa: BLE001 — a reaction must never break signal ingest
        logger.warning("leave-by reaction failed for %s", household_id, exc_info=True)
        return "error"


# ── fire-and-forget scheduling from the (sync) ingest path ────────────────────
def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop


def schedule_appt_upcoming_reaction(
    household_id: str, node_id: str | None, user_id: int | None,
    kind: str | None, facts: dict[str, Any],
) -> None:
    """Schedule the reaction for an ``appt.upcoming`` signal onto the main loop —
    fire-and-forget, never blocks or raises into the (sync) ingest path."""
    if kind != "appt.upcoming":
        return
    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = _main_loop
        if loop is None:
            return
        loop.call_soon_threadsafe(
            lambda: loop.create_task(
                react_to_appt_upcoming(household_id, facts or {}, node_id, user_id)
            )
        )
    except Exception:  # noqa: BLE001 — scheduling must never break ingest
        pass
