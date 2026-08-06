"""Errand executor — dispatch a confirmed errand plan step-by-step.

The errand-flavored driver over the general ``workflow_engine``. An errand is NOT
a node routine: it's a plan of building-block steps the engine dispatches ONE AT A
TIME, routing each to its plane at run time (SEAM 2, ``workflow_engine`` handler
registry) and SUSPENDING on a deferred step (a phone call) until its outcome
resumes the run:

  - SERVER tool (make_phone_call, quick_search, remember, …) → run in-process via
    the same ``tool_executor.execute_tool`` the voice loop uses. Those tools read
    their household/user from ``conversation_cache`` by conversation_id, so a
    DETACHED run (a card tap — there is no live conversation) would find nothing.
    This module seeds a synthetic cache entry from the errand's household/node/user
    (the ``WorkflowContext.conv_id``); every server tool then resolves context
    identically to a real turn. make_phone_call is a DEFERRED handler that routes to
    ``phone_call_service.create_call_plan`` (confirm-before-dial) and suspends.

  - NODE command (get_weather, control_device, any installed command) → dispatch
    the ``tool_call`` MQTT verb and await the node's structured result.

This module owns the errand-side orchestration the engine leaves to its consumer:
the run loop with FAIL-FAST + suspend-on-deferred, and the friendly LLM-composed
completion summary. The per-step dispatch fork itself lives in ``workflow_engine``.
Nothing here speaks on the node — errands are headless (card feedback only).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

# The step-dispatch machinery now lives in the general engine (SEAM 2). Re-export
# the pieces the durable resume path + tests reference from this module so the
# extraction doesn't churn call sites (errand_service imports ``Suspended`` here).
from app.services.workflow_engine import (  # noqa: F401
    DEFERRED_SERVER_TOOLS,
    Suspend,
    Suspended,
    WorkflowContext,
    _node_step_outcome,
    _place_call,
    _resolve_node_args,
    _server_step_outcome,
    _summarize_server_result,
    run_step,
)

logger = logging.getLogger("uvicorn")


def _compose_message(results: list[dict[str, Any]]) -> str:
    """A plain-English roll-up of the step outcomes for the completion card.

    POC: a labelled join of each step's message/error. An LLM-composed summary
    (like the node RoutineCommand used to produce) is a follow-up — this keeps the
    honest per-step detail with zero extra failure modes.
    """
    if not results:
        return "The errand had no steps to run."
    parts: list[str] = []
    for r in results:
        label = r.get("label") or r.get("command") or "step"
        if r.get("success"):
            msg = r.get("message")
            parts.append(f"{label}: {msg}" if msg else f"{label}: done")
        else:
            parts.append(f"{label}: {r.get('error') or 'failed'}")
    return " • ".join(parts)


def _compact_data(data: Any, cap: int = 240) -> str:
    """A short string from a step's structured payload (context_data / tool result)
    for the compose prompt — node commands return DATA, not prose, so this is what
    the model phrases from. Skips bookkeeping keys and caps length."""
    if not isinstance(data, dict):
        return ""
    skip = {"success", "error", "actions", "message", "status"}
    parts: list[str] = []
    for key, value in data.items():
        if key in skip:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
        elif isinstance(value, (list, dict)):
            parts.append(f"{key}={json.dumps(value)}")
        if sum(len(p) for p in parts) > cap:
            break
    return "; ".join(parts)[:cap]


async def _compose_errand_message(goal: str, results: list[dict[str, Any]], fallback: str) -> str:
    """LLM-compose a short, friendly summary of the errand outcome from the step
    results — the errand-side equivalent of the node RoutineCommand's compose step
    (the old routine path did this on-node). Node commands return structured
    context_data for the model to phrase, so without this a data step reads as a
    bare 'done'. Falls back to ``fallback`` (the plain per-step join) on ANY failure
    or empty output — composition must never fail or block the run's terminal card.
    """
    lines: list[str] = []
    for r in results:
        label = r.get("label") or r.get("command") or "step"
        ok = r.get("success")
        detail = r.get("message") or _compact_data(r.get("data")) or r.get("error") or ""
        if not detail:
            # A step that "succeeded" but returned nothing — the model must NOT invent
            # what it produced (live: it said "a joke was told" for an empty tell_joke).
            detail = "finished but returned no information" if ok else "failed with no detail"
        lines.append(f"- {label} [{'ok' if ok else 'FAILED'}]: {detail}")
    prompt = (
        "You are Jarvis giving the user the FINAL report on a background errand that "
        f"has now FINISHED. Their goal was: {goal}\n\nWhat ACTUALLY happened, step by "
        "step:\n" + "\n".join(lines) +
        "\n\nWrite a short, friendly 1-2 sentence summary. Base EVERY statement STRICTLY "
        "on the detail shown after each step — quote the useful information (the weather, "
        "the answer, who you reached). NEVER invent an outcome: if a step's detail says it "
        "'returned no information', report that it ran but do NOT claim what it produced — "
        "do NOT say a joke was told, a message was sent, an item was found, or a task was "
        "done unless that detail is actually shown above. The errand is OVER, so NEVER say "
        "you 'will' do something and never mention any action not in the steps above. If a "
        "step FAILED, say so honestly (a failed step stops the errand, so later steps did "
        "NOT run).\n"
        "CRITICAL — the goal may DESCRIBE actions (set a reminder, send a message, make a "
        "call, add a to-do) that were NOT performed: ONLY the steps listed above are real. "
        "If the goal mentions such an action but there is NO step for it above, you MUST "
        "state it was NOT done (e.g. 'I checked the forecast — 100% rain — but I did not set "
        "a reminder'). Do not imply a conditional follow-up happened just because its "
        "condition was met. No preamble, no markdown.\n\n/no_think"
    )
    try:
        from app.core.llm_proxy_client import LLMProxyClient

        response = await LLMProxyClient().chat_completion(
            messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=400,
        )
        content = ""
        if isinstance(response, dict):
            choices = response.get("choices") or []
            if choices:
                content = choices[0].get("message", {}).get("content", "") or ""
        # Strip a stray Qwd3 <think> block if /no_think was ignored.
        if "</think>" in content:
            content = content.split("</think>", 1)[1]
        content = content.strip()
        return content or fallback
    except Exception:  # noqa: BLE001 — compose is best-effort; never block the card
        logger.warning("Errand result compose failed; using the plain summary")
        return fallback


async def aggregate_and_compose(goal: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate step outcomes into a final result AND LLM-compose the completion
    message. Shared by the executor's completion path and the resume handler's
    fail-fast path so both produce the same friendly, honest summary.

    Control-flow steps (an approved ``request_replan``; ``control: True``) are not
    user-facing WORK — a completion card must not narrate "the replanning task
    returned no information". They're dropped from the summary + the pass/fail
    tally (but kept in the record). If filtering leaves nothing, fall back to the
    full list so an all-control run still reports sensibly."""
    visible = [r for r in results if not r.get("control")] or results
    result = _aggregate(visible)
    result["results"] = results  # keep the full record, incl. control steps
    result["message"] = await _compose_errand_message(goal, visible, result["message"])
    return result


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for r in results if r.get("success"))
    failed = len(results) - passed
    if not results:
        status = "failed"
    elif failed == 0:
        status = "success"
    elif passed == 0:
        status = "failed"
    else:
        status = "partial"
    return {
        "status": status,
        "passed": passed,
        "failed": failed,
        "message": _compose_message(results),
        "results": results,
    }


async def execute_errand(
    household_id: str,
    node_id: str | None,
    steps: list[dict[str, Any]],
    *,
    user_id: int | None = None,
    goal: str = "",
    errand_id: str | None = None,
    start_index: int = 0,
    prior_results: list[dict[str, Any]] | None = None,
    node_timeout: float = 30.0,
) -> dict[str, Any] | Suspended:
    """Run a confirmed errand plan from ``start_index``, returning EITHER an
    aggregate ``{status, passed, failed, message, results}`` when it finishes, OR a
    ``Suspended`` when it hit a deferred (phone) step and is now waiting for that
    call's outcome to resume it (the caller persists ``state="waiting"``).

    ``steps`` are node-native ``[{command, args:{k:v}, label}]``. FAIL-FAST: the run
    stops at the first failed step — the user's "don't place call 2 unless call 1
    succeeded" — and reports what completed. A step never raises out: a crash
    becomes a failed outcome (which then stops the run). ``prior_results`` carries
    the outcomes from before a suspend so the completion card summarizes everything.
    """
    from app.core.conversation_cache import conversation_cache

    # Seed the context that server tools read by conversation_id. A card tap has
    # no live conversation, so without this a server step (e.g. make_phone_call
    # reading household_id/speaker_user_id) would fail. Removed in `finally`.
    # Household timezone for date-key resolution (headless — no client-provided tz
    # like the voice path has). attention.timezone is the household-level setting;
    # default UTC. Fetched + the date context built lazily, only if a step needs it.
    tz = "UTC"
    try:
        from app.services.settings_service import get_settings_service

        tz = get_settings_service().get("attention.timezone", household_id=str(household_id)) or "UTC"
    except Exception:  # noqa: BLE001 — settings unavailable → UTC
        tz = "UTC"

    _flat_cache: dict[str, Any] = {}

    def _date_flat() -> dict[str, Any]:
        if "flat" not in _flat_cache:
            try:
                from app.core.date_resolution import flatten_date_context
                from app.core.general_context import generate_date_context_object

                _flat_cache["flat"] = flatten_date_context(generate_date_context_object(tz))
            except Exception:  # noqa: BLE001 — degrade to sending keys unresolved
                logger.warning("Errand: date context unavailable; date keys sent unresolved")
                _flat_cache["flat"] = {}
        return _flat_cache["flat"]

    conv_id = f"errand-{uuid4().hex}"
    conversation_cache.set(
        conv_id,
        messages=[],
        available_commands=[],
        timezone=tz,
        tools=[],
        node_context={
            "household_id": household_id,
            "node_id": node_id,
            "speaker_user_id": user_id,
            "timezone": tz,
        },
    )
    ctx = WorkflowContext(
        household_id=household_id,
        node_id=node_id,
        user_id=user_id,
        goal=goal,
        timezone=tz,
        conv_id=conv_id,
        workflow_id=errand_id,
        node_timeout=node_timeout,
        date_flat=_date_flat,
    )

    results: list[dict[str, Any]] = list(prior_results or [])
    # Share the SAME list with the context so a step handler sees the outcomes of the
    # steps before it (a later call's brief summarizes the earlier ones). Appends below
    # are visible through ctx.results because it's the same object.
    ctx.results = results
    suspended: Suspended | None = None
    all_steps = steps or []
    try:
        for i in range(start_index, len(all_steps)):
            step = all_steps[i]
            command = (step.get("command") or "").strip()
            if not command:
                continue
            args = step.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            label = step.get("label") or command

            # Dispatch the step to its plane via the engine (SEAM 2). A SYNC step
            # returns an outcome dict; a DEFERRED step (a phone call) returns a
            # Suspend, at which point the errand pauses until the call's outcome
            # resumes it (continue on done, fail-fast otherwise).
            ret = await run_step(ctx, command, args, label, i)
            if isinstance(ret, Suspend):
                ret.results = results  # attach the outcomes completed so far
                suspended = ret
                break
            outcome = ret
            outcome["label"] = label
            results.append(outcome)
            if not outcome["success"]:
                logger.info("Errand fail-fast: step %d (%r) failed; stopping", i, command)
                break  # fail-fast — don't run steps that depended on this one
    finally:
        conversation_cache.remove(conv_id)

    if suspended is not None:
        logger.info(
            "Errand %s suspended on call session %s (resume at step %d)",
            errand_id, suspended.session_id, suspended.cursor,
        )
        return suspended

    # Compose a friendly summary from the step data (node commands return
    # structured context_data, not prose). Best-effort — falls back to a plain join.
    result = await aggregate_and_compose(goal, results)
    logger.info(
        "Errand run complete: status=%s passed=%d failed=%d (household=%s node=%s)",
        result["status"], result["passed"], result["failed"], household_id, node_id,
    )
    return result
