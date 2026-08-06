"""Workflow engine — the durable, resumable, event-driven job engine.

This is the general engine extracted from the Errand Runner (prds/errand-runner.md
§2.5-2.6). An errand is just its first consumer. A *workflow* is a plan of
building-block steps that CC dispatches ONE AT A TIME, routing each to wherever
its capability physically lives, and SUSPENDS on a step that can't answer inline
(a phone call, later a timer) until an external signal resumes it.

The engine has three seams, and this module owns the first two:

  SEAM 2 — the HANDLER REGISTRY (this module). A step names a capability; a
    registered ``StepHandler`` says how to run it and whether it is SYNC (returns
    a result inline) or DEFERRED (kicks off an async, externally-completed action
    and SUSPENDS the workflow on a named signal). Three handlers ship:

      - ``phone_call``   (DEFERRED) — make_phone_call → phone_call_service; the
                          workflow suspends on the call session's terminal outcome.
      - ``server_tool``  (SYNC)     — any registered IServerTool, run in-process via
                          the same ``tool_executor`` the voice loop uses.
      - ``node_command`` (SYNC)     — any installed node IJarvisCommand, dispatched
                          headless over MQTT (``dispatch_node_command``).

    Handlers are tried in registration order (priority); the node handler is the
    catch-all fallback and registers last, so ``resolve_handler`` never fails. This
    is the SAME server-vs-node discriminator the conversation tool loop uses
    (``tool_registry.has_tool``) — no step carries a "plane" tag; the plane is
    decided at run time.

  SEAM 1 — the SIGNAL primitive (deliver_signal) and SEAM 3 — the PROGRESS
    channel arrive in later phases; a deferred handler already declares the signal
    kind it suspends on (``Suspend.signal_kind``) so those seams have something to
    key on.

Nothing here speaks on a node or renders a card — a workflow is headless; its
consumer owns how progress is surfaced (SEAM 3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("uvicorn")


# ── Workflow-run context — what a handler needs to run one step ───────────────
@dataclass
class WorkflowContext:
    """The per-run context threaded to every step handler.

    ``conv_id`` is a synthetic ``conversation_cache`` entry the runner seeds so
    server tools (which read household/speaker by conversation id) resolve context
    on a detached run — a card tap has no live conversation. ``workflow_id`` is the
    durable run this belongs to (an errand id today); a DEFERRED handler links its
    external resource back to ``(workflow_id, step_index)`` so a signal can find it.
    ``date_flat`` lazily yields the flattened date context for node-arg resolution.
    """

    household_id: str
    node_id: str | None
    user_id: int | None = None
    goal: str = ""
    timezone: str = "UTC"
    conv_id: str = ""
    workflow_id: str | None = None
    node_timeout: float = 30.0
    date_flat: Callable[[], dict[str, Any]] = field(default=lambda: {})
    # The outcomes of the steps completed so far this run — the runner shares its
    # accumulating list here so a step handler can summarize earlier steps (a later
    # call's brief references what the earlier ones established).
    results: list[dict[str, Any]] = field(default_factory=list)


# ── Suspend — a deferred step paused the workflow on a named signal ───────────
class Suspend:
    """A DEFERRED step suspended the workflow; it is now waiting for an external
    signal to resume it. ``cursor`` = the next step index to run on resume;
    ``signal_kind`` = which wakeup source will resume it (``phone_call`` today,
    ``timer`` later); ``signal_key`` = the correlation key for that signal (a
    phone_call_session id today); ``results`` = the per-step outcomes so far,
    attached by the runner before it returns.
    """

    __slots__ = ("cursor", "signal_kind", "signal_key", "results", "wake_at")

    def __init__(
        self,
        cursor: int,
        signal_key: str | None = None,
        results: list[dict[str, Any]] | None = None,
        *,
        signal_kind: str = "phone_call",
        session_id: str | None = None,
        wake_at: Any = None,
    ):
        # ``session_id`` is the historical name for the phone-call signal key; accept
        # it as an alias so the ``Suspended`` back-compat call sites keep working.
        self.cursor = cursor
        self.signal_key = signal_key if signal_key is not None else (session_id or "")
        self.signal_kind = signal_kind
        self.results = results if results is not None else []
        # For a ``timer`` wait: the UTC instant to resume (a ``datetime``). None for
        # an external-event (phone) wait — the event, not the clock, resumes it.
        self.wake_at = wake_at

    # Back-compat: the durable executor + tests refer to a suspended phone step's
    # session by ``.session_id``. It's just the signal key for the phone_call kind.
    @property
    def session_id(self) -> str:
        return self.signal_key


# ``Suspended`` is the historical name (errand_executor + tests import it). Keep it
# as an alias so the extraction doesn't churn call sites.
Suspended = Suspend


# ── Step-outcome normalizers — shared by the sync handlers ────────────────────
def _summarize_server_result(obj: dict[str, Any]) -> str | None:
    """A brief human line for a successful server tool that returned DATA but no
    ``message`` (quick_search → ``sources``, recall → ``memories``). Without this
    such a step renders as a bare "done" and the work looks empty. The full payload
    is still kept on the outcome's ``data`` field for a richer card later."""
    for key in ("sources", "memories", "results", "items"):
        val = obj.get(key)
        if isinstance(val, list) and val:
            return f"{len(val)} {key}"
    status = obj.get("status")
    if isinstance(status, str) and status not in ("accepted",):
        return status
    return None


def _server_step_outcome(command: str, raw: Any) -> dict[str, Any]:
    """Normalize a server tool's result (``tool_executor.execute_tool`` returns a
    JSON string) into a step outcome. A dict carrying an ``error`` key is a
    failure (the registry wraps raised exceptions and missing tools that way);
    otherwise the step succeeded — ``message`` is the human text, or a brief
    summary of returned data, and the raw payload is preserved on ``data``."""
    try:
        obj = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        obj = None
    if not isinstance(obj, dict):
        obj = {}
    err = obj.get("error")
    if err is not None:
        return {
            "command": command, "success": False, "message": None,
            "error": obj.get("message") or str(err), "data": None,
        }
    return {
        "command": command, "success": True,
        "message": obj.get("message") or _summarize_server_result(obj),
        "error": None, "data": obj,  # keep the payload so it isn't lost
    }


def _node_step_outcome(command: str, output: Any) -> dict[str, Any]:
    """Normalize a node command's ``output`` dict (``context_data`` + ``success``
    + optional ``error``/``message``) into a step outcome. A plain context_data
    return with no explicit ``success`` is treated as success."""
    if not isinstance(output, dict):
        return {"command": command, "success": True, "message": None, "error": None, "data": None}
    err = output.get("error")
    success = bool(output.get("success", True)) and not err
    return {
        "command": command,
        "success": success,
        "message": output.get("message") if success else None,
        "error": str(err) if err else (None if success else "the command reported a failure"),
        "data": output if success else None,  # the node command's context_data
    }


def _resolve_datetimes(values: list, flat: dict[str, Any]) -> list:
    """Resolve relative date keys ('today', 'tomorrow') to the ISO datetimes node
    commands expect, keeping any already-ISO values. The workflow-time equivalent of
    the conversation loop's ``_inject_date_keys`` for the canonical
    ``resolved_datetimes`` param — without it, get_weather/get_calendar_events get
    a raw keyword and fail ('Dates not found in forecast')."""
    from app.core.date_resolution import normalize_date_key
    from app.core.param_validation import is_iso_datetime

    out: list = []
    for v in values:
        if not isinstance(v, str):
            continue
        if is_iso_datetime(v):
            out.append(v)  # already resolved
            continue
        resolved = flat.get(normalize_date_key(v))
        if isinstance(resolved, str):
            out.append(resolved)
        elif isinstance(resolved, list):
            out.extend(x for x in resolved if isinstance(x, str))
    return out or values  # if nothing resolved, send the originals rather than empty


def _resolve_node_args(args: dict[str, Any], flat_getter: Callable[[], dict]) -> dict[str, Any]:
    """Resolve date-key args in a node step before dispatch. Currently the
    canonical ``resolved_datetimes`` list/scalar; other args pass through."""
    dt = args.get("resolved_datetimes")
    if dt is None:
        return args
    dt_list = dt if isinstance(dt, list) else [dt]
    return {**args, "resolved_datetimes": _resolve_datetimes(dt_list, flat_getter())}


# ── Deferred: the phone-call plane ────────────────────────────────────────────
# Server tools that are DEFERRED — they don't return a result inline; they kick
# off an async, human-gated action (a phone call: draft → confirm → dial →
# outcome) and the workflow must SUSPEND until it finishes, then decide whether to
# continue. make_phone_call is the first. The handler routes these to their real
# handler (create_call_plan) and suspends on the call's terminal outcome.
DEFERRED_SERVER_TOOLS = frozenset({"make_phone_call"})


def _prior_steps_context(results: list[dict[str, Any]] | None) -> str:
    """A short summary of the errand's EARLIER steps, for the next call's brief so it
    knows what came before (e.g. the medication a prior pharmacy call confirmed). Uses
    each step's wrapup ``summary`` when present, else its message/error."""
    lines: list[str] = []
    for r in results or []:
        label = (r.get("label") or "Step").strip()
        detail = (r.get("summary") or r.get("message") or r.get("error") or "").strip()
        if detail:
            lines.append(f"- {label}: {detail}")
    if not lines:
        return ""
    return ("Earlier in this errand (background — use only if the business asks or it's "
            "relevant):\n" + "\n".join(lines))


async def _place_call(
    household_id: str, user_id: int | None, args: dict[str, Any], goal: str,
    workflow_id: str, step_index: int, prior_results: list[dict[str, Any]] | None = None,
) -> str | None:
    """Draft a workflow-linked phone call via its real handler (create_call_plan).
    Returns the session id (drafted → the workflow SUSPENDS on it, and the call's
    terminal outcome resumes it) or None (couldn't even draft → a failed step)."""
    from app.services.phone_call_service import create_call_plan

    business = (args.get("business") or "").strip()
    call_goal = (args.get("goal") or goal or "").strip()
    if not business or not call_goal:
        return None
    try:
        return await create_call_plan(
            business=business, goal=call_goal, household_id=household_id,
            user_id=user_id, errand_id=workflow_id, errand_step=step_index,
            prior_context=_prior_steps_context(prior_results),
        )
    except Exception:  # noqa: BLE001 — a draft failure is a failed step, not a crash
        logger.exception("Workflow call step failed to draft (%s)", business)
        return None


# ── Step handlers (SEAM 2) ────────────────────────────────────────────────────
class StepHandler:
    """A capability plane. ``kind`` is "sync" (returns an outcome dict inline) or
    "deferred" (may return a ``Suspend`` to pause the workflow on a signal).
    ``handles`` is the resolution predicate; handlers are tried in registration
    order and the first match wins."""

    kind: str = "sync"
    signal_kind: str | None = None  # deferred handlers only

    def handles(self, command: str) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    async def run(
        self, ctx: WorkflowContext, command: str, args: dict[str, Any],
        label: str, step_index: int,
    ) -> dict[str, Any] | Suspend:  # pragma: no cover - overridden
        raise NotImplementedError

    def interpret_signal(
        self, outcome: Any, steps: list[dict[str, Any]], step_index: int,
    ) -> dict[str, Any]:  # pragma: no cover - only deferred handlers implement this
        """DEFERRED handlers only: turn the external event that resumes the workflow
        into this step's outcome dict. ``deliver_signal`` calls it."""
        raise NotImplementedError


def _outcome_goal_achieved(outcome: Any) -> bool | None:
    """Whether the phone call's GOAL was achieved — the truth signal for the errand's
    fail-fast decision, distinct from the call merely reaching state 'done'. The
    gateway reports it in the outcome; read a direct ``goal_achieved`` attribute
    (tests set it) else parse the session's ``outcome_json`` (the ORM row / the
    resume snapshot). None = unconfirmed."""
    direct = getattr(outcome, "goal_achieved", None)
    if direct is not None:
        return direct
    raw = getattr(outcome, "outcome_json", None)
    if raw:
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(data, dict):
                return data.get("goal_achieved")
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def _outcome_summary(outcome: Any) -> str:
    """The gateway's post-call wrapup summary (+ any confirmed facts) for this call —
    what a LATER step needs to know about what this one established (e.g. the medication
    the pharmacy confirmed). Empty when unavailable."""
    raw = getattr(outcome, "outcome_json", None)
    if not raw:
        return ""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    summary = (data.get("summary") or "").strip()
    facts = data.get("facts")
    if isinstance(facts, list) and facts:
        summary = f"{summary} (confirmed: {', '.join(str(f) for f in facts)})".strip()
    return summary


class PhoneCallHandler(StepHandler):
    """DEFERRED — a phone call. Drafts the call linked to the workflow, then
    SUSPENDS on the call session; the session's terminal outcome resumes it."""

    kind = "deferred"
    signal_kind = "phone_call"

    def handles(self, command: str) -> bool:
        return command in DEFERRED_SERVER_TOOLS

    async def run(self, ctx, command, args, label, step_index):
        if not ctx.workflow_id:  # can't resume an unlinked call → fail the step
            return {"command": command, "label": label, "success": False,
                    "message": None, "error": "call step has no workflow link", "data": None}
        session_id = await _place_call(
            ctx.household_id, ctx.user_id, args, ctx.goal, ctx.workflow_id, step_index,
            prior_results=ctx.results,
        )
        if session_id:
            return Suspend(cursor=step_index + 1, signal_key=session_id,
                           signal_kind=self.signal_kind or "phone_call")
        return {"command": command, "label": label, "success": False,
                "message": None, "error": "couldn't set up the call", "data": None}

    def interpret_signal(
        self, outcome: Any, steps: list[dict[str, Any]], step_index: int,
    ) -> dict[str, Any]:
        """Turn a terminal phone_call_session (the external event that resumes the
        workflow) into this step's outcome dict, deciding whether the errand may
        CONTINUE to the next step.

        Critically, a ``done`` call STATE means the call *connected and finished* —
        NOT that it achieved the errand's goal. The gateway reports the real result
        in the outcome's ``goal_achieved`` (True | False | None). A call to a
        stranger is a high-stakes, irreversible step, so we chain the next call ONLY
        on an EXPLICIT success: ``goal_achieved is True``. A call that connected but
        did NOT achieve the goal (False) — or where the result is unconfirmed (None)
        — FAIL-FASTS the errand (don't dial the next business on ambiguous success).

        ``outcome`` is the phone session (an ORM row or a resume snapshot) exposing
        ``state``/``contact_name``/``error_message`` and either ``goal_achieved`` or
        ``outcome_json``; ``steps``/``step_index`` supply the step's label."""
        label = "Call"
        if 0 <= step_index < len(steps):
            label = steps[step_index].get("label") or "Call"
        who = getattr(outcome, "contact_name", None) or "the business"
        state = getattr(outcome, "state", None) or ""
        if state == "done":
            achieved = _outcome_goal_achieved(outcome)
            summary = _outcome_summary(outcome)  # threaded into later steps' briefs
            if achieved is True:
                return {"command": "make_phone_call", "label": label, "success": True,
                        "message": f"Called {who} and achieved the goal.",
                        "summary": summary, "error": None, "data": None}
            # Connected, but the goal was not confirmed achieved → fail-fast.
            reason = (f"reached {who} but the goal wasn't achieved" if achieved is False
                      else f"reached {who} but couldn't confirm the goal was achieved")
            return {"command": "make_phone_call", "label": label, "success": False,
                    "message": None, "summary": summary, "error": reason, "data": None}
        reason = {
            "failed": "the call didn't go through",
            "declined": "the call was declined",
            "expired": "the call plan expired before it was confirmed",
        }.get(state, f"the call ended as {state}")
        return {"command": "make_phone_call", "label": label, "success": False,
                "message": None, "error": (getattr(outcome, "error_message", None) or reason),
                "data": None}


def _resolve_wake_at(args: dict[str, Any]) -> Any:
    """Resolve a ``wait_for`` step's wake instant (naive UTC ``datetime``) from its
    args, or None if it can't be understood. Supports ``delay_seconds`` (a relative
    offset from now) and ``until`` (an ISO-8601 instant). Kept deliberately small —
    the point of phase 4 is proving the timer wakeup, not a rich time grammar."""
    from datetime import datetime, timedelta

    delay = args.get("delay_seconds")
    if isinstance(delay, (int, float)) and not isinstance(delay, bool) and delay >= 0:
        return datetime.utcnow() + timedelta(seconds=float(delay))
    until = args.get("until")
    if isinstance(until, str) and until.strip():
        try:
            dt = datetime.fromisoformat(until.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
        # Store naive UTC to match the ``workflows.wake_at`` column (naive) + the
        # sweep's ``datetime.utcnow()`` comparison.
        if dt.tzinfo is not None:
            dt = dt.astimezone(_utc()).replace(tzinfo=None)
        return dt
    return None


def _utc():
    from datetime import timezone

    return timezone.utc


class WaitForHandler(StepHandler):
    """DEFERRED (timer) — the ``wait_for`` control-flow step. Computes the wake
    instant and SUSPENDS the run on the TIMER wakeup source; the timer sweep resumes
    it when ``wake_at`` passes, through the SAME ``deliver_signal`` primitive the
    phone outcome uses. A wait already in the past runs through as an immediate
    success (no suspend). This is the second signal source that earns the engine its
    name (errand-runner PRD §2.5)."""

    kind = "deferred"
    signal_kind = "timer"

    def handles(self, command: str) -> bool:
        return command == "wait_for"

    async def run(self, ctx, command, args, label, step_index):
        from datetime import datetime

        wake_at = _resolve_wake_at(args)
        if wake_at is None:
            return {"command": "wait_for", "label": label, "success": False,
                    "message": None, "error": "couldn't understand the wait time", "data": None}
        if wake_at <= datetime.utcnow():
            # Already due → don't suspend; the wait is trivially over.
            return {"command": "wait_for", "label": label, "success": True,
                    "message": "the wait was already over", "error": None, "data": None}
        if not ctx.workflow_id:  # can't resume an unlinked timer → fail the step
            return {"command": "wait_for", "label": label, "success": False,
                    "message": None, "error": "wait step has no workflow link", "data": None}
        return Suspend(cursor=step_index + 1, signal_key=wake_at.isoformat(),
                       signal_kind=self.signal_kind or "timer", wake_at=wake_at)

    def interpret_signal(self, outcome, steps, step_index):
        """The timer fired — the wait is over. Always a success (a wait doesn't
        'fail'); the run continues from the cursor."""
        label = "Wait"
        if 0 <= step_index < len(steps):
            label = steps[step_index].get("label") or "Wait"
        return {"command": "wait_for", "label": label, "success": True,
                "message": "the wait finished", "error": None, "data": None}


def widens_envelope(
    approved_steps: list[dict[str, Any]], amended_steps: list[dict[str, Any]]
) -> bool:
    """Does the amended plan WIDEN the errand's envelope vs. the approved plan?

    The gate for mid-run pause-and-replan: a widening change must pause for a
    one-tap human confirm (a delta card); a non-widening change auto-continues.
    Four axes over the step dicts (errand-runner PRD §2.6). The autonomy-dial is
    deferred, so this is a strict binary — no tunability, no pre-approving
    patterns; ANY axis true → widen:

      1. NEW CAPABILITY   — a command in the amended plan not already approved.
      2. NEW COUNTERPARTY — a step whose ``args.business`` is a party not already
                            being contacted in the approved plan.
      3. NEW RISKY STEP   — the amended plan has MORE ``is_risky`` steps than the
                            approved one (spends money / contacts a stranger /
                            irreversible — e.g. an extra make_phone_call).
      4. GUARDRAIL-LOOSENING — stubbed False until per-step clamp values exist
                            (lands with the deferred autonomy-dial / clamp).

    Pure + dependency-free so it unit-tests as a truth table. Reordering or
    REMOVING steps never widens (no new command/party, risky count can't rise).
    """
    def _party(step: dict[str, Any]) -> str | None:
        return ((step.get("args") or {}).get("business") or "").strip() or None

    approved_cmds = {s.get("command") for s in approved_steps}
    approved_parties = {_party(s) for s in approved_steps if _party(s)}
    approved_risky = sum(1 for s in approved_steps if s.get("is_risky"))

    # Axis 3: a net-new risky step (catches a second call even to a known party).
    if sum(1 for s in amended_steps if s.get("is_risky")) > approved_risky:
        return True
    for s in amended_steps:
        if s.get("command") not in approved_cmds:
            return True  # axis 1
        party = _party(s)
        if party and party not in approved_parties:
            return True  # axis 2
    return False  # axis 4 stubbed → not looser


class ReplanHandler(StepHandler):
    """DEFERRED (approval) — the ``request_replan`` control-flow step. A mid-run plan
    change (a discovered follow-up, extra work) SUSPENDS the run on the APPROVAL
    wakeup source; the consumer surfaces a delta card, and a human tap — or an
    auto-continue for an in-envelope change — resumes it through the SAME
    ``deliver_signal`` primitive the phone/timer sources use. The step's args carry
    the proposed delta (``add_steps``, ``reason``), which the consumer reads off the
    step to render the card and (on approval) splice in. This is the third signal
    source: gates, escalations, wait-fors and discovered branches all reduce to one
    suspend→get-a-decision→resume primitive (errand-runner PRD §2.6)."""

    kind = "deferred"
    signal_kind = "approval"

    def handles(self, command: str) -> bool:
        return command == "request_replan"

    async def run(self, ctx, command, args, label, step_index):
        if not ctx.workflow_id:  # can't resume an unlinked approval → fail the step
            return {"command": "request_replan", "label": label, "success": False,
                    "message": None, "error": "replan step has no workflow link", "data": None}
        return Suspend(cursor=step_index + 1, signal_key=f"{ctx.workflow_id}:{step_index}",
                       signal_kind=self.signal_kind or "approval")

    def interpret_signal(self, outcome, steps, step_index):
        """The replan was approved (a tap, or an in-envelope auto-continue) → a
        success outcome so the run continues from the cursor over the AMENDED plan
        (the consumer splices the delta in BEFORE delivering this signal). A declined
        replan never delivers ``approval`` — Stop cancels the run instead — so this
        only ever runs for an approval. ``control`` marks it a control-flow step so
        the completion summary can skip it."""
        label = "Replan"
        if 0 <= step_index < len(steps):
            label = steps[step_index].get("label") or "Replan"
        return {"command": "request_replan", "label": label, "success": True,
                "message": None, "error": None, "data": None, "control": True}


class ServerToolHandler(StepHandler):
    """SYNC — any registered IServerTool, run in-process via the same tool_executor
    the voice loop uses. Reads household/speaker from the seeded ``conv_id``."""

    kind = "sync"

    def handles(self, command: str) -> bool:
        from app.core.tool_registry import tool_registry

        return tool_registry.has_tool(command)

    async def run(self, ctx, command, args, label, step_index):
        from app.core.tool_executor import tool_executor

        raw = tool_executor.execute_tool(
            command, args, conversation_id=ctx.conv_id, user_utterance=ctx.goal or None
        )
        outcome = _server_step_outcome(command, raw)
        outcome["label"] = label
        logger.info("Workflow step (server) %r → success=%s", command, outcome["success"])
        return outcome


class NodeCommandHandler(StepHandler):
    """SYNC (fallback) — any installed node IJarvisCommand, dispatched headless
    over MQTT and awaited. Registered LAST so it catches anything the other
    handlers don't claim; ``handles`` is always True."""

    kind = "sync"

    def handles(self, command: str) -> bool:
        return True

    async def run(self, ctx, command, args, label, step_index):
        from app.services.node_command_service import dispatch_node_command

        node_args = _resolve_node_args(args, ctx.date_flat)  # keys → ISO datetimes
        output = await dispatch_node_command(
            ctx.node_id, command, node_args,
            user_id=ctx.user_id, voice_command=ctx.goal or None, timeout=ctx.node_timeout,
        )
        outcome = _node_step_outcome(command, output)
        outcome["label"] = label
        logger.info("Workflow step (node) %r → success=%s", command, outcome["success"])
        return outcome


# The registry — ordered by priority. Phone (deferred) is checked before the
# server-tool handler because make_phone_call IS a server tool; the node handler
# is the catch-all and must be LAST.
_HANDLERS: list[StepHandler] = []


def register_handler(handler: StepHandler) -> None:
    """Append a step handler to the resolution chain (lowest priority last). The
    engine registers the three built-in planes at import; consumers may add more."""
    _HANDLERS.append(handler)


def resolve_handler(command: str) -> StepHandler:
    """Pick the handler for a command — first match in registration order. The node
    handler's ``handles`` is always True and it is registered last, so this never
    raises for a non-empty command."""
    for handler in _HANDLERS:
        if handler.handles(command):
            return handler
    # Unreachable while NodeCommandHandler is registered, but be explicit.
    raise LookupError(f"no workflow step handler for command {command!r}")


async def run_step(
    ctx: WorkflowContext, command: str, args: dict[str, Any], label: str, step_index: int,
) -> dict[str, Any] | Suspend:
    """Dispatch ONE workflow step to its resolved handler. Returns an outcome dict
    (``{command, success, message, error, data, label}``) for a SYNC step, or a
    ``Suspend`` when a DEFERRED handler paused the workflow. A crash inside a
    handler is caught and returned as a failed outcome — one bad step must never
    raise out of the runner loop (which then fail-fasts on the failure)."""
    handler = resolve_handler(command)
    try:
        result = await handler.run(ctx, command, args, label, step_index)
    except Exception as exc:  # noqa: BLE001 — one bad step must not raise out
        logger.exception("Workflow step %r crashed", command)
        return {"command": command, "label": label, "success": False,
                "message": None, "error": str(exc), "data": None}
    if isinstance(result, Suspend):
        return result
    if isinstance(result, dict):
        result.setdefault("label", label)
    return result


def _deferred_handler(signal_kind: str) -> StepHandler | None:
    """The registered DEFERRED handler that suspends on ``signal_kind`` (e.g. the
    phone handler for ``phone_call``). ``deliver_signal`` uses it to interpret an
    external event into a step outcome."""
    for handler in _HANDLERS:
        if handler.kind == "deferred" and handler.signal_kind == signal_kind:
            return handler
    return None


# ── The run/instance the durable primitive operates on ────────────────────────
@dataclass
class WorkflowRun:
    """A snapshot of a durable workflow RUN — what ``deliver_signal`` reads to
    resume it. Decoupled from any ORM: a store maps its backing row (``ErrandPlan``
    today, a ``workflows`` row after the data split) onto this shape, so the engine
    never imports a consumer's model. ``goal`` feeds the completion summary;
    ``title`` is the display label for progress; ``kind`` selects the consumer."""

    id: str
    kind: str
    household_id: str
    node_id: str | None
    user_id: int | None
    goal: str
    title: str
    steps: list[dict[str, Any]]
    cursor: int
    results: list[dict[str, Any]]
    state: str


class WorkflowStore:
    """Durable persistence for ONE workflow RUN as a short unit of work — the
    load-bearing seam that decouples the engine from a consumer's ORM. The errand
    consumer backs it with the ``workflows`` table; a different consumer could back
    it with anything, WITHOUT changing the engine. A fresh store is opened per
    ``deliver_signal``/``run_workflow`` (via the registered factory), holds its own
    DB session, and is closed when the call ends. ``claim_running`` is the atomic
    ``waiting→running`` guard that serializes the immediate hook against the sweep."""

    def load(self, workflow_id: str) -> WorkflowRun | None:  # pragma: no cover - overridden
        raise NotImplementedError

    def claim_running(self, workflow_id: str) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def save_waiting(  # pragma: no cover - overridden
        self, run: WorkflowRun, cursor: int, results: list[dict[str, Any]],
        *, signal_kind: str = "phone_call", wake_at: Any = None,
    ) -> None:
        raise NotImplementedError

    def save_terminal(self, run: WorkflowRun, state: str, error: str | None) -> None:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# ── SEAM 3: the progress channel ──────────────────────────────────────────────
@dataclass
class ProgressEvent:
    """A moment in a workflow's life the consumer may surface however it likes
    (errands render inbox cards). ``complete`` carries the aggregate ``result``."""

    type: str  # "started" | "step_done" | "needs_a_tap" | "complete"
    result: dict[str, Any] | None = None


class ProgressEmitter:
    def emit(self, run: WorkflowRun, event: ProgressEvent) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


# ── Consumer registration — how a workflow KIND runs, finalizes, and reports ──
@dataclass
class WorkflowConsumer:
    """Everything the engine needs to drive a workflow of a given ``kind`` without
    importing the consumer: how to run its steps, how to finalize into a summary,
    and how to surface progress. Errands register one under ``"errand"``."""

    run_fn: Callable[..., Any]       # async: run the plan from a cursor → aggregate | Suspend
    finalize_fn: Callable[..., Any]  # async: (goal, results) → aggregate {status, message, ...}
    progress: ProgressEmitter


WorkflowStoreFactory = Callable[[], WorkflowStore]
_STORE_FACTORY: WorkflowStoreFactory | None = None
_CONSUMERS: dict[str, WorkflowConsumer] = {}


def register_workflow_store(factory: WorkflowStoreFactory) -> None:
    """Install THE durable run store factory (single — the workflows table is
    single, kind-agnostic). Each ``deliver_signal``/``run_workflow`` opens one store
    (its own DB session). The errand consumer installs a ``workflows``-backed factory."""
    global _STORE_FACTORY
    _STORE_FACTORY = factory


def register_consumer(kind: str, consumer: WorkflowConsumer) -> None:
    """Register how a workflow ``kind`` runs/finalizes/reports."""
    _CONSUMERS[kind] = consumer


def emit_progress(run: WorkflowRun, event: ProgressEvent) -> None:
    """Fan a progress event to the run's consumer emitter. Best-effort — a
    reporting failure must never crash the durable path (a completion card is the
    user-facing artifact but the run already happened)."""
    consumer = _CONSUMERS.get(run.kind)
    if consumer is None:
        return
    try:
        consumer.progress.emit(run, event)
    except Exception:  # noqa: BLE001 — progress is best-effort
        logger.warning("Workflow %s: progress emit %s failed", run.id, event.type)


def terminal_state(result: dict[str, Any]) -> str:
    """Map an aggregate status to the run's terminal state (``success`` → ``done``;
    ``partial``/``failed``/``timeout`` pass through)."""
    status = result.get("status") or "failed"
    return "done" if status == "success" else str(status)


async def _drive_run(
    store: WorkflowStore, run: WorkflowRun, consumer: WorkflowConsumer,
    prior_results: list[dict[str, Any]],
) -> None:
    """Run the plan from ``run.cursor`` carrying ``prior_results``; SUSPEND on the
    next deferred step (persist waiting), else finalize + report + persist terminal.
    The shared tail of both entrypoints — ``run_workflow`` (a fresh run from cursor 0)
    and ``deliver_signal`` (resume after a signal, prior_results = the signal-appended
    outcomes)."""
    result = await consumer.run_fn(
        run.household_id, run.node_id, run.steps,
        user_id=run.user_id, goal=run.goal, errand_id=run.id,
        start_index=run.cursor, prior_results=prior_results,
    )
    if isinstance(result, Suspend):
        store.save_waiting(
            run, result.cursor, result.results,
            signal_kind=result.signal_kind, wake_at=result.wake_at,
        )  # suspended on the next deferred step (phone outcome, timer, or approval)
        if result.signal_kind == "approval":
            # A request_replan step paused the run pending a decision. Surface the
            # proposed delta so the consumer can decide (widens_envelope) whether to
            # post a change card and wait, or auto-continue an in-envelope change.
            step_index = result.cursor - 1
            replan_args: dict[str, Any] = {}
            if 0 <= step_index < len(run.steps):
                replan_args = run.steps[step_index].get("args") or {}
            emit_progress(run, ProgressEvent("needs_a_tap", {
                "workflow_id": run.id,
                "step_index": step_index,
                "reason": replan_args.get("reason") or "",
                "add_steps": replan_args.get("add_steps") or [],
            }))
        logger.info("Workflow %s waiting on %s (%s)", run.id, result.signal_kind, result.signal_key)
        return
    emit_progress(run, ProgressEvent("complete", result))
    store.save_terminal(
        run, terminal_state(result),
        None if result.get("status") == "success" else (result.get("message") or "")[:500],
    )
    logger.info("Workflow %s complete → %s", run.id, result.get("status"))


async def run_workflow(workflow_id: str) -> bool:
    """Drive a freshly-created workflow RUN from its cursor — the Run-tap entrypoint
    (``deliver_signal`` is the resume-on-signal entrypoint; both share ``_drive_run``,
    the store, the consumer, and the progress channel). The consumer creates the
    Workflow row (state ``running``, cursor 0) and calls this; the engine runs the
    plan, suspending on a deferred step or landing terminal. Never raises: a crash
    reports an honest failure and lands the run terminal rather than vanishing."""
    if _STORE_FACTORY is None:
        logger.error("run_workflow: no workflow store registered")
        return False

    store = _STORE_FACTORY()
    try:
        run = store.load(workflow_id)
        if run is None:
            logger.error("run_workflow: no run %s", workflow_id)
            return False
        consumer = _CONSUMERS.get(run.kind)
        if consumer is None:
            logger.error("run_workflow: no consumer for kind=%s", run.kind)
            store.save_terminal(run, "failed", "no consumer for this workflow kind")
            return False
        try:
            await _drive_run(store, run, consumer, list(run.results))
            return True
        except Exception:  # noqa: BLE001 — a run crash must not vanish
            logger.exception("Workflow %s run failed", run.id)
            try:
                emit_progress(run, ProgressEvent(
                    "complete", {"status": "failed", "message": "The workflow couldn't run."}))
                store.save_terminal(run, "failed", "run crashed")
            except Exception:  # noqa: BLE001
                logger.warning("Workflow %s: failure report also failed", run.id)
            return True
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001 — closing is best-effort
            logger.warning("run_workflow: store close failed for %s", workflow_id)


# ── SEAM 1: the SIGNAL primitive ──────────────────────────────────────────────
async def deliver_signal(
    workflow_id: str, step: int, kind: str, outcome: Any,
) -> bool:
    """Deliver an external event to a suspended workflow and resume it — the ONE
    primitive that unifies the wakeup sources (a phone outcome today, a timer
    next). Generalizes the errand's ``resume_errand_after_call``:

      1. Load the run; ignore unless it is ``waiting`` on exactly this step.
      2. Atomically claim ``waiting→running`` (only the first trigger proceeds —
         the immediate hook and the safety sweep can both fire).
      3. Ask the DEFERRED handler for ``kind`` to interpret the event into a step
         outcome (done → success, else a failure that fail-fasts the run).
      4. On failure → finalize + report + persist terminal. On success → resume the
         plan from the cursor; suspend again on the next deferred step, else finish.

    Returns True if it claimed and drove the workflow, False on a no-op (not
    waiting / stale / lost the claim race). Never raises: a crash mid-resume
    reports an honest failure and lands the run terminal rather than vanishing."""
    if _STORE_FACTORY is None:
        logger.error("deliver_signal: no workflow store registered")
        return False

    store = _STORE_FACTORY()
    try:
        # One load — the run's steps/cursor/results are stable until we resolve it;
        # the claim only flips state, so the pre-claim snapshot stays authoritative.
        run = store.load(workflow_id)
        if run is None or run.state != "waiting":
            return False  # already resumed / not waiting
        if run.cursor != step + 1:
            return False  # a stale/older signal for this workflow — ignore
        if not store.claim_running(workflow_id):
            return False  # lost the race to the other trigger

        consumer = _CONSUMERS.get(run.kind)
        handler = _deferred_handler(kind)
        if consumer is None or handler is None:
            logger.error("deliver_signal: no consumer/handler for kind=%s signal=%s", run.kind, kind)
            store.save_terminal(run, "failed", "no consumer/handler for this signal")
            return False

        try:
            step_outcome = handler.interpret_signal(outcome, run.steps, step)
            results = list(run.results)
            results.append(step_outcome)

            if not step_outcome.get("success"):
                # Fail-fast: the deferred step failed, so dependent steps don't run.
                result = await consumer.finalize_fn(run.goal, results)
                emit_progress(run, ProgressEvent("complete", result))
                store.save_terminal(run, terminal_state(result), (result.get("message") or "")[:500])
                logger.info("Workflow %s fail-fast after %s signal (%s)", run.id, kind,
                            getattr(outcome, "state", "?"))
                return True

            # The deferred step succeeded → continue the plan from the cursor
            # (suspends again on the next deferred step, else lands terminal).
            await _drive_run(store, run, consumer, results)
            return True
        except Exception:  # noqa: BLE001 — a resume crash must not vanish; leave an honest record
            logger.exception("Workflow %s resume failed on %s signal", run.id, kind)
            try:
                emit_progress(run, ProgressEvent(
                    "complete",
                    {"status": "failed", "message": "The workflow hit a snag after the last step."}))
                store.save_terminal(run, "failed", "resume crashed")
            except Exception:  # noqa: BLE001
                logger.warning("Workflow %s: failure report also failed", run.id)
            return True
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001 — closing is best-effort
            logger.warning("deliver_signal: store close failed for %s", workflow_id)


# Register the built-in capability planes in priority order (node last = fallback).
# The two deferred handlers (phone_call, wait_for/timer) come first — they claim
# specific commands; the server-tool handler then claims registered tools; the node
# handler is the catch-all.
register_handler(PhoneCallHandler())
register_handler(WaitForHandler())
register_handler(ReplanHandler())
register_handler(ServerToolHandler())
register_handler(NodeCommandHandler())
