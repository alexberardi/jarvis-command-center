"""Errand service — turn a goal into a reviewable plan card, then run it.

The Errand Runner (prds/errand-runner.md §2-§3). The flow:

    plan_errand(goal)  →  persist a DRAFT ErrandPlan (its ``steps`` ARE the plan)
    →  push a plan card with Run/Revise/Cancel.

"LLM proposes, card disposes": nothing runs until the user taps **Run**, which a
server-plane callback wires to ``errand_executor.execute_errand`` — the per-step
dispatcher that routes each step to its plane (node command → the node over MQTT;
server tool → in-process) and pushes ONE completion card.

An errand is NOT a node routine: the plan can mix the node's installed commands
with enabled server tools (phone calls, research, memory…) as building blocks, and
each step is dispatched to the right plane at run time. There is no transient
routine — ``ErrandPlan.steps`` is the single source of truth for what runs.

The card is a first-party interactive inbox item built exactly like the phone
call-plan card (``phone_call_service.create_call_plan``): ``metadata`` carries
``interactive_elements`` with ``target: "server"`` so the taps route to the
``server_callback_registry`` (CC plane), NOT the node plane. ``register_errand_callbacks``
wires ``("errand", "approve_errand_plan")`` / ``…refine…`` / ``…discard…``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any

from sqlalchemy.orm import Session

from app.models import ErrandPlan, Workflow
from app.services.errand_planner import (
    build_full_errand_menu,
    plan_errand,
    refine_errand_plan,
    replan_from_progress,
)
from app.services.inbox_notification_service import (
    post_inbox_item_sync,
    update_inbox_item_sync,
)
from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)
from app.services.workflow_engine import (
    ProgressEmitter,
    ProgressEvent,
    WorkflowConsumer,
    WorkflowRun,
    WorkflowStore,
    deliver_signal,
    register_consumer,
    register_workflow_store,
    widens_envelope,
)

logger = logging.getLogger("uvicorn")

# The command + callback names the plan card's Run/Cancel buttons target. Chunk 4
# registers handlers for these pairs in the server_callback_registry; the mobile
# tap routes here because interactive_elements carry target:"server".
ERRAND_CALLBACK_COMMAND = "errand"
ERRAND_APPROVE_CALLBACK = "approve_errand_plan"
ERRAND_DISCARD_CALLBACK = "discard_errand_plan"
ERRAND_REPLAN_CALLBACK = "replan_errand_plan"
ERRAND_REFINE_CALLBACK = "refine_errand_plan"
# Mid-run pause-and-replan: a running errand hit a request_replan step and posted a
# delta card. Approve splices the proposed steps into the RUN and resumes it; Stop
# ends the errand. These target the Workflow (run), not the draft ErrandPlan.
ERRAND_REPLAN_APPROVE_CALLBACK = "approve_replan"
ERRAND_STOP_CALLBACK = "stop_errand"

# Own inbox category so the plan card is filterable and distinct from the
# routine completion card (category="routine").
ERRAND_PLAN_CATEGORY = "errand_plan"


async def _resolve_node_menu(
    node_id: str, household_id: str, speaker_user_id: int | None
) -> list[dict[str, Any]] | None:
    """The full errand menu for a non-voice path (POST /errands, re-plan, refine):
    the target node's live commands (MQTT round-trip) UNION the enabled server
    tools. Server tools are included even if the node can't be reached — they run
    in CC, so a phone-call errand still plans when the node is offline. Returns
    None only when NOTHING is available → the planner falls back to its built-in
    default menu.
    """
    available: list[dict[str, Any]] | None = None
    try:
        from app.api.node_tools import _request_tools_from_node

        report = await _request_tools_from_node(node_id, timeout=10.0)
        if report:
            available = report.get("available_commands")
    except Exception:  # noqa: BLE001 — node sourcing is best-effort; server tools still apply
        logger.warning("Errand: couldn't fetch node %s commands; server tools only", node_id)
    menu = build_full_errand_menu(available, household_id, speaker_user_id)
    return menu or None


def _plan_card_body(summary: str, node_steps: list[dict[str, Any]]) -> str:
    """Human-readable plan for the card body (markdown)."""
    lines = [summary, "", "**Plan:**"]
    for i, step in enumerate(node_steps, start=1):
        lines.append(f"{i}. {step.get('label') or step.get('command')}")
    lines.append("")
    lines.append(
        "Tap **Run** to start it. To change it, tell me what to change above and "
        "tap **Revise**, or tap **Cancel** to discard."
    )
    return "\n".join(lines)


def build_plan_card_metadata(
    row: ErrandPlan, node_steps: list[dict[str, Any]], household_id: str
) -> dict[str, Any]:
    """The plan card's ``metadata`` block (server-plane interactive card).

    ``household_id`` is required — the mobile app copies it into the server-plane
    ``POST /callbacks`` body. ``interactive_elements`` with ``target:"server"``
    are what make Run/Cancel dispatch to the ``server_callback_registry`` instead
    of a node. ``data`` round-trips ``plan_id`` + ``revision`` so chunk 4's
    handler can load the row and reject a stale card.
    """
    # Run/Cancel carry only the identity. The mobile merges the typed instruction
    # ONLY into a button whose data already has the "instruction" key — so the
    # Revise button seeds it, and Run/Cancel omit it (a tap on them ignores an
    # unsent instruction and is never blocked by the required-field check).
    ident = {"plan_id": row.id, "revision": row.revision}
    revise = {**ident, "instruction": ""}
    return {
        "household_id": household_id,
        "plan_id": row.id,
        "revision": row.revision,
        "steps": node_steps,  # structured read-only view of the steps
        # A free-text "tell it what to change" box → Revise re-plans the current
        # plan conversationally. NB: do NOT set editor_schema (>2 disables buttons).
        "editable_fields": [
            {
                "label": "Tell me what to change",
                "initial": "",
                "data_key": "instruction",
                "input_type": "text",
                "required": True,
            },
        ],
        "interactive_elements": [
            {
                "id": "run-errand",
                "label": "Run",
                "command": ERRAND_CALLBACK_COMMAND,
                "callback": ERRAND_APPROVE_CALLBACK,
                "target": "server",
                "data": ident,
            },
            {
                "id": "revise-errand",
                "label": "Revise",
                "command": ERRAND_CALLBACK_COMMAND,
                "callback": ERRAND_REFINE_CALLBACK,
                "target": "server",
                "data": revise,
            },
            {
                "id": "cancel-errand",
                "label": "Cancel",
                "command": ERRAND_CALLBACK_COMMAND,
                "callback": ERRAND_DISCARD_CALLBACK,
                "target": "server",
                "data": ident,
            },
        ],
    }


def _post_errand_plan_card(
    row: ErrandPlan,
    node_steps: list[dict[str, Any]],
    household_id: str,
    user_id: int | None,
) -> str | None:
    """Post the plan card, OR — if this plan already has a card (``inbox_item_id``)
    — UPDATE it IN PLACE so a Revise/re-plan refreshes the card the user is looking
    at instead of posting a duplicate. Returns the card's inbox item id (unchanged
    on update, new on create), or None on failure. Best-effort: the draft is
    already committed, so a lost card never fails the errand.
    """
    summary = row.summary or row.goal
    title = f"🗒️ Errand plan: {summary}"
    body = _plan_card_body(summary, node_steps)
    metadata = build_plan_card_metadata(row, node_steps, household_id)

    if row.inbox_item_id:
        # In-place update (same card id) — the whole point of storing the id.
        if update_inbox_item_sync(
            item_id=row.inbox_item_id, household_id=household_id,
            title=title, summary=summary, body=body,
            category=ERRAND_PLAN_CATEGORY, metadata=metadata,
        ):
            return row.inbox_item_id
        # The card was deleted/gone — fall through to post a fresh one.

    return post_inbox_item_sync(
        household_id=household_id,
        title=title,
        summary=summary,
        body=body,
        category=ERRAND_PLAN_CATEGORY,
        metadata=metadata,
        user_id=user_id,
        push=True,
        target_type="user" if user_id is not None else "household",
    )


# ── Mid-run pause-and-replan ──────────────────────────────────────────────────
# A running errand can hit a ``request_replan`` step (the engine's ``approval``
# signal): it suspends the RUN, the emitter checks ``widens_envelope`` and either
# posts a delta card for a one-tap decision (widen) or auto-continues silently
# (in-envelope). Approve splices the proposal into the run and resumes it.


def _replan_card_body(title: str, reason: str, add_steps: list[dict[str, Any]]) -> str:
    """Markdown body for a mid-run delta card."""
    lines = [f"While running **{title}**, I have a change to propose:", ""]
    if reason:
        lines += [reason, ""]
    lines.append("**Proposed additional steps:**")
    for i, step in enumerate(add_steps, start=1):
        lines.append(f"{i}. {step.get('label') or step.get('command')}")
    lines += ["", "Tap **Approve** to include these and continue, or **Stop** to end the errand here."]
    return "\n".join(lines)


def build_replan_card_metadata(
    workflow_id: str, revision: int, add_steps: list[dict[str, Any]], household_id: str
) -> dict[str, Any]:
    """The delta card's ``metadata`` (server-plane interactive card). ``data`` round-
    trips ``workflow_id`` + ``revision`` so Approve loads the RUN and rejects a stale
    card (a change the run already moved past). Approve/Stop only."""
    ident = {"workflow_id": workflow_id, "revision": revision}
    return {
        "household_id": household_id,
        "workflow_id": workflow_id,
        "revision": revision,
        "steps": add_steps,  # the proposed additions, read-only
        "interactive_elements": [
            {"id": "approve-replan", "label": "Approve", "command": ERRAND_CALLBACK_COMMAND,
             "callback": ERRAND_REPLAN_APPROVE_CALLBACK, "target": "server", "data": ident},
            {"id": "stop-errand", "label": "Stop", "command": ERRAND_CALLBACK_COMMAND,
             "callback": ERRAND_STOP_CALLBACK, "target": "server", "data": ident},
        ],
    }


def _apply_workflow_replan(db: Session, workflow_id: str, at_step: int) -> bool:
    """Splice the ``request_replan`` step's proposed ``add_steps`` into the RUN's
    plan, right AFTER that step (index ``at_step+1``), and bump ``revision`` (so the
    delta card's Approve becomes stale). Shared by the human-approve and in-envelope
    auto-continue paths. Returns True if it spliced."""
    wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if wf is None:
        return False
    steps = json.loads(wf.steps or "[]")
    if not (0 <= at_step < len(steps)):
        return False
    add = (steps[at_step].get("args") or {}).get("add_steps") or []
    if not isinstance(add, list):
        add = []
    steps[at_step + 1:at_step + 1] = add
    wf.steps = json.dumps(steps)
    wf.revision = (wf.revision or 1) + 1
    db.commit()
    return True


def _post_replan_card(
    wf: Workflow, reason: str, add_steps: list[dict[str, Any]], db: Session
) -> None:
    """Post (or in-place update) the delta card for a WAITING run and store its id on
    ``wf.inbox_item_id``. Best-effort — the run is already durably waiting."""
    title_summary = wf.title or wf.goal or "your errand"
    title = f"🔀 Errand update: {title_summary}"
    summary = reason or "I have a change to propose."
    body = _replan_card_body(title_summary, reason, add_steps)
    metadata = build_replan_card_metadata(wf.id, wf.revision, add_steps, wf.household_id)
    item_id: str | None = None
    if wf.inbox_item_id and update_inbox_item_sync(
        item_id=wf.inbox_item_id, household_id=wf.household_id,
        title=title, summary=summary, body=body,
        category=ERRAND_PLAN_CATEGORY, metadata=metadata,
    ):
        item_id = wf.inbox_item_id
    else:
        item_id = post_inbox_item_sync(
            household_id=wf.household_id, title=title, summary=summary, body=body,
            category=ERRAND_PLAN_CATEGORY, metadata=metadata, user_id=wf.user_id,
            push=True, target_type="user" if wf.user_id is not None else "household",
        )
    if item_id and item_id != wf.inbox_item_id:
        wf.inbox_item_id = item_id
        try:
            db.commit()
        except Exception:  # noqa: BLE001 — storing the card id is best-effort
            db.rollback()


async def _auto_continue_replan(workflow_id: str, step_index: int) -> None:
    """In-envelope replan → splice the delta + resume WITHOUT a human tap. Runs as a
    detached task off the (sync) progress emitter. The run is durably 'waiting' on
    approval; if this task is lost to a restart before it fires the run parks — a rare
    gap (no approval sweep yet; single-instance CC). Never raises out."""
    from app.db import get_session_local

    try:
        db = get_session_local()()
        try:
            _apply_workflow_replan(db, workflow_id, step_index)
        finally:
            db.close()
        await deliver_signal(workflow_id, step_index, "approval", SimpleNamespace(state="approved"))
    except Exception:  # noqa: BLE001 — a detached auto-continue must not crash the loop
        logger.exception("Errand %s: in-envelope auto-continue failed", workflow_id)


def _handle_replan_needs_a_tap(run: WorkflowRun, data: dict[str, Any]) -> None:
    """The engine paused a run on a ``request_replan`` checkpoint (SEAM 3,
    ``needs_a_tap``). The proposed delta is resolved CURSOR-AWARE (re-plan from the
    goal + results so far) unless the step pre-baked ``add_steps``, then
    ``widens_envelope`` decides: a widening change posts a delta card and waits for a
    tap; an in-envelope change auto-continues silently. All of that needs an LLM call
    + DB work, so it runs as a detached task off this sync emitter hook."""
    step_index = data.get("step_index")
    if step_index is None:
        return
    static_add = data.get("add_steps") or []
    reason = str(data.get("reason") or "")
    try:
        asyncio.get_running_loop().create_task(
            _resolve_and_decide_replan(run.id, step_index, reason, list(static_add))
        )
    except RuntimeError:
        logger.warning("Errand %s: no running loop to resolve a replan checkpoint", run.id)


def _store_replan_add_steps(
    workflow_id: str, step_index: int, add_steps: list[dict[str, Any]], reason: str
) -> None:
    """Persist the resolved continuation onto the checkpoint step's ``args.add_steps``
    (and ``args.reason``) so the card, the Approve tap, and the auto-continue path all
    splice the SAME steps — ``_apply_workflow_replan`` reads them from there."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
        if wf is None:
            return
        steps = json.loads(wf.steps or "[]")
        if not (0 <= step_index < len(steps)):
            return
        args = steps[step_index].get("args") or {}
        args["add_steps"] = add_steps
        if reason:
            args.setdefault("reason", reason)
        steps[step_index]["args"] = args
        wf.steps = json.dumps(steps)
        db.commit()
    except Exception:  # noqa: BLE001 — best-effort; a failed store degrades to auto-continue
        logger.exception("Errand %s: couldn't store replan add_steps", workflow_id)
        db.rollback()
    finally:
        db.close()


async def _resolve_and_decide_replan(
    workflow_id: str, step_index: int, reason: str, static_add: list[dict[str, Any]]
) -> None:
    """Resolve a checkpoint's continuation and decide pause-vs-continue. Runs detached
    off the emitter. Cursor-aware: re-plans from the goal + the results collected so
    far. Never raises — on ANY failure it auto-continues so the errand can't hang
    parked on the checkpoint forever."""
    from app.db import get_session_local

    try:
        db = get_session_local()()
        try:
            wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
            if wf is None or wf.state != "waiting":
                return  # already resumed / cancelled
            goal = wf.goal or ""
            steps = json.loads(wf.steps or "[]")
            results = json.loads(wf.results_json or "[]")
            node_id, household_id, user_id = wf.node_id, wf.household_id, wf.user_id
        finally:
            db.close()

        add_steps = list(static_add)
        if not add_steps:
            # Dynamic: plan the rest from what actually happened up to the checkpoint.
            from app.core.llm_proxy_client import LLMProxyClient

            menu = await _resolve_node_menu(node_id, household_id, user_id)
            try:
                plan = await replan_from_progress(
                    goal, steps[:step_index], results, reason, LLMProxyClient(), menu=menu
                )
                add_steps = plan.routine_steps()
            except ValueError:
                logger.info("Errand %s: replan produced no steps — continuing", workflow_id)
                add_steps = []

        # Persist so card/approve/auto-continue splice the same continuation.
        _store_replan_add_steps(workflow_id, step_index, add_steps, reason)

        amended = list(steps) + list(add_steps)
        if add_steps and widens_envelope(steps, amended):
            db = get_session_local()()
            try:
                wf = db.query(Workflow).filter(Workflow.id == workflow_id).first()
                if wf is not None:
                    _post_replan_card(wf, reason, add_steps, db)
            finally:
                db.close()
        else:
            # In-envelope, or nothing to add → splice (if any) + resume, no tap.
            await _auto_continue_replan(workflow_id, step_index)
    except Exception:  # noqa: BLE001 — never leave the run parked on a failed replan
        logger.exception("Errand %s: replan resolution failed; continuing", workflow_id)
        try:
            await _auto_continue_replan(workflow_id, step_index)
        except Exception:  # noqa: BLE001
            logger.exception("Errand %s: replan fallback continue also failed", workflow_id)


async def create_errand_plan(
    db: Session,
    household_id: str,
    node_id: str,
    goal: str,
    user_id: int | None = None,
    menu: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> ErrandPlan:
    """Plan an errand from a goal, persist it as a DRAFT, and post its plan card.

    Returns the committed DRAFT ``ErrandPlan`` row. Raises ``ValueError`` (from
    ``plan_errand``) when the goal can't be turned into a usable plan — the caller
    turns that into a "couldn't plan that" response rather than a broken plan.
    """
    if llm_client is None:
        from app.core.llm_proxy_client import LLMProxyClient

        llm_client = LLMProxyClient()

    # Plan over the errand's full building-block set: the voice path passes a menu
    # built from the live conversation (node commands ∪ server tools); otherwise
    # source it here (node MQTT round-trip ∪ server tools). None → the planner's
    # built-in default menu.
    if menu is None:
        menu = await _resolve_node_menu(node_id, household_id, user_id)
    plan = await plan_errand(goal, llm_client, menu=menu)  # raises ValueError on no usable plan
    node_steps = plan.routine_steps()  # [{command, args:{k:v}, label}] — the execution source

    # Persist the DRAFT plan. ``steps`` IS the plan — the per-step executor
    # dispatches each one to its plane (node command / server tool) at Run time;
    # there is no transient routine to pull.
    row = ErrandPlan(
        household_id=household_id,
        user_id=user_id,
        node_id=node_id,
        goal=goal,
        summary=plan.summary,
        steps=json.dumps(node_steps),
        state="draft",
        revision=1,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # Plan card — best-effort; the draft is already committed. Store its inbox
    # item id so a later Revise updates THIS card in place.
    try:
        card_id = _post_errand_plan_card(row, node_steps, household_id, user_id)
        if card_id and card_id != row.inbox_item_id:
            row.inbox_item_id = card_id
            db.commit()
    except Exception:  # noqa: BLE001 — a failed card must not fail the errand
        logger.exception("Errand plan card push failed for %s", row.id)

    logger.info(
        "Errand plan %s drafted (household=%s node=%s steps=%d)",
        row.id, household_id, node_id, len(node_steps),
    )
    return row


def _post_couldnt_plan_card(
    household_id: str, goal: str, user_id: int | None, summary: str, body: str
) -> None:
    """Best-effort 'couldn't plan that' card so a run_errand user isn't left
    hanging after the spoken ack. Never raises."""
    try:
        post_inbox_item_sync(
            household_id=household_id,
            title="🗒️ Couldn't plan that errand",
            summary=summary,
            body=body,
            category=ERRAND_PLAN_CATEGORY,
            metadata={"household_id": household_id},
            user_id=user_id,
            target_type="user" if user_id is not None else "household",
            push=True,
        )
    except Exception:  # noqa: BLE001 — the failure card is itself best-effort
        logger.warning("Couldn't-plan card failed for goal=%r", goal)


async def draft_errand_plan_detached(
    household_id: str, node_id: str, goal: str, user_id: int | None = None,
    menu: list[dict[str, Any]] | None = None,
) -> None:
    """Draft an errand plan on a FRESH session — for fire-and-forget callers.

    The ``run_errand`` voice tool fires this detached (it has no request-scoped DB
    session, and one would be closed by the time the task runs — cf. phone's
    ``create_call_plan`` opening its own session). It passes ``menu`` built from
    the live conversation's node commands. Because the tool already spoke "I'll
    send it to your phone," ANY failure posts a card so the user is never left
    hanging — the phone stack's never-vanish rule (create_call_plan:543).
    """
    from app.db import get_session_local

    db = get_session_local()()
    try:
        await create_errand_plan(db, household_id, node_id, goal, user_id=user_id, menu=menu)
    except ValueError as e:
        # The planner couldn't turn the goal into usable steps → rephrase hint.
        logger.info("Errand planning produced no usable plan for goal=%r: %s", goal, e)
        _post_couldnt_plan_card(
            household_id, goal, user_id,
            summary=f'I couldn\'t turn "{goal}" into a plan I can run.',
            body=f'I couldn\'t turn "{goal}" into a plan I can run — try rephrasing it.',
        )
    except Exception:  # noqa: BLE001 — infra failure (planner/LLM proxy down, DB error)
        # The user was already told a card is coming; surface an honest "try
        # again" card rather than vanishing.
        logger.exception("Detached errand draft failed for goal=%r", goal)
        _post_couldnt_plan_card(
            household_id, goal, user_id,
            summary="I hit a snag planning that errand.",
            body="I hit a snag putting that errand together — try again in a bit.",
        )
    finally:
        db.close()


# ── Server-plane callbacks: the plan card's Run / Cancel taps ─────────────────
#
# Registered under (command="errand", callback=...) so the card's
# interactive_elements (target:"server") route here via server_callback_registry
# / app.api.callbacks. Each handler is self-contained (own DB session, no shared
# state) — the callback dispatcher runs them in a background task.


# Status → (icon, headline) for the errand completion card. Honest on EVERY
# terminal state — an errand you tapped Run on must tell you how it finished,
# including partial/failed, never stay silent.
_ERRAND_STATUS_CARD = {
    "success": ("✅", "done"),
    "partial": ("⚠️", "finished with issues"),
    "failed": ("⚠️", "couldn't finish"),
    "timeout": ("⏱️", "didn't finish"),
}


def _post_errand_completion_card(
    household_id: str, title_summary: str, result: dict[str, Any], user_id: int | None
) -> None:
    """Post ONE completion card for a run errand (headless feedback channel).

    Best-effort — a notification failure must never crash the callback (mirrors
    routines' ``_notify_routine_complete``). Targets the initiating speaker when
    known, else the whole household. ``result`` is the ``execute_errand`` aggregate
    (``{status, message, ...}``); ``message`` is the plain-English step roll-up.
    """
    try:
        status = str(result.get("status") or "failed")
        icon, headline = _ERRAND_STATUS_CARD.get(status, _ERRAND_STATUS_CARD["failed"])
        summary = result.get("message") or "The errand finished."
        post_inbox_item_sync(
            household_id=household_id,
            title=f"{icon} Errand {headline}: {title_summary}",
            summary=summary,
            body=summary,
            category="errand",
            metadata={"household_id": household_id, "status": status},
            user_id=user_id,
            target_type="user" if user_id is not None else "household",
            push=True,
        )
    except Exception:  # noqa: BLE001 — a completion card must never fail the run
        logger.warning("Errand completion card failed for household %s", household_id)


def _revisions_match(card_revision: Any, row_revision: int) -> bool:
    """True unless the card carries a definitively-older revision.

    A missing/unparseable revision falls through to the state guard rather than
    hard-rejecting — the JSON round-trip through the mobile client may stringify
    the int, so compare leniently.
    """
    try:
        return int(card_revision) == row_revision
    except (TypeError, ValueError):
        return True


async def _handle_approve_errand_plan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Run tap → spawn a durable Workflow RUN from the draft and drive it.

    The run/instance split: the ``ErrandPlan`` is the DRAFT/DEFINITION; tapping Run
    creates a ``Workflow`` (the RUN), links it back (``workflow_id``), marks the
    draft ``launched``, and hands the run to the engine (``run_workflow``). The
    engine owns every terminal path — it dispatches each step to its plane, SUSPENDS
    on a phone step (state ``waiting`` on the Workflow, no completion card yet), and
    on completion posts exactly one card via the errand progress emitter (SEAM 3).
    Single-use + stale-card guarded on the DRAFT. Anti-vanishing lives in the engine.
    """
    from app.db import get_session_local
    from app.services.workflow_engine import run_workflow

    db = get_session_local()()
    try:
        plan_id = ctx.data.get("plan_id")
        if not plan_id:
            return ServerCallbackResult(success=False, error="Missing plan_id")
        row = (
            db.query(ErrandPlan)
            .filter(ErrandPlan.id == plan_id, ErrandPlan.household_id == ctx.household_id)
            .first()
        )
        if row is None:
            return ServerCallbackResult(success=False, error="Errand plan not found")

        # Single-use: a second tap on an already-launched/cancelled plan is a
        # friendly no-op card, not an error (mirrors the phone confirm handler).
        if row.state != "draft":
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "Errand already handled",
                    "summary": f"This errand is already {row.state}.",
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        if not _revisions_match(ctx.data.get("revision"), row.revision):
            return ServerCallbackResult(
                success=False,
                error="This plan was updated — open the latest card to run it.",
            )

        node_steps = json.loads(row.steps or "[]")
        if not node_steps:
            row.state = "failed"
            row.error = "no steps to run"
            db.commit()
            return ServerCallbackResult(success=False, error="This errand has no steps to run.")

        # Spawn the RUN (Workflow) from this draft and link it. The workflow carries
        # the execution state (running/waiting/…); the draft just points at it.
        workflow = Workflow(
            kind="errand",
            household_id=ctx.household_id,
            user_id=row.user_id,
            node_id=row.node_id,
            goal=row.goal or row.summary or "",
            title=row.summary or row.goal,
            steps=row.steps,
            cursor=0,
            results_json="[]",
            state="running",
        )
        db.add(workflow)
        db.flush()  # assign workflow.id before we link + run
        row.workflow_id = workflow.id
        row.state = "launched"
        row.confirmed_at = datetime.utcnow()
        db.commit()
        workflow_id = workflow.id

        # Drive the run via the engine. It owns suspend/complete + the completion
        # card (SEAM 3) and never raises — a crash lands the run terminal with an
        # honest card, so a tap on Run never vanishes.
        await run_workflow(workflow_id)
        logger.info("Errand %s launched workflow %s", row.id, workflow_id)
        return ServerCallbackResult(success=True)
    finally:
        db.close()


async def resume_errand_after_call(session: Any) -> None:
    """A phone call linked to an errand reached a terminal state — resume the errand.

    This is the errand's binding of the engine's SIGNAL primitive: a terminal phone
    call is one external event, so it becomes ``deliver_signal(errand_id, step,
    "phone_call", session)``. The engine loads the run, atomically claims it, asks
    the phone handler to interpret the outcome (``done`` → continue from the cursor;
    ``failed``/``declined``/``expired`` → fail-fast), and reports via the errand's
    progress emitter. Idempotent (called from the phone finalization hook AND the
    safety sweep) — the atomic claim serializes the two.
    """
    errand_id = getattr(session, "errand_id", None)
    if not errand_id or session.state not in ("done", "failed", "declined", "expired"):
        return
    step = session.errand_step if session.errand_step is not None else 0
    await deliver_signal(errand_id, step, "phone_call", session)


# A wait-and-decide errand parks in 'waiting' until its call finishes. If the call
# NEVER reaches a terminal state (gateway dropped a confirmed dial job, worker down),
# the errand would hang forever — so after this long we give up and fail-fast it.
_ERRAND_WAIT_DEADLINE = timedelta(minutes=60)
_TERMINAL_CALL_STATES = ("done", "failed", "declined", "expired")


async def recover_orphaned_running_workflows() -> int:
    """STARTUP recovery (run ONCE from the app lifespan, before the periodic sweep).

    Any Workflow still in 'running' at boot was orphaned by the previous process — a
    crash or deploy mid-drive — and nothing is driving it now (CC is single-instance).
    Land it terminal with an honest 'interrupted' card rather than leaving it stuck
    forever with no feedback. This is the general, kind-AGNOSTIC orphan recovery (it
    covers phone waits, timer waits, and a crash during a run's very first drive).

    Doing this at STARTUP is what makes it correct and race-free: a runtime recover
    can't tell a dead driver from a merely slow one (no heartbeat) and would re-drive
    a live resume, double-executing steps. At boot there is no live driver, so failing
    every 'running' run is unambiguous. A run genuinely 'waiting' on a signal survives
    a restart untouched and resumes via its signal/timer — only actively-driving runs
    are orphans. (Multi-instance CC would need a lease instead; single-instance today.)
    """
    from app.db import get_session_local
    from app.services.workflow_engine import (
        ProgressEvent,
        WorkflowRun,
        emit_progress,
        terminal_state,
    )

    db = get_session_local()()
    try:
        stuck = db.query(Workflow).filter(Workflow.state == "running").all()
        for wf in stuck:
            results = json.loads(wf.results_json or "[]")
            passed = sum(1 for r in results if r.get("success"))
            result = {
                "status": "partial" if passed else "failed",
                "passed": passed,
                "failed": max(len(results) - passed, 0),
                "message": "This errand was interrupted by a restart and couldn't be finished.",
                "results": results,
            }
            run = WorkflowRun(
                id=wf.id, kind=wf.kind, household_id=wf.household_id, node_id=wf.node_id,
                user_id=wf.user_id, goal=wf.goal or "", title=wf.title or wf.goal or "your errand",
                steps=[], cursor=wf.cursor, results=results, state=wf.state,
            )
            emit_progress(run, ProgressEvent("complete", result))  # honest card via the consumer
            wf.state = terminal_state(result)  # partial | failed
            wf.error = "interrupted by a restart"
        db.commit()
        if stuck:
            logger.warning("Startup recovery: failed %d orphaned 'running' workflow(s)", len(stuck))
        return len(stuck)
    except Exception:  # noqa: BLE001 — recovery is best-effort; don't block startup
        logger.exception("Startup workflow recovery failed")
        db.rollback()
        return 0
    finally:
        db.close()


async def resume_waiting_errands() -> int:
    """Safety sweep (runs every 20s from the app lifespan). Operates on WORKFLOW RUNS
    waiting on a PHONE call. Two jobs, idempotent (the atomic waiting→running claim in
    ``deliver_signal`` serializes with the immediate phone hook, so nothing double-runs):

    1. RESUME 'waiting' runs whose linked call reached a terminal state (also the
       'declined'/'expired' paths that skip post_phone_session_event).
    2. TIME OUT 'waiting' runs whose call never terminates (confirmed-but-never-
       dialed, etc.) past the deadline — fail-fast so the run never hangs forever.

    Orphaned-'running' recovery is NOT here — it's race-free STARTUP recovery
    (``recover_orphaned_running_workflows``); a runtime recover would re-drive a slow
    live resume. Timer waits are resumed by ``resume_due_timer_workflows``. The phone
    session links to the WORKFLOW id (``PhoneCallSession.errand_id`` holds the run id)
    at the run's current cursor step.
    """
    from app.db import get_session_local
    from app.models import PhoneCallSession

    now = datetime.utcnow()

    # Scan 'waiting' runs.
    scan_db = get_session_local()()
    try:
        waiting = scan_db.query(Workflow).filter(Workflow.state == "waiting").all()
    finally:
        scan_db.close()

    resumed = 0
    for row in waiting:
        look = get_session_local()()
        try:
            session = (
                look.query(PhoneCallSession)
                .filter(PhoneCallSession.errand_id == row.id,
                        PhoneCallSession.errand_step == max(row.cursor - 1, 0))
                .order_by(PhoneCallSession.created_at.desc())
                .first()
            )
            trigger = None
            if session is not None and session.state in _TERMINAL_CALL_STATES:
                trigger = session  # loaded ORM row — resume reads its columns
            elif (session is not None and session.created_at is not None
                  and session.created_at < now - _ERRAND_WAIT_DEADLINE):
                # The call never finished — synthesize a failure so the run gives up.
                trigger = SimpleNamespace(
                    id=session.id, errand_id=session.errand_id, errand_step=session.errand_step,
                    state="failed", contact_name=session.contact_name,
                    error_message="the call wasn't completed in time",
                    household_id=session.household_id, user_id=session.user_id,
                )
        finally:
            look.close()
        if trigger is not None:
            await resume_errand_after_call(trigger)
            resumed += 1
    if resumed:
        logger.info("Workflow resume sweep: resumed %d waiting run(s)", resumed)
    return resumed


async def resume_due_timer_workflows() -> int:
    """Timer wakeup sweep (runs alongside the phone sweep from the app lifespan).
    Resumes any run WAITING on a ``wait_for`` timer whose ``wake_at`` has passed —
    through the SAME ``deliver_signal`` primitive the phone outcome uses. This is the
    engine's SECOND signal source (errand-runner PRD §2.5): the timer wakeup that
    proves the engine is general, not phone-specific. Idempotent (the atomic
    waiting→running claim serializes it)."""
    from app.db import get_session_local

    now = datetime.utcnow()
    scan = get_session_local()()
    try:
        due = (
            scan.query(Workflow)
            .filter(
                Workflow.state == "waiting",
                Workflow.waiting_on == "timer",
                Workflow.wake_at.isnot(None),
                Workflow.wake_at <= now,
            )
            .all()
        )
        # Snapshot (id, cursor) so we don't hold the scan session while resuming.
        pending = [(w.id, w.cursor) for w in due]
    finally:
        scan.close()

    fired = 0
    for wf_id, cursor in pending:
        step = max(cursor - 1, 0)
        # The timer has no external session; a bare marker is enough — the timer
        # handler's interpret_signal ignores it and returns a plain "wait finished".
        await deliver_signal(wf_id, step, "timer", SimpleNamespace(state="fired"))
        fired += 1
    if fired:
        logger.info("Workflow timer sweep: fired %d due wait(s)", fired)
    return fired


def _handle_discard_errand_plan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Cancel tap → discard the errand. A DRAFT is simply marked cancelled; a
    LAUNCHED errand whose RUN is mid-call (the Workflow is ``waiting``) is genuinely
    cancelled AND its pending call declined so it never dials."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        plan_id = ctx.data.get("plan_id")
        row = (
            db.query(ErrandPlan)
            .filter(ErrandPlan.id == plan_id, ErrandPlan.household_id == ctx.household_id)
            .first()
        )
        if row is None:
            return ServerCallbackResult(success=True)  # already gone — idempotent
        if row.state == "draft":
            row.state = "cancelled"
            db.commit()
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "🗑️ Errand discarded",
                    "summary": row.summary or row.goal,
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        if row.state == "launched" and row.workflow_id:
            # The RUN carries the execution state now. Only a WAITING run is safe to
            # cancel: it's parked (no active _drive_run), so setting 'cancelled'
            # sticks and the next deliver_signal can't claim it (claim is from
            # 'waiting' only). A 'running' run is actively executing in another
            # session whose save_terminal would overwrite a cancel — so we DON'T
            # cancel it here (a stuck 'running' is recovered→'waiting' by the sweep,
            # then cancellable). Anything else already finished.
            wf = db.query(Workflow).filter(Workflow.id == row.workflow_id).first()
            if wf is not None and wf.state == "waiting":
                wf.state = "cancelled"
                row.state = "cancelled"
                db.commit()
                declined = _decline_pending_workflow_calls(db, wf.id)
                note = (" I won't place the remaining call(s)." if declined
                        else " A call already in progress may still finish.")
                return ServerCallbackResult(
                    success=True,
                    context_data={"inbox": {
                        "title": "🗑️ Errand cancelled",
                        "summary": (row.summary or row.goal) + note,
                        "metadata": {"household_id": ctx.household_id},
                    }},
                )
            # Running (actively executing) or already terminal → honest no-op.
            state_word = wf.state if wf is not None else "done"
        else:
            # A plan directly in cancelled/expired (never launched): honest no-op.
            state_word = row.state
        return ServerCallbackResult(
            success=True,
            context_data={"inbox": {
                "title": "Errand already handled",
                "summary": f"This errand is already {state_word} — nothing to discard.",
                "metadata": {"household_id": ctx.household_id},
            }},
        )
    finally:
        db.close()


def _decline_pending_workflow_calls(db: Session, workflow_id: str) -> bool:
    """Decline any not-yet-dialed call this workflow placed (draft/confirmed
    sessions), so cancelling the run actually stops the call before it rings. A call
    already dialing can't be un-dialed here, but the run is now 'cancelled' so
    ``deliver_signal`` won't continue it either. Best-effort; returns True if it
    declined at least one pending session."""
    try:
        from app.models import PhoneCallSession
        from app.services.phone_call_service import transition

        sessions = (
            db.query(PhoneCallSession)
            .filter(PhoneCallSession.errand_id == workflow_id,
                    PhoneCallSession.state.in_(("draft", "confirmed")))
            .all()
        )
        declined_any = False
        for s in sessions:
            if transition(s, "declined"):  # legal only from draft
                s.error_message = "errand cancelled"
                declined_any = True
        db.commit()
        return declined_any
    except Exception:  # noqa: BLE001 — cancel must not fail on a cleanup blip
        logger.exception("Workflow %s: couldn't decline pending calls on cancel", workflow_id)
        db.rollback()
        return False


async def _handle_approve_replan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Approve tap on a mid-run delta card → splice the proposed steps into the RUN
    and resume it. Guards: the Workflow is WAITING on ``approval`` and the card's
    revision matches (a stale card — one the run already moved past — is rejected).
    The splice happens BEFORE ``deliver_signal`` so the resumed drive sees the
    amended plan. Never raises: deliver_signal owns anti-vanishing."""
    from app.db import get_session_local

    db = get_session_local()()
    step_index: int
    workflow_id: Any
    try:
        workflow_id = ctx.data.get("workflow_id")
        if not workflow_id:
            return ServerCallbackResult(success=False, error="Missing workflow_id")
        wf = (
            db.query(Workflow)
            .filter(Workflow.id == workflow_id, Workflow.household_id == ctx.household_id)
            .first()
        )
        if wf is None:
            return ServerCallbackResult(success=False, error="Errand not found")
        # Single-use: a second tap once the run has moved on is a friendly no-op.
        if wf.state != "waiting" or wf.waiting_on != "approval":
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "Errand already handled",
                    "summary": "That change was already handled.",
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        if not _revisions_match(ctx.data.get("revision"), wf.revision):
            return ServerCallbackResult(
                success=False,
                error="This change was already updated — open the latest card.",
            )
        step_index = wf.cursor - 1  # the request_replan step the run is parked on
        if not _apply_workflow_replan(db, workflow_id, step_index):
            return ServerCallbackResult(success=False, error="Couldn't apply the change.")
    finally:
        db.close()

    # Resume over the amended plan (fresh store inside deliver_signal).
    await deliver_signal(workflow_id, step_index, "approval", SimpleNamespace(state="approved"))
    logger.info("Errand %s replan approved → resumed at step %s", workflow_id, step_index + 1)
    return ServerCallbackResult(success=True)


def _handle_stop_errand(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Stop tap on a mid-run delta card → cancel the errand here. Only a WAITING run
    is safe to cancel (parked, no active drive); its pending downstream call is
    declined so nothing dials after the stop. Mirrors ``_handle_discard_errand_plan``
    for the launched-waiting case."""
    from app.db import get_session_local

    db = get_session_local()()
    try:
        workflow_id = ctx.data.get("workflow_id")
        wf = (
            db.query(Workflow)
            .filter(Workflow.id == workflow_id, Workflow.household_id == ctx.household_id)
            .first()
        )
        if wf is None:
            return ServerCallbackResult(success=True)  # gone — idempotent
        if wf.state == "waiting":
            wf.state = "cancelled"
            db.commit()
            declined = _decline_pending_workflow_calls(db, wf.id)
            db.query(ErrandPlan).filter(ErrandPlan.workflow_id == wf.id).update(
                {"state": "cancelled"}
            )
            db.commit()
            note = " I won't place the remaining call(s)." if declined else ""
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "🗑️ Errand stopped",
                    "summary": (wf.title or wf.goal or "The errand") + " stopped." + note,
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        return ServerCallbackResult(
            success=True,
            context_data={"inbox": {
                "title": "Errand already handled",
                "summary": f"This errand is already {wf.state}.",
                "metadata": {"household_id": ctx.household_id},
            }},
        )
    finally:
        db.close()


async def _handle_replan_errand_plan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Update-plan tap → re-plan the errand from the edited goal, re-post the card.

    The mobile merges the edited goal into this button's data (data_key "goal").
    We re-run the planner, update the draft's steps IN PLACE, bump the revision
    (so the old card's Run is rejected as stale), and post the refreshed plan card
    for another review pass.
    """
    from app.core.llm_proxy_client import LLMProxyClient
    from app.db import get_session_local

    db = get_session_local()()
    try:
        plan_id = ctx.data.get("plan_id")
        new_goal = (ctx.data.get("goal") or "").strip()
        if not plan_id:
            return ServerCallbackResult(success=False, error="Missing plan_id")
        if not new_goal:
            return ServerCallbackResult(success=False, error="The errand goal can't be empty.")
        row = (
            db.query(ErrandPlan)
            .filter(ErrandPlan.id == plan_id, ErrandPlan.household_id == ctx.household_id)
            .first()
        )
        if row is None:
            return ServerCallbackResult(success=False, error="Errand plan not found")
        if row.state != "draft":
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "Errand already handled",
                    "summary": f"This errand is already {row.state} — it can't be edited.",
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        if not _revisions_match(ctx.data.get("revision"), row.revision):
            return ServerCallbackResult(
                success=False,
                error="This plan was updated — open the latest card to edit it.",
            )

        menu = await _resolve_node_menu(row.node_id, ctx.household_id, row.user_id)
        try:
            plan = await plan_errand(new_goal, LLMProxyClient(), menu=menu)
        except ValueError as e:
            logger.info("Re-plan produced no usable plan for goal=%r: %s", new_goal, e)
            return ServerCallbackResult(
                success=False,
                error=f'I couldn\'t turn "{new_goal}" into a plan I can run — try rephrasing it.',
            )
        except Exception:  # noqa: BLE001 — infra (LLM proxy down/timeout); don't leak the raw error
            # The row is untouched here (plan_errand runs before any mutation), so
            # the old plan card stays valid — return a friendly, retryable message.
            logger.exception("Re-plan planner call failed for goal=%r", new_goal)
            return ServerCallbackResult(
                success=False, error="I hit a snag re-planning that — try again in a bit.",
            )

        _apply_plan_revision(db, row, plan, ctx.household_id, new_goal=new_goal)
        logger.info("Errand %s re-planned (goal=%r rev=%d)", row.id, new_goal, row.revision)
        return ServerCallbackResult(success=True)
    finally:
        db.close()


def _apply_plan_revision(
    db: Session, row: ErrandPlan, plan: Any, household_id: str, new_goal: str | None = None
) -> None:
    """Update the draft's steps IN PLACE from a revised plan, bump the revision
    (old card's Run → stale), and re-post the plan card. Shared by re-plan (edits
    the goal) and refine (keeps it). The card re-post is best-effort so a committed
    revision never flips the callback to FAILED."""
    node_steps = plan.routine_steps()
    if new_goal is not None:
        row.goal = new_goal
    row.summary = plan.summary
    row.steps = json.dumps(node_steps)
    row.revision = row.revision + 1
    db.commit()

    try:
        card_id = _post_errand_plan_card(row, node_steps, household_id, row.user_id)
        if card_id and card_id != row.inbox_item_id:
            row.inbox_item_id = card_id
            db.commit()
    except Exception:  # noqa: BLE001 — committed revision must survive a dead card
        logger.exception("Errand %s card re-post failed", row.id)


async def _handle_refine_errand_plan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Revise tap → refine the plan from a natural-language instruction ("call,
    don't email"). Feeds the planner the CURRENT plan + the instruction (not a
    fresh goal) and re-posts the refreshed card. Same guards as re-plan."""
    from app.core.llm_proxy_client import LLMProxyClient
    from app.db import get_session_local

    db = get_session_local()()
    try:
        plan_id = ctx.data.get("plan_id")
        instruction = (ctx.data.get("instruction") or "").strip()
        if not plan_id:
            return ServerCallbackResult(success=False, error="Missing plan_id")
        if not instruction:
            return ServerCallbackResult(success=False, error="Tell me what to change first.")
        row = (
            db.query(ErrandPlan)
            .filter(ErrandPlan.id == plan_id, ErrandPlan.household_id == ctx.household_id)
            .first()
        )
        if row is None:
            return ServerCallbackResult(success=False, error="Errand plan not found")
        if row.state != "draft":
            return ServerCallbackResult(
                success=True,
                context_data={"inbox": {
                    "title": "Errand already handled",
                    "summary": f"This errand is already {row.state} — it can't be changed.",
                    "metadata": {"household_id": ctx.household_id},
                }},
            )
        if not _revisions_match(ctx.data.get("revision"), row.revision):
            return ServerCallbackResult(
                success=False,
                error="This plan was updated — open the latest card to change it.",
            )

        current_steps = json.loads(row.steps or "[]")
        menu = await _resolve_node_menu(row.node_id, ctx.household_id, row.user_id)
        try:
            plan = await refine_errand_plan(
                row.summary or row.goal, current_steps, instruction, LLMProxyClient(), menu=menu
            )
        except ValueError as e:
            logger.info("Refine produced no usable plan (%r): %s", instruction, e)
            return ServerCallbackResult(
                success=False, error="I couldn't apply that change — try saying it differently.",
            )
        except Exception:  # noqa: BLE001 — infra; row untouched, old card stays valid
            logger.exception("Refine planner call failed (%r)", instruction)
            return ServerCallbackResult(
                success=False, error="I hit a snag revising that — try again in a bit.",
            )

        _apply_plan_revision(db, row, plan, ctx.household_id)  # goal unchanged by a refine
        logger.info("Errand %s refined (%r rev=%d)", row.id, instruction, row.revision)
        return ServerCallbackResult(success=True)
    finally:
        db.close()


# ── The errand as a workflow-engine CONSUMER (SEAM 1 + SEAM 3 wiring) ─────────
#
# Errands are the engine's first consumer. After the run/instance split the durable
# RUN is a ``Workflow`` row (NOT the ``ErrandPlan`` draft): the store maps a Workflow
# onto the engine's ``WorkflowRun`` and persists resume state in place; the progress
# emitter renders the completion inbox card; ``run_fn``/``finalize_fn`` late-bind the
# executor so tests can patch them. The store is kind-AGNOSTIC — it serves every
# workflow kind (the Workflow row carries ``kind``); only the consumer is per-kind.


class _WorkflowRunStore(WorkflowStore):
    """A ``WorkflowStore`` unit of work over one ``Workflow`` (RUN) row. Loads the
    row once and mutates it in place on save (rather than a bulk UPDATE) so the
    run's ORM object stays authoritative for the rest of the call — the atomic claim
    is still a bulk ``waiting→running`` UPDATE (the real serialization point)."""

    def __init__(self, db: Session):
        self._db = db
        self._row: Workflow | None = None

    def load(self, workflow_id: str) -> WorkflowRun | None:
        row = self._db.query(Workflow).filter(Workflow.id == workflow_id).first()
        self._row = row
        if row is None:
            return None
        return WorkflowRun(
            id=row.id,
            kind=row.kind,
            household_id=row.household_id,
            node_id=row.node_id,
            user_id=row.user_id,
            goal=row.goal or "",
            title=row.title or row.goal or "your errand",
            steps=json.loads(row.steps or "[]"),
            cursor=row.cursor,
            results=json.loads(row.results_json or "[]"),
            state=row.state,
        )

    def claim_running(self, workflow_id: str) -> bool:
        claimed = (
            self._db.query(Workflow)
            .filter(Workflow.id == workflow_id, Workflow.state == "waiting")
            .update({"state": "running"})
        )
        self._db.commit()
        if claimed and self._row is not None:
            self._row.state = "running"
        return bool(claimed)

    def save_waiting(
        self, run: WorkflowRun, cursor: int, results: list[dict[str, Any]],
        *, signal_kind: str = "phone_call", wake_at: Any = None,
    ) -> None:
        try:
            if self._row is not None:
                self._row.state = "waiting"
                self._row.cursor = cursor
                self._row.results_json = json.dumps(results)
                self._row.waiting_on = signal_kind  # phone_call | timer
                self._row.wake_at = wake_at          # set only for a timer wait
            self._db.commit()
        except Exception:  # noqa: BLE001 — the phone cards are the live feedback
            logger.exception("Workflow %s: couldn't persist waiting state", run.id)
            self._db.rollback()

    def save_terminal(self, run: WorkflowRun, state: str, error: str | None) -> None:
        # The completion card is posted independently (SEAM 3), so a commit blip here
        # must not crash the callback — anti-vanishing over bookkeeping.
        try:
            if self._row is not None:
                self._row.state = state
                self._row.error = error
            self._db.commit()
        except Exception:  # noqa: BLE001 — the card already went out; don't fail the callback
            logger.exception("Workflow %s: couldn't persist terminal state %s", run.id, state)
            self._db.rollback()

    def close(self) -> None:
        self._db.close()


def _workflow_store_factory() -> _WorkflowRunStore:
    """Open a fresh Workflow-backed store (its own DB session). ``get_session_local``
    is imported lazily so tests can patch it."""
    from app.db import get_session_local

    return _WorkflowRunStore(get_session_local()())


class _ErrandProgressEmitter(ProgressEmitter):
    """SEAM 3 for errands: render engine progress as inbox cards. Today only the
    terminal ``complete`` event surfaces (the headless completion card); the phone
    confirm/outcome cards are the live feedback while a call is in flight."""

    def emit(self, run: WorkflowRun, event: ProgressEvent) -> None:
        if event.type == "complete" and event.result is not None:
            _post_errand_completion_card(run.household_id, run.title, event.result, run.user_id)
        elif event.type == "needs_a_tap" and event.result is not None:
            _handle_replan_needs_a_tap(run, event.result)


async def _errand_run_fn(*args: Any, **kwargs: Any) -> Any:
    """Late-bind the executor so ``deliver_signal`` picks up a patched
    ``errand_executor.execute_errand`` in tests."""
    from app.services.errand_executor import execute_errand

    return await execute_errand(*args, **kwargs)


async def _errand_finalize_fn(goal: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Late-bind the aggregator/composer (same reason as ``_errand_run_fn``)."""
    from app.services.errand_executor import aggregate_and_compose

    return await aggregate_and_compose(goal, results)


def register_errand_workflow() -> None:
    """Register the errand as a workflow-engine consumer (store + run/finalize +
    progress). Idempotent; called at import so the durable resume path works even
    when the app lifespan hasn't run (tests import this module directly). The store
    is kind-agnostic (installed once); the consumer is keyed by the ``errand`` kind."""
    register_workflow_store(_workflow_store_factory)
    register_consumer(
        "errand",
        WorkflowConsumer(
            run_fn=_errand_run_fn,
            finalize_fn=_errand_finalize_fn,
            progress=_ErrandProgressEmitter(),
        ),
    )


def register_errand_callbacks() -> None:
    """Wire the plan card's Run/Revise/Cancel taps to their handlers. Call at
    startup (main.py lifespan) — an unregistered pair 400s at the tap."""
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_APPROVE_CALLBACK, _handle_approve_errand_plan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_REFINE_CALLBACK, _handle_refine_errand_plan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_REPLAN_CALLBACK, _handle_replan_errand_plan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_DISCARD_CALLBACK, _handle_discard_errand_plan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_REPLAN_APPROVE_CALLBACK, _handle_approve_replan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_STOP_CALLBACK, _handle_stop_errand)
    logger.info("🗒️ Errand server callbacks registered")


# Register the errand consumer at import — the durable resume primitive
# (deliver_signal) needs the store + consumer available even without the lifespan.
register_errand_workflow()
