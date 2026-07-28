"""Errand service — turn a goal into a reviewable plan card (Errand Runner POC).

Chunk 3 of the planner + plan-card POC (prds/errand-runner.md §2-§3). The flow:

    plan_errand(goal)  →  transient Routine (so the node pre-pulls the steps by
    slug)  →  persist a DRAFT ErrandPlan  →  push a plan card with Run/Cancel.

"LLM proposes, card disposes": nothing runs until the user taps **Run** on the
card. Chunk 4 wires that tap to a server-plane callback that executes the
transient routine via ``execute_routine_on_node`` (the fc936a5 detached-run
path) and pushes the completion card.

The card is a first-party interactive inbox item built exactly like the phone
call-plan card (``phone_call_service.create_call_plan``): ``metadata`` carries
``interactive_elements`` with ``target: "server"`` so the taps route to the
``server_callback_registry`` (CC plane), NOT the node plane. Chunk 4 registers
``("errand", "approve_errand_plan")`` / ``("errand", "discard_errand_plan")``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import ErrandPlan, Routine
from app.services.errand_planner import plan_errand
from app.services.inbox_notification_service import post_inbox_item_sync
from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)

logger = logging.getLogger("uvicorn")

# The command + callback names the plan card's Run/Cancel buttons target. Chunk 4
# registers handlers for these pairs in the server_callback_registry; the mobile
# tap routes here because interactive_elements carry target:"server".
ERRAND_CALLBACK_COMMAND = "errand"
ERRAND_APPROVE_CALLBACK = "approve_errand_plan"
ERRAND_DISCARD_CALLBACK = "discard_errand_plan"

# Own inbox category so the plan card is filterable and distinct from the
# routine completion card (category="routine").
ERRAND_PLAN_CATEGORY = "errand_plan"


def _node_args_to_mobile(args: dict[str, Any]) -> list[dict[str, str]]:
    """Invert ``routines._flatten_args``: node-native ``{k: v}`` → mobile-native
    ``[{key, value}]`` pairs, the shape ``Routine.steps`` stores.

    Every value becomes a string; a list/dict is JSON-encoded so that
    ``_flatten_args`` (which ``json.loads`` any value beginning with ``[`` or
    ``{``) round-trips it back to the real type at the node-pull boundary.
    Storing the flat dict directly would make ``_flatten_args`` iterate the dict
    KEYS and silently drop every arg — the node would then run each step with
    empty args, with no error. See ``app/api/routines.py:124``.
    """
    pairs: list[dict[str, str]] = []
    for key, value in args.items():
        if isinstance(value, str):
            encoded = value
        elif isinstance(value, (list, dict)):
            encoded = json.dumps(value)
        else:
            encoded = str(value)
        pairs.append({"key": key, "value": encoded})
    return pairs


def _plan_card_body(summary: str, node_steps: list[dict[str, Any]]) -> str:
    """Human-readable, read-only plan for the card body (markdown)."""
    lines = [summary, "", "**Plan:**"]
    for i, step in enumerate(node_steps, start=1):
        lines.append(f"{i}. {step.get('label') or step.get('command')}")
    lines.append("")
    lines.append("Tap **Run** to start it, or **Cancel** to discard.")
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
    data = {"plan_id": row.id, "revision": row.revision}
    return {
        "household_id": household_id,
        "plan_id": row.id,
        "revision": row.revision,
        "routine_slug": row.routine_slug,
        "steps": node_steps,  # read-only, for a structured render (card v1 = read-only)
        "interactive_elements": [
            {
                "id": "run-errand",
                "label": "Run",
                "command": ERRAND_CALLBACK_COMMAND,
                "callback": ERRAND_APPROVE_CALLBACK,
                "target": "server",
                "data": data,
            },
            {
                "id": "cancel-errand",
                "label": "Cancel",
                "command": ERRAND_CALLBACK_COMMAND,
                "callback": ERRAND_DISCARD_CALLBACK,
                "target": "server",
                "data": data,
            },
        ],
    }


def _post_errand_plan_card(
    row: ErrandPlan,
    node_steps: list[dict[str, Any]],
    household_id: str,
    user_id: int | None,
) -> str | None:
    """Push the plan card (Run/Cancel) to the initiating user's inbox.

    Best-effort: ``post_inbox_item_sync`` swallows its own failures and returns
    None, and the draft is already committed — a lost card never fails the errand
    (mirrors the phone card's fire-and-forget discipline).
    """
    summary = row.summary or row.goal
    return post_inbox_item_sync(
        household_id=household_id,
        title=f"🗒️ Errand plan: {summary}",
        summary=summary,
        body=_plan_card_body(summary, node_steps),
        category=ERRAND_PLAN_CATEGORY,
        metadata=build_plan_card_metadata(row, node_steps, household_id),
        user_id=user_id,
        push=True,
        target_type="user" if user_id is not None else "household",
    )


async def create_errand_plan(
    db: Session,
    household_id: str,
    node_id: str,
    goal: str,
    user_id: int | None = None,
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

    # Routine helpers live in the routines API module — lazy-import them to match
    # routine_scheduler's convention and avoid an import cycle (routines.py:101).
    from app.api.routines import _unique_slug, publish_routines_sync

    plan = await plan_errand(goal, llm_client)  # raises ValueError on no usable plan
    node_steps = plan.routine_steps()  # [{command, args:{k:v}, label}] — node-native

    # 1. Transient routine. The node runs errands BY SLUG from its local store, so
    #    the steps must live in a Routine it can pull. Routine.steps is stored
    #    MOBILE-native (args as [{key,value}] pairs) — the node-pull endpoint
    #    flattens it back; the ErrandPlan row below stores the opposite shape.
    # Bound the slug source — Routine.slug is String(255) and plan.summary is
    # free-form; a long summary would otherwise overflow the column.
    slug = _unique_slug(db, household_id, f"errand {plan.summary[:100]}")
    mobile_steps = [
        {
            "command": s["command"],
            "args": _node_args_to_mobile(s.get("args", {})),
            "label": s.get("label", ""),
        }
        for s in node_steps
    ]
    routine = Routine(
        id=str(uuid4()),
        household_id=household_id,
        slug=slug,
        name=f"Errand: {plan.summary}"[:255],
        trigger_phrases=json.dumps([]),  # no voice trigger
        steps=json.dumps(mobile_steps),
        response_instruction="",
        response_length="short",
        schedule=None,
        enabled=True,  # must be True or the node-pull endpoint skips it
    )
    db.add(routine)

    # 2. Persist the DRAFT plan. steps here are NODE-native (args-as-object) per
    #    the ErrandPlan model contract — the opposite shape from Routine.steps.
    row = ErrandPlan(
        household_id=household_id,
        user_id=user_id,
        node_id=node_id,
        goal=goal,
        summary=plan.summary,
        steps=json.dumps(node_steps),
        routine_slug=slug,
        state="draft",
        revision=1,
    )
    db.add(row)
    db.commit()
    db.refresh(row)

    # 3. Nudge the household's active nodes to pre-pull the transient routine now,
    #    so the target node has it in its local store before the user taps Run.
    #    Best-effort — the draft is already committed, so a failed nudge (like a
    #    failed card below) must not turn a successful errand into a 500.
    try:
        publish_routines_sync(household_id, db)
    except Exception:  # noqa: BLE001 — a failed nudge must not fail the errand
        logger.exception("Errand routine-sync nudge failed for %s", row.id)

    # 4. Plan card — best-effort; the draft is already committed.
    try:
        _post_errand_plan_card(row, node_steps, household_id, user_id)
    except Exception:  # noqa: BLE001 — a failed card must not fail the errand
        logger.exception("Errand plan card push failed for %s", row.id)

    logger.info(
        "Errand plan %s drafted (household=%s node=%s slug=%s steps=%d)",
        row.id, household_id, node_id, slug, len(node_steps),
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
    household_id: str, node_id: str, goal: str, user_id: int | None = None
) -> None:
    """Draft an errand plan on a FRESH session — for fire-and-forget callers.

    The ``run_errand`` voice tool fires this detached (it has no request-scoped DB
    session, and one would be closed by the time the task runs — cf. phone's
    ``create_call_plan`` opening its own session). Because the tool already spoke
    "I'll send it to your phone," ANY failure posts a card so the user is never
    left hanging — the phone stack's never-vanish rule (create_call_plan:543).
    """
    from app.db import get_session_local

    db = get_session_local()()
    try:
        await create_errand_plan(db, household_id, node_id, goal, user_id=user_id)
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


def _delete_transient_routine(db: Session, household_id: str, slug: str | None) -> None:
    """Remove the single-use transient routine and nudge nodes to drop it.

    Best-effort: cleanup failure must never fail the callback. Deleting the row
    + re-nudging keeps errand execution-vehicle routines from accumulating in the
    household routine set (the deferred-cleanup finding from chunk 3's review).
    """
    if not slug:
        return
    try:
        deleted = (
            db.query(Routine)
            .filter(Routine.slug == slug, Routine.household_id == household_id)
            .delete()
        )
        db.commit()
        if deleted:
            from app.api.routines import publish_routines_sync

            publish_routines_sync(household_id, db)
    except Exception:  # noqa: BLE001 — cleanup must never fail the callback
        logger.exception("Failed to delete transient errand routine %s", slug)
        db.rollback()


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
    """Run tap → execute the drafted errand on its node.

    Reuses ``execute_routine_on_node(notify_on_complete=True)`` (the fc936a5
    detached-run path), which posts the completion card itself — so this handler
    returns success WITHOUT a second inbox card. Single-use + stale-card guarded.
    """
    from app.api.routines import execute_routine_on_node
    from app.db import get_session_local

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

        # Single-use: a second tap on an already-run/cancelled plan is a friendly
        # no-op card, not an error (mirrors the phone confirm handler).
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

        routine = (
            db.query(Routine)
            .filter(Routine.slug == row.routine_slug, Routine.household_id == ctx.household_id)
            .first()
        )
        if routine is None:
            row.state = "failed"
            row.error = "transient routine missing"
            db.commit()
            return ServerCallbackResult(success=False, error="Errand steps not found — re-plan it.")

        row.state = "running"
        row.confirmed_at = datetime.utcnow()
        db.commit()

        # The completion card is posted by execute_routine_on_node itself
        # (notify_on_complete), so we return no context_data['inbox'] on success
        # (avoids a double card). A RAISED failure must still leave a terminal
        # state — never a stuck 'running' (anti-vanishing, cf. phone enqueue_dial).
        try:
            result = await execute_routine_on_node(
                db, ctx.household_id, routine, row.node_id, "errand",
                notify_on_complete=True, notify_user_id=row.user_id,
            )
        except Exception as e:  # noqa: BLE001 — never leave the errand stuck running
            logger.exception("Errand %s failed to run", row.id)
            row.state = "failed"
            row.error = str(e)[:500]
            db.commit()
            _delete_transient_routine(db, ctx.household_id, row.routine_slug)  # cleanup on every terminal path
            return ServerCallbackResult(success=False, error="Couldn't run the errand.")

        status = (result or {}).get("status") or "failed"
        row.state = "done" if status == "success" else status  # partial|failed|timeout
        db.commit()

        _delete_transient_routine(db, ctx.household_id, row.routine_slug)
        logger.info("Errand %s ran on node %s → %s", row.id, row.node_id, row.state)
        return ServerCallbackResult(success=True)
    finally:
        db.close()


def _handle_discard_errand_plan(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Cancel tap → discard a draft errand and drop its transient routine."""
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
            _delete_transient_routine(db, ctx.household_id, row.routine_slug)
        return ServerCallbackResult(
            success=True,
            context_data={"inbox": {
                "title": "🗑️ Errand discarded",
                "summary": row.summary or row.goal,
                "metadata": {"household_id": ctx.household_id},
            }},
        )
    finally:
        db.close()


def register_errand_callbacks() -> None:
    """Wire the plan card's Run/Cancel taps to their handlers. Call at startup
    (main.py lifespan) — an unregistered pair 400s at the tap."""
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_APPROVE_CALLBACK, _handle_approve_errand_plan)
    register_server_callback(ERRAND_CALLBACK_COMMAND, ERRAND_DISCARD_CALLBACK, _handle_discard_errand_plan)
    logger.info("🗒️ Errand server callbacks registered")
