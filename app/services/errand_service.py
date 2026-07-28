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
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.models import ErrandPlan, Routine
from app.services.errand_planner import plan_errand
from app.services.inbox_notification_service import post_inbox_item_sync

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
