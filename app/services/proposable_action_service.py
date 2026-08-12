"""Proposable-action dispatcher — the single generic server-plane chokepoint
that turns "an agent proposed running a command" into an actual, gated write.

This is the trust boundary for the whole "any agent → any command" mechanism.
A proposable-action card's Confirm button routes here (command
``jarvis.proposable_action``, callback ``execute``, ``target: "server"``). One
handler, registered once, serves every command+action — no per-command CC code,
so third-party Pantry packages need zero command-center changes.

On Confirm the handler:
  A. fail-closed household gate  (proposals.enabled — off ⇒ refuse)
  B. resolve the node that will run the write
  C. OPT-IN check — the target command must have *advertised* the callback as
     proposable (capability_registry). Not advertised ⇒ refuse. This is what
     stops an agent (or a crafted tap) from firing an un-opted-in callback.
  D. validate the proposed params against the declared schema (drop unknowns)
  E. idempotency — a completed job with the same key short-circuits (no re-run)
  F. execute the command's @callback on the node via the AUTHENTICATED node-plane
     callback path (create a node-plane CallbackJob, poll it), then post a
     success/failure card so a write never fails silently.

Everything after (F) reuses the shipping node-plane callback transport verbatim;
(A)-(E) are the new value and the new safety.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)

logger = logging.getLogger("uvicorn")

COMMAND_NAME = "jarvis.proposable_action"
_NODE_JOB_POLL_SECONDS = 0.5
_NODE_JOB_TIMEOUT_SECONDS = 25.0


# ── gate ─────────────────────────────────────────────────────────────────────


def _proposals_enabled(household_id: str) -> bool:
    """Fail-closed per-household read of ``proposals.enabled`` (default False).

    Any settings error disables proposals — deliberately the opposite of the
    memory gate's fail-open, because this lets background agents originate writes.
    """
    try:
        from app.services.settings_service import get_settings_service

        value = get_settings_service().get("proposals.enabled", household_id=str(household_id))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return False
    except Exception:  # noqa: BLE001 — fail closed
        return False


# ── node resolution ──────────────────────────────────────────────────────────


def _household_node_ids_by_recency(household_id: str) -> list[str]:
    """Household node ids in preference order: active nodes we've heard from,
    most-recent first, then any remaining nodes (inactive / never-seen) in last_seen
    order. Empty on lookup failure. This is the order the directed resolvers below
    walk when deciding which node should run a command."""
    try:
        from app.db import get_session_local
        from app.models import Node

        db = get_session_local()()
        try:
            rows = (
                db.query(Node)
                .filter(Node.household_id == household_id)
                .order_by(Node.last_seen.desc())
                .all()
            )
            # Active + recently-seen first (preserving last_seen order), then the rest —
            # a card with no explicit node must NOT resolve to a long-dead node.
            preferred = [n.node_id for n in rows if n.is_active and n.last_seen is not None]
            remaining = [n.node_id for n in rows if n.node_id not in preferred]
            return preferred + remaining
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        logger.warning("proposable_action: could not list household nodes")
        return []


def _first_household_node_id(household_id: str) -> str | None:
    """The single best household node when the caller has no specific target AND no
    command to match against (v0 fallback). Command-aware callers should use
    ``resolve_household_node_for_command`` so a multi-node household routes to a node
    that can actually run the command."""
    ids = _household_node_ids_by_recency(household_id)
    return ids[0] if ids else None


# Node-resolution probes must fail FAST. In the directed-ingest case the resolver
# runs on the SYNC /signals threadpool, so a slow/offline node must not pin a worker.
# We probe candidates CONCURRENTLY with a short per-node timeout AND an overall cap —
# otherwise K offline nodes would serialize into K×(interactive 10s) of blocking, a
# threadpool-exhaustion risk (a stream of directed signals naming an unadvertised
# command would walk every node). Concurrency also avoids false refusals: the overall
# cap can't cut the walk off before the advertiser is reached.
_RESOLVE_PROBE_TIMEOUT_S = 4.0
_RESOLVE_TOTAL_TIMEOUT_S = 6.0


async def _probe_node_tools(node_id: str) -> dict[str, Any] | None:
    """A capability fetch with a short timeout — an offline node should fail fast
    during resolution, not block for the interactive default."""
    from app.api.node_tools import _request_tools_from_node

    return await _request_tools_from_node(node_id, timeout=_RESOLVE_PROBE_TIMEOUT_S)


async def resolve_household_node_for_command(
    household_id: str,
    command_name: str,
    callback_name: str | None = None,
    *,
    fetch: Any = None,
) -> str | None:
    """A household node that currently ADVERTISES ``command_name`` (and
    ``callback_name`` when given), preferring the most-recently-seen active node.

    A household can have several nodes with different installed commands, so 'the
    most recent node' (``_first_household_node_id``) may not be the one that can run
    the target. Nodes are probed CONCURRENTLY (see the timeout constants above) and
    the first in preference order that advertises the command wins. Returns ``None``
    if NO reachable node advertises it — so the caller refuses visibly rather than
    dispatching to a node that can't run it (or silently dropping a directed proposal).
    """
    from app.services.capability_registry import (
        list_proposable_actions,
        resolve_proposable_action,
    )

    node_ids = _household_node_ids_by_recency(household_id)
    if not node_ids:
        return None
    probe = fetch or _probe_node_tools

    async def _advertises(node_id: str) -> bool:
        try:
            if callback_name:
                return bool(
                    await resolve_proposable_action(
                        node_id, command_name, callback_name, fetch=probe
                    )
                )
            actions = await list_proposable_actions(node_id, fetch=probe)
            return any(a.get("command") == command_name for a in actions)
        except Exception:  # noqa: BLE001 — an unreachable node is simply not a match
            return False

    try:
        flags = await asyncio.wait_for(
            asyncio.gather(*(_advertises(nid) for nid in node_ids)),
            timeout=_RESOLVE_TOTAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "proposable_action: node resolution timed out for command %s", command_name
        )
        return None
    # First node in preference order that advertises the command.
    for node_id, advertises in zip(node_ids, flags):
        if advertises:
            return node_id
    return None


# ── param validation ─────────────────────────────────────────────────────────


def validate_against_params(
    raw: dict[str, Any], param_specs: list[dict[str, Any]]
) -> tuple[bool, dict[str, Any], str | None]:
    """Coerce/strip ``raw`` to exactly the declared params.

    Drops keys not in the schema (bounds the "attacker-chosen data dict" to the
    declared shape), enforces required-present and enum membership. Returns
    (ok, validated_args, error). Type coercion is intentionally light — values
    arrive as strings and the target command parses them (e.g. ISO datetimes).
    """
    declared = {p["name"]: p for p in param_specs}
    args: dict[str, Any] = {}
    for name, spec in declared.items():
        present = name in raw and raw[name] not in (None, "")
        if not present:
            if spec.get("required"):
                return False, {}, f"missing required field '{name}'"
            continue
        value = raw[name]
        enum_values = spec.get("enum_values")
        if enum_values and str(value) not in enum_values:
            return False, {}, f"'{name}' must be one of {enum_values}"
        args[name] = value
    return True, args, None


# ── idempotency ──────────────────────────────────────────────────────────────


def _already_completed(household_id: str, idempotency_key: str) -> bool:
    """True if a callback job with this (household, key) already completed —
    so a double-tap / re-post never runs the target callback twice."""
    if not idempotency_key:
        return False
    try:
        from app.db import get_session_local
        from app.models import CallbackJob

        db = get_session_local()()
        try:
            existing = (
                db.query(CallbackJob)
                .filter(
                    CallbackJob.household_id == household_id,
                    CallbackJob.idempotency_key == idempotency_key,
                    CallbackJob.status == "completed",
                )
                .first()
            )
            return existing is not None
        finally:
            db.close()
    except Exception:  # noqa: BLE001
        return False


# ── node-plane execution (authenticated @callback path) ──────────────────────


async def _execute_target_callback_on_node(
    *,
    node_id: str,
    household_id: str,
    user_id: int | None,
    command_name: str,
    callback_name: str,
    args: dict[str, Any],
    idempotency_key: str,
    timeout: float = _NODE_JOB_TIMEOUT_SECONDS,
) -> tuple[bool, dict[str, Any] | None, str | None]:
    """Run the target command's @callback on ``node_id`` and await the result.

    Creates a node-plane CallbackJob (navigation_type="stack" so its result does
    NOT auto-fan-out an inbox card — this dispatcher owns the user-facing card),
    publishes the MQTT job, and polls the job row until it leaves ``pending``.
    Returns (success, result_context_data, error).
    """
    from app.db import get_session_local
    from app.models import CallbackJob
    from app.services.node_command_service import get_node_command_service

    job_id: str | None = None
    db = get_session_local()()
    try:
        job = CallbackJob(
            node_id=node_id,
            household_id=household_id,
            user_id=user_id,
            command_name=command_name,
            callback_name=callback_name,
            data_json=json.dumps(args),
            navigation_type="stack",
            status="pending",
            idempotency_key=idempotency_key or None,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(minutes=5),
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    get_node_command_service().publish_command_with_id(
        node_id=node_id, command="callback", details=None, request_id=job_id
    )

    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(_NODE_JOB_POLL_SECONDS)
        db = get_session_local()()
        try:
            row = db.query(CallbackJob).filter(CallbackJob.id == job_id).first()
            if row is None:
                return False, None, "job vanished"
            if row.status == "completed":
                data = json.loads(row.result_context_data_json) if row.result_context_data_json else {}
                return True, data, None
            if row.status == "failed":
                return False, None, row.error_message or "callback failed on the node"
        finally:
            db.close()
    return False, None, "the device didn't respond in time"


# ── the handlers ─────────────────────────────────────────────────────────────


def _inbox(title: str, summary: str = "", *, household_id: str | None = None) -> dict[str, Any]:
    meta = {"household_id": household_id} if household_id else {}
    return {"inbox": {"title": title, "summary": summary, "metadata": meta}}


async def _handle_execute(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Confirm tap on a proposable-action card. See module docstring for the
    A-F pipeline. Every failure returns a visible card (never a silent no-op)."""
    from app.services.capability_registry import resolve_proposable_action

    action_meta = (ctx.data or {}).get("_action", {}) or {}
    target_command = action_meta.get("target_command")
    target_callback = action_meta.get("target_callback")
    idempotency_key = action_meta.get("idempotency_key") or ""
    raw_params = {k: v for k, v in (ctx.data or {}).items() if k != "_action"}

    if not target_command or not target_callback:
        return ServerCallbackResult(success=False, error="malformed proposal")

    # (A) fail-closed household gate
    if not _proposals_enabled(ctx.household_id):
        return ServerCallbackResult(
            success=False,
            error="proposals disabled",
            context_data=_inbox(
                "Action not run",
                "Agent action cards are turned off for this household.",
                household_id=ctx.household_id,
            ),
        )

    # (B) resolve the node — prefer the card's explicit node, else a household node
    # that actually ADVERTISES this command (not merely the most-recent one).
    node_id = action_meta.get("node_id")
    if not node_id:
        node_id = await resolve_household_node_for_command(
            ctx.household_id, target_command, target_callback
        )
    if not node_id:
        return ServerCallbackResult(
            success=False,
            error="no node",
            context_data=_inbox("Couldn't run that", "No device is available to run it.",
                                household_id=ctx.household_id),
        )

    # (C) OPT-IN: the command must have advertised this callback as proposable
    spec = await resolve_proposable_action(node_id, target_command, target_callback)
    if spec is None:
        return ServerCallbackResult(
            success=False,
            error="not proposable",
            context_data=_inbox(
                "Couldn't run that",
                f"No device here can run '{target_command}'.",
                household_id=ctx.household_id,
            ),
        )

    # (D) validate params against the DECLARED schema
    ok, args, verr = validate_against_params(raw_params, spec.get("params", []))
    if not ok:
        return ServerCallbackResult(
            success=False,
            error=f"invalid params: {verr}",
            context_data=_inbox("Couldn't run that", f"Something was missing: {verr}.",
                                household_id=ctx.household_id),
        )

    # (E) idempotency short-circuit
    if _already_completed(ctx.household_id, idempotency_key):
        return ServerCallbackResult(
            success=True,
            context_data=_inbox("Already done", "This was already handled.",
                                household_id=ctx.household_id),
        )

    # (F) run the @callback on the node, authenticated path
    success, result, err = await _execute_target_callback_on_node(
        node_id=node_id,
        household_id=ctx.household_id,
        user_id=ctx.user_id,
        command_name=target_command,
        callback_name=target_callback,
        args=args,
        idempotency_key=idempotency_key,
    )
    if not success:
        return ServerCallbackResult(
            success=False,
            error=err or "execution failed",
            context_data=_inbox("Couldn't finish that", err or "The device couldn't complete it.",
                                household_id=ctx.household_id),
        )

    spoken = ""
    if isinstance(result, dict):
        spoken = result.get("message") or ""
    return ServerCallbackResult(
        success=True,
        context_data=_inbox(spoken or "Done", "", household_id=ctx.household_id),
    )


async def _handle_dismiss(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """Dismiss tap — nothing runs; no follow-up card."""
    return ServerCallbackResult(success=True)


async def _handle_suppress(ctx: ServerCallbackContext) -> ServerCallbackResult:
    """'Never suggest this' tap — record a blocklist entry for this user.

    Captures the deterministic ``source`` (a hard key, e.g. the sender) and the
    semantic ``descriptor`` (a negative example for the detector's prompt),
    scoped to the ``target_command``. The detector agent consults these each run
    so it stops proposing this — and look-alikes. Nothing is executed.
    """
    from app.services.proposal_suppressions import record_suppression

    action_meta = (ctx.data or {}).get("_action", {}) or {}
    command = action_meta.get("target_command")
    source = (ctx.data or {}).get("source") or None
    descriptor = (ctx.data or {}).get("descriptor") or None
    if not command or (not source and not descriptor):
        return ServerCallbackResult(success=False, error="nothing to suppress")

    record_suppression(
        household_id=ctx.household_id,
        user_id=ctx.user_id,
        command=command,
        source_key=source,
        descriptor=descriptor,
    )
    return ServerCallbackResult(
        success=True,
        context_data=_inbox(
            "Won't suggest that again",
            "You won't get more cards like this one. Manage these in the app.",
            household_id=ctx.household_id,
        ),
    )


def register_proposable_action_callbacks() -> None:
    """Register the single generic dispatcher's callbacks. Called once at startup."""
    register_server_callback(COMMAND_NAME, "execute", _handle_execute)
    register_server_callback(COMMAND_NAME, "dismiss", _handle_dismiss)
    register_server_callback(COMMAND_NAME, "suppress", _handle_suppress)
    logger.info(
        "Registered proposable-action server callbacks (%s.execute/dismiss/suppress)",
        COMMAND_NAME,
    )
