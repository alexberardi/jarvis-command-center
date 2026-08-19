"""Signal-automation execution — interpret a household's free-text instruction
when a signal fires, then act.

When a catalog signal (presence.left/seen, appt.upcoming) is ingested and the
household has an ENABLED free-text instruction for that kind (authored on the
signal-automations screen), this reaction:

  1. resolves a reachable household node + its tool menu (``client_tools``),
  2. runs ONE background-model inference with those tools: "when <label> happens
     the user wants <instruction>; here's what happened <facts> — call one tool
     with the right arguments, or none", and
  3. acts per the rule's DELIVERY choice:
     ``automatic``     → dispatches it to the node immediately (``dispatch_node_command``);
     ``notification``  → posts a tap-to-confirm inbox card whose Confirm routes to
                         our OWN server-plane callback, which then dispatches it.

Delivery is the USER'S per-rule choice, not our guess — they decide whether each
automation just happens or asks first (default ``notification``, the safe one).
This replaces the old reversible-vs-sensitive classifier: we no longer infer intent.

Safety posture: every automation is per-rule opt-in (an admin enabled it), the LLM
runs on the BACKGROUND model slot (never contends with live voice), a home↔away
heartbeat re-assert / calendar re-emit is de-duped so we don't re-run, and the
whole reaction never raises into signal ingest.

Registered on the generic :mod:`signal_reaction_registry` for each catalog kind;
the confirm path registers a ``jarvis.signal_automation`` server callback.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from app.services.server_callback_registry import (
    ServerCallbackContext,
    ServerCallbackResult,
    register_server_callback,
)
from app.services.signal_reaction_registry import ReactionContext, register_signal_reaction

logger = logging.getLogger("uvicorn")

# Our own server-plane confirm target (NOT jarvis.proposable_action, which refuses
# commands that haven't advertised a proposable_action).
_SERVER_COMMAND = "jarvis.signal_automation"
_EXECUTE_CALLBACK = "execute"
_DISMISS_CALLBACK = "dismiss"

_LLM_MAX_TOKENS = 512
_DISPATCH_TIMEOUT_S = 20.0
_TOOLS_TIMEOUT_S = 4.0

# In-process per-(household, user, kind) signature of the last signal we acted on,
# so a heartbeat re-assert / calendar re-emit doesn't re-run the LLM + re-actuate.
# Resets on restart (a first post-restart signal acts once). See presence_automation
# for the same rationale on the deterministic path.
_last_signature: dict[tuple[str, str, str], str] = {}


def clear_state() -> None:
    """Test hook — drop the per-(household, user, kind) dedup memory."""
    _last_signature.clear()


def _dedup_identity(ctx: ReactionContext) -> tuple[tuple[str, str, str], str]:
    """The dedup (key, value) for a signal occurrence — we act once per distinct
    value under a key.

    presence.left / presence.seen SHARE one key (…, "presence") whose value is the
    state, so an arrival ("home") resets the departure dedup: a leave→arrive→leave
    cycle acts all three times, while a same-state heartbeat re-assert is skipped.
    (Keying per-kind instead would latch "away" forever after the first leave —
    every later departure would be wrongly skipped.) Event-scoped kinds dedup
    per-kind on the event identity (a calendar re-emit of the same event is one
    occurrence)."""
    facts = ctx.facts or {}
    hh, uid = str(ctx.household_id), str(ctx.user_id)
    if ctx.kind in ("presence.left", "presence.seen"):
        return (hh, uid, "presence"), str(facts.get("state") or ctx.kind)
    return (hh, uid, ctx.kind), str(
        facts.get("event_id") or facts.get("id") or sorted(facts.items())
    )


def _label_for(kind: str) -> str:
    from app.services import signal_catalog

    entry = signal_catalog.get_kind(kind)
    return entry.label if entry else kind


# ── node + tools ────────────────────────────────────────────────────────────
async def _resolve_node_and_tools(
    household_id: str, preferred_node_id: str | None
) -> tuple[str | None, list[dict[str, Any]]]:
    """Find a reachable household node and its tool menu. Presence signals carry no
    node, so walk the household's nodes (most-recent first); the first that returns
    tools wins. Returns (node_id, cleaned function-tools) or (None, [])."""
    from app.api.node_tools import _request_tools_from_node
    from app.core.tool_builder import ToolBuilder
    from app.services.proposable_action_service import _household_node_ids_by_recency

    candidates: list[str] = []
    if preferred_node_id:
        candidates.append(preferred_node_id)
    for nid in _household_node_ids_by_recency(household_id):
        if nid not in candidates:
            candidates.append(nid)

    for nid in candidates:
        report = await _request_tools_from_node(nid, timeout=_TOOLS_TIMEOUT_S)
        tools = (report or {}).get("client_tools") or []
        if tools:
            # client_tools are already OpenAI function-tools; strip our private
            # extensions before handing them to the model.
            return nid, ToolBuilder.strip_jarvis_extensions(tools)
    return None, []


# ── the single inference ────────────────────────────────────────────────────
async def _pick_action(
    *, label: str, instruction: str, facts: dict[str, Any], tools: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]] | None:
    """One background-model tool-call inference: choose the single tool + arguments
    that carry out the instruction, or None if the model calls no tool."""
    from app.core.llm_proxy_client import LLMProxyClient

    system = (
        "You carry out a household's standing automation rule. When a household "
        "event happens, do exactly what the user asked by calling ONE of the "
        "available tools with correct arguments. If no available tool fits the "
        "instruction, call no tool. Never invent tools or arguments."
    )
    user = (
        f"Event: {label}\n"
        f"Event details: {json.dumps(facts, default=str)}\n"
        f'The user\'s instruction for this event: "{instruction}"\n'
        "Call the single tool that carries out the instruction, or none if nothing fits."
    )
    resp = await LLMProxyClient().chat_completion(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model="background",
        temperature=0,
        tools=tools,
        tool_choice="auto",
        max_tokens=_LLM_MAX_TOKENS,
        include_date_context=False,
        extra_body={"reasoning_budget": 0},  # no thinking pass — just pick a tool
    )
    choice = (resp.get("choices") or [{}])[0]
    tool_calls = (choice.get("message") or {}).get("tool_calls") or []
    if not tool_calls:
        return None
    fn = (tool_calls[0] or {}).get("function") or {}
    name = fn.get("name")
    if not name:
        return None
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        args = {}
    return name, (args if isinstance(args, dict) else {})


# ── dispatch ────────────────────────────────────────────────────────────────
def _dispatch_ok(out: Any) -> bool:
    return bool(isinstance(out, dict) and out.get("success", True) and not out.get("timeout"))


async def _dispatch_now(
    node_id: str, command_name: str, arguments: dict[str, Any], user_id: int | None
) -> bool:
    """Run the chosen action on the node immediately (automatic delivery — no
    confirmation)."""
    from app.services.node_command_service import dispatch_node_command

    out = await dispatch_node_command(
        node_id, command_name, arguments, user_id=user_id, timeout=_DISPATCH_TIMEOUT_S
    )
    ok = _dispatch_ok(out)
    logger.info("signal automation: auto-ran %s -> success=%s", command_name, ok)
    return ok


def _action_idem(household_id: str, command_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(arguments, sort_keys=True, default=str)
    return f"sigauto:{household_id}:{command_name}:{abs(hash(payload))}"


def _emit_confirm_card(
    *,
    household_id: str,
    user_id: int | None,
    node_id: str,
    command_name: str,
    arguments: dict[str, Any],
    label: str,
    instruction: str,
) -> bool:
    """Post a tap-to-confirm inbox card for a SENSITIVE action. Confirm routes to
    our ``jarvis.signal_automation.execute`` server callback with the chosen action
    in ``_action`` (control metadata mobile's field-merge never touches)."""
    from app.services.inbox_notification_service import post_inbox_item_sync

    idem = _action_idem(household_id, command_name, arguments)
    action = {
        "node_id": node_id,
        "command_name": command_name,
        "arguments": arguments,
        "idempotency_key": idem,
    }
    elements = [
        {
            "id": f"confirm-{idem}",
            "label": "Do it",
            "kind": "confirm",
            "command": _SERVER_COMMAND,
            "callback": _EXECUTE_CALLBACK,
            "target": "server",
            "data": {"_action": action},
            "navigation_type": "new_notification",
        },
        {
            "id": f"dismiss-{idem}",
            "label": "Not now",
            "kind": "dismiss",
            "command": _SERVER_COMMAND,
            "callback": _DISMISS_CALLBACK,
            "target": "server",
            "data": {"_action": {"idempotency_key": idem}},
            "navigation_type": "new_notification",
        },
    ]
    item_id = post_inbox_item_sync(
        household_id=household_id,
        title="Confirm automation",
        summary=f"{label}: {instruction}",
        body=(
            f'Your automation for "{label}" wants to run a sensitive action '
            f"({command_name}). Tap to confirm."
        ),
        category="proposal",
        metadata={"interactive_elements": elements},
        user_id=user_id,
        target_type="user" if user_id else "household",
    )
    return item_id is not None


# ── the reaction ────────────────────────────────────────────────────────────
async def react_to_signal_automation(
    ctx: ReactionContext,
    *,
    get_rule: Callable[[str, str], dict[str, str] | None] | None = None,
    resolve: Callable[..., Awaitable[tuple[str | None, list[dict[str, Any]]]]] | None = None,
    pick: Callable[..., Awaitable[tuple[str, dict[str, Any]] | None]] | None = None,
    run_automatic: Callable[..., Awaitable[bool]] | None = None,
    emit_confirm: Callable[..., bool] | None = None,
) -> str:
    """Interpret the household's instruction for this signal and act, per the rule's
    DELIVERY choice (the user's, not our guess): ``automatic`` runs the chosen
    action immediately; ``notification`` proposes it as a tap-to-confirm card.

    Status strings (logging/tests): ``no_rule`` (no enabled rule), ``unchanged``
    (a re-assert we already handled), ``no_tools`` (no reachable node / tools),
    ``no_action`` (the model chose nothing), ``ran:<cmd>`` / ``failed:<cmd>``
    (automatic delivery), ``confirm:<cmd>`` / ``confirm_failed`` (notification
    delivery), ``error``. Never raises.
    """
    get_rule = get_rule or _get_enabled_rule
    rule = get_rule(ctx.household_id, ctx.kind)
    if not rule:
        return "no_rule"
    instruction = rule["instruction"]
    delivery = rule.get("delivery") or "notification"

    # Act once per distinct signal occurrence (skip heartbeat re-asserts / re-emits).
    dkey, sig = _dedup_identity(ctx)
    if _last_signature.get(dkey) == sig:
        return "unchanged"

    try:
        resolve = resolve or _resolve_node_and_tools
        node_id, tools = await resolve(ctx.household_id, ctx.node_id)
        if not node_id or not tools:
            return "no_tools"

        label = _label_for(ctx.kind)
        pick = pick or _pick_action
        chosen = await pick(
            label=label, instruction=instruction, facts=ctx.facts or {}, tools=tools
        )
        # Latch now (before dispatch): we've handled this occurrence, so a re-assert
        # won't re-run the LLM — even if the model chose nothing.
        _last_signature[dkey] = sig
        if not chosen:
            return "no_action"

        command_name, arguments = chosen
        if delivery == "automatic":
            run_automatic = run_automatic or _dispatch_now
            ok = await run_automatic(node_id, command_name, arguments, ctx.user_id)
            return f"ran:{command_name}" if ok else f"failed:{command_name}"

        emit_confirm = emit_confirm or _emit_confirm_card
        posted = emit_confirm(
            household_id=ctx.household_id,
            user_id=ctx.user_id,
            node_id=node_id,
            command_name=command_name,
            arguments=arguments,
            label=label,
            instruction=instruction,
        )
        return f"confirm:{command_name}" if posted else "confirm_failed"
    except Exception:  # noqa: BLE001 — a reaction must never break signal ingest
        logger.warning(
            "signal automation execution failed for %s", ctx.household_id, exc_info=True
        )
        return "error"


def _get_enabled_rule(household_id: str, kind: str) -> dict[str, str] | None:
    from app.services.signal_automation_store import get_enabled_rule

    return get_enabled_rule(household_id, kind)


# ── the confirm-card server callback ────────────────────────────────────────
async def _execute_confirmed_action(sctx: ServerCallbackContext) -> ServerCallbackResult:
    """Run the sensitive action the user just confirmed (server plane)."""
    action = (sctx.data or {}).get("_action") or {}
    node_id = action.get("node_id")
    command_name = action.get("command_name")
    arguments = action.get("arguments") or {}
    if not node_id or not command_name:
        return ServerCallbackResult(success=False, error="malformed automation action")

    from app.services.node_command_service import dispatch_node_command

    out = await dispatch_node_command(
        node_id, command_name, arguments, user_id=sctx.user_id, timeout=_DISPATCH_TIMEOUT_S
    )
    ok = _dispatch_ok(out)
    return ServerCallbackResult(
        success=ok,
        error=None if ok else (out.get("error") if isinstance(out, dict) else "dispatch failed"),
        context_data=(
            {"inbox": {"title": "Done", "summary": f"Ran {command_name}", "body": ""}}
            if ok
            else None
        ),
    )


async def _dismiss_confirmed_action(sctx: ServerCallbackContext) -> ServerCallbackResult:
    """The user tapped "Not now" — nothing to do, just close the card cleanly."""
    return ServerCallbackResult(success=True)


# ── registration ────────────────────────────────────────────────────────────
def register_signal_automation_executor() -> None:
    """Register the interpret-at-fire reaction for every authorable catalog kind,
    plus the confirm-card server callbacks (called once at startup)."""
    from app.services import signal_catalog

    for entry in signal_catalog.catalog():
        register_signal_reaction(entry.kind, "automation", react_to_signal_automation)
    register_server_callback(_SERVER_COMMAND, _EXECUTE_CALLBACK, _execute_confirmed_action)
    register_server_callback(_SERVER_COMMAND, _DISMISS_CALLBACK, _dismiss_confirmed_action)
