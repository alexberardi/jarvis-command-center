"""Capability matcher — the OPEN proposal mode ("here's some data; is there any
command that can help?").

Where the *directed* path has the agent name a target (``propose_action``), the
open path hands over detected data and lets an LLM pick the action from the
registry of *proposable* actions a node advertises, filling its params. A brand
new Pantry command that declares a proposable action becomes reachable by
existing agents' open proposals with zero agent code change — the generic payoff.

This reuses the errand planner's shipped LLM-over-menu primitive (`_run_planner`)
and inherits its anti-hallucination: a matched action not present in the
registry is dropped, and params are validated against the declared schema
(`proposable_action_service.validate_against_params`) before anything is
proposed. The system-injected idempotency key is stamped here, not asked of the
LLM.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from app.services.capability_registry import list_proposable_actions
from app.services.proposable_action_service import validate_against_params

logger = logging.getLogger("uvicorn")


def _stable_idempotency_key(data: dict[str, Any], command: str, action: str) -> str:
    """Deterministic dedup key for an open-mode match, so the same detected data
    proposed twice collapses to one action."""
    blob = json.dumps({"d": data, "c": command, "a": action}, sort_keys=True, default=str)
    return "match:" + hashlib.sha256(blob.encode()).hexdigest()[:16]


def _build_menu(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    menu = []
    for a in actions:
        args = {
            p["name"]: (p.get("description") or p.get("type") or "")
            for p in a.get("params", [])
            if p.get("name") != a.get("idempotency_param")  # system-injected, not LLM-filled
        }
        menu.append(
            {
                "command": a["command"],
                "action": a["callback"],
                "description": a.get("card_title") or f"{a['command']} {a['callback']}",
                "args": args,
            }
        )
    return menu


def _build_match_prompt(data: dict[str, Any], menu: list[dict[str, Any]]) -> str:
    menu_text = "\n".join(
        f"- {m['command']}.{m['action']}: {m['description']} | args: {json.dumps(m['args'])}"
        for m in menu
    )
    return (
        "You route detected data to an available ACTION. Given the DATA and the list of "
        "ACTIONS, choose the single best action that genuinely helps with this data, or "
        "NONE if nothing fits. Fill the chosen action's args from the data using the "
        "user's own values. NEVER invent an action or an arg name not listed.\n\n"
        f"ACTIONS:\n{menu_text}\n\n"
        f"DATA:\n{json.dumps(data, default=str)}\n\n"
        'Return JSON: {"matches": [{"command": "<name>", "action": "<name>", "args": {...}}]}. '
        "Return an empty matches list if nothing fits. No markdown, no prose."
    )


async def match_proposals(
    *,
    data: dict[str, Any],
    node_id: str,
    llm_client: Any = None,
    fetch: Any = None,
) -> list[dict[str, Any]]:
    """Match ``data`` against ``node_id``'s advertised proposable actions.

    Returns a list of ``{command, action, args, idempotency_key, spec}`` — each
    already validated against the declared param schema, with the idempotency
    key stamped in. Empty when nothing is advertised or nothing fits.
    """
    actions = await list_proposable_actions(node_id, fetch=fetch)
    if not actions:
        return []
    by_key = {(a["command"], a["callback"]): a for a in actions}
    menu = _build_menu(actions)

    if llm_client is None:
        from app.core.llm_proxy_client import LLMProxyClient

        llm_client = LLMProxyClient()
    from app.services.errand_planner import _run_planner

    try:
        parsed = await _run_planner(_build_match_prompt(data, menu), llm_client)
    except Exception:  # noqa: BLE001 — a planner failure is "no match", never a crash
        logger.warning("proposal_matcher: planner call failed", exc_info=True)
        return []

    results: list[dict[str, Any]] = []
    for m in parsed.get("matches", []) or []:
        key = (m.get("command"), m.get("action"))
        spec = by_key.get(key)
        if spec is None:
            logger.info("proposal_matcher: dropping unadvertised match %s", key)
            continue  # hallucinated / not advertised — the anti-invention guard
        raw_args = dict(m.get("args", {}) or {})
        idem_param = spec.get("idempotency_param")
        idem_key = _stable_idempotency_key(data, spec["command"], spec["callback"])
        if idem_param:
            raw_args[idem_param] = idem_key  # system-injected, not from the LLM
        ok, args, verr = validate_against_params(raw_args, spec.get("params", []))
        if not ok:
            logger.info("proposal_matcher: dropping invalid match %s: %s", key, verr)
            continue
        results.append(
            {
                "command": spec["command"],
                "action": spec["callback"],
                "args": args,
                "idempotency_key": idem_key,
                "spec": spec,
            }
        )
    return results
