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

The post-processing loop is factored into ``finalize_situation_matches`` so the
LIVE path (``match_situation``) and the BACKGROUND proactive path (the situation
matcher's queue callback) apply identical rules to the LLM's raw output.
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


def _build_situation_prompt(
    bundle: list[dict[str, Any]], menu: list[dict[str, Any]]
) -> str:
    menu_text = "\n".join(
        f"- {m['command']}.{m['action']}: {m['description']} | args: {json.dumps(m['args'])}"
        for m in menu
    )
    items_text = "\n".join(
        f"#{i}: {json.dumps(item.get('data', {}), default=str)}"
        for i, item in enumerate(bundle)
    )
    return (
        "You watch the household SITUATION (a list of current observations) and an "
        "available ACTION menu. Choose the single best action that genuinely helps "
        "right now, or NONE if nothing fits. Fill the chosen action's args from the "
        "observations using the user's own values. Cite the observation indices you "
        'used in a "sources" list. NEVER invent an action or an arg name not listed.\n\n'
        f"ACTIONS:\n{menu_text}\n\n"
        f"SITUATION:\n{items_text}\n\n"
        'Return JSON: {"matches": [{"command": "<name>", "action": "<name>", '
        '"args": {...}, "sources": [<indices>]}]}. '
        "Return an empty matches list if nothing fits. No markdown, no prose."
    )


def finalize_situation_matches(
    parsed: dict[str, Any],
    bundle: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Turn a parsed LLM ``{"matches": [...]}`` into validated proposal results.

    Pure (no I/O). Shared by :func:`match_situation` (live) and the proactive
    queue callback (background) so both apply the same rules: drop matches not in
    ``actions`` (anti-hallucination), resolve the contributing observation indices
    (the LLM's cited ``sources``, out-of-range dropped; none cited → all), collect
    their ``source_keys``, stamp an idempotency key scoped to ONLY the contributing
    observations' data, and validate the args against the declared schema.
    """
    by_key = {(a["command"], a["callback"]): a for a in actions}
    results: list[dict[str, Any]] = []
    for m in parsed.get("matches", []) or []:
        key = (m.get("command"), m.get("action"))
        spec = by_key.get(key)
        if spec is None:
            logger.info("match_situation: dropping unadvertised match %s", key)
            continue  # hallucinated / not advertised — the anti-invention guard

        raw = m.get("sources")
        if raw:  # the LLM cited specific observations
            contributing = [i for i in raw if isinstance(i, int) and 0 <= i < len(bundle)]
        else:  # none cited → treat every observation as contributing
            contributing = list(range(len(bundle)))
        contributing_keys = [
            bundle[i].get("source_key") for i in contributing if bundle[i].get("source_key")
        ]

        # Idempotency scoped to ONLY the contributing observations' data: the same
        # situation proposing the same action collapses to one key, while unrelated
        # noise elsewhere in the bundle never changes it.
        idem_key = _stable_idempotency_key(
            {"items": [bundle[i].get("data", {}) for i in contributing]},
            spec["command"],
            spec["callback"],
        )
        raw_args = dict(m.get("args", {}) or {})
        idem_param = spec.get("idempotency_param")
        if idem_param:
            raw_args[idem_param] = idem_key  # system-injected, not from the LLM
        ok, args, verr = validate_against_params(raw_args, spec.get("params", []))
        if not ok:
            logger.info("match_situation: dropping invalid match %s: %s", key, verr)
            continue
        results.append(
            {
                "command": spec["command"],
                "action": spec["callback"],
                "args": args,
                "idempotency_key": idem_key,
                "source_keys": contributing_keys,
                "spec": spec,
            }
        )
    return results


async def match_situation(
    *,
    bundle: list[dict[str, Any]],
    node_id: str,
    llm_client: Any = None,
    fetch: Any = None,
) -> list[dict[str, Any]]:
    """Match a BUNDLE of observations against ``node_id``'s advertised proposable
    actions.

    Each bundle item is ``{"source_key": <str|None>, "data": {...}}``. Returns a
    list of ``{command, action, args, idempotency_key, source_keys, spec}`` — each
    validated against the declared param schema, with the idempotency key stamped
    in and scoped to ONLY the observations the LLM cited (``source_keys``). Empty
    when nothing is advertised or nothing fits.
    """
    actions = await list_proposable_actions(node_id, fetch=fetch)
    if not actions:
        return []
    menu = _build_menu(actions)

    if llm_client is None:
        from app.core.llm_proxy_client import LLMProxyClient

        llm_client = LLMProxyClient()
    from app.services.errand_planner import _run_planner

    try:
        parsed = await _run_planner(_build_situation_prompt(bundle, menu), llm_client)
    except Exception:  # noqa: BLE001 — a planner failure is "no match", never a crash
        logger.warning("match_situation: planner call failed", exc_info=True)
        return []

    return finalize_situation_matches(parsed, bundle, actions)


async def match_proposals(
    *,
    data: dict[str, Any],
    node_id: str,
    llm_client: Any = None,
    fetch: Any = None,
) -> list[dict[str, Any]]:
    """Single-observation adapter over :func:`match_situation` — the shipped
    open-mode entry point (one opaque ``data`` dict). Kept so the email→calendar
    pilot is untouched."""
    return await match_situation(
        bundle=[{"source_key": None, "data": data}],
        node_id=node_id,
        llm_client=llm_client,
        fetch=fetch,
    )
