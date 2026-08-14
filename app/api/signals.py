"""POST /api/v0/signals — the Signal Bus ingress (open + directed, dual auth).

Mirrors app/api/memories.py. Auth is node-key OR app-to-app; the command
boundary for a directed proposal is the node's LIVE advertisement (no per-source
allowlist in core). An inbound signal can only ever PROPOSE (a tap-to-confirm
card), never execute.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.deps import get_db, verify_api_key
from app.services import capability_registry, proposal_matcher
from app.services.proposal_card import emit_proposal_card
from app.services.signal_service import CacheableSignalError, SignalService

logger = logging.getLogger("uvicorn")

router = APIRouter()


# --- auth -------------------------------------------------------------------
@dataclass
class SignalsAuthContext:
    """Auth result for the signals ingress."""
    auth_type: str            # "node" or "app"
    household_id: str | None = None
    node_id: str | None = None   # the authenticated node (node auth only)


def _get_auth_base_url() -> str:
    return os.getenv("JARVIS_AUTH_URL", "http://localhost:7701")


def _verify_signals_auth(
    x_api_key: str | None = Header(None),
    x_jarvis_app_id: str | None = Header(None),
    x_jarvis_app_key: str | None = Header(None),
    db: Session = Depends(get_db),
) -> SignalsAuthContext:
    """Node API key (auto-derives household) OR app-to-app credentials.

    No per-source key / allowlist — a third party integrates by POSTing with a
    household credential; the command boundary comes from the node.
    """
    if x_api_key:
        try:
            node_ctx = verify_api_key(x_api_key=x_api_key, db=db)
            return SignalsAuthContext(
                auth_type="node",
                household_id=node_ctx.household_id,
                node_id=getattr(getattr(node_ctx, "node", None), "node_id", None),
            )
        except HTTPException:
            pass  # fall through to app-to-app

    if x_jarvis_app_id and x_jarvis_app_key:
        app_ping_url = _get_auth_base_url().rstrip("/") + "/internal/app-ping"
        try:
            resp = httpx.get(
                app_ping_url,
                headers={
                    "X-Jarvis-App-Id": x_jarvis_app_id,
                    "X-Jarvis-App-Key": x_jarvis_app_key,
                },
                timeout=5.0,
            )
            if resp.status_code == 200:
                return SignalsAuthContext(auth_type="app")
        except httpx.RequestError as exc:
            logger.error("Failed to reach jarvis-auth for app auth: %s", exc)
            raise HTTPException(status_code=502, detail="Auth service unavailable") from exc
        raise HTTPException(status_code=401, detail="Invalid app credentials")

    raise HTTPException(
        status_code=401,
        detail="Authentication required (X-Api-Key or X-Jarvis-App-Id/Key)",
    )


# --- payloads ---------------------------------------------------------------
class SignalIn(BaseModel):
    kind: str = Field(..., max_length=255)
    source_key: str = Field(..., max_length=512)
    subject: str | None = Field(None, max_length=255)
    summary: str | None = Field(None, max_length=2000)
    scope: dict | None = None                         # {user_id?, node_id?, room?}
    ttl_seconds: int | None = Field(None, gt=0, le=604800)   # <= 7 days
    cacheable: bool = False
    salience: float | None = None
    source_agent: str = Field("external", max_length=255)


class SignalPostRequest(BaseModel):
    household_id: str | None = None                   # optional — auto-derived from node auth
    signal: SignalIn
    command: str | None = None                        # set → DIRECTED mode
    data: dict | None = None                          # → the Signal's facts


class SignalPostResponse(BaseModel):
    signal_id: int
    mode: str                                         # "open" | "directed"
    proposed: bool


# --- gates (module-level so tests can patch) --------------------------------
_RATE_BUDGET_PER_MIN = 60
_rate_state: dict[tuple[str, str], tuple[float, float]] = {}


def _signals_enabled(household_id: str) -> bool:
    """Fail-OPEN: enabled unless the household explicitly set signals.enabled false."""
    try:
        from app.services.settings_service import get_settings_service

        val = get_settings_service().get("signals.enabled", household_id=household_id)
        if val is not None and str(val).lower() in ("false", "0"):
            return False
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read signals.enabled, proceeding: %s", e)
    return True


def _rate_ok(household_id: str, source_agent: str) -> bool:
    """In-memory token bucket keyed on (household, source_agent). No persistence."""
    key = (household_id, source_agent or "external")
    now = time.monotonic()
    tokens, last = _rate_state.get(key, (float(_RATE_BUDGET_PER_MIN), now))
    tokens = min(
        float(_RATE_BUDGET_PER_MIN),
        tokens + (now - last) * (_RATE_BUDGET_PER_MIN / 60.0),
    )
    if tokens < 1.0:
        _rate_state[key] = (tokens, now)
        return False
    _rate_state[key] = (tokens - 1.0, now)
    return True


def _emit_directed_proposal(
    db: Session,
    household_id: str,
    node_id: str | None,
    command: str,
    args: dict[str, Any] | None,
    source_key: str,
) -> bool:
    """Resolve the named command against the node's advertised proposable actions.

    Returns True if a card was emitted, False if the command isn't advertised
    (refused) or emission failed. On Confirm the card re-validates at the
    jarvis.proposable_action.execute dispatcher before anything runs on the node.
    """
    import asyncio

    # The signal names a command as "command" or explicit "command.callback"; parse it
    # up front so node resolution can be command-aware.
    if "." in command:
        cmd_name, cb_name = command.split(".", 1)
    else:
        cmd_name, cb_name = command, None

    async def _resolve_and_list() -> tuple[str | None, list[dict[str, Any]]]:
        from app.services.proposable_action_service import (
            _household_node_ids_by_recency,
            _probe_node_tools,
            resolve_household_node_for_command,
        )
        nid = node_id
        # scope.node_id is client-supplied and NOT validated upstream against the
        # household. Honor it ONLY if it belongs to this household — a foreign/bogus
        # id must never be contacted (cross-household targeting) nor blindly fetched
        # (a 10s round-trip to an arbitrary id). Otherwise fall to the resolver.
        if nid and nid not in set(_household_node_ids_by_recency(household_id)):
            nid = None
        if not nid:
            # No (valid) node in scope → pick a household node that actually
            # ADVERTISES this command, so multi-node households route correctly.
            nid = await resolve_household_node_for_command(household_id, cmd_name, cb_name)
        if not nid:
            return None, []
        # Short-timeout fetch — _emit_directed_proposal runs on the SYNC /signals
        # threadpool worker, so this must not block on the interactive 10s default.
        return nid, await capability_registry.list_proposable_actions(nid, fetch=_probe_node_tools)

    try:
        node_id, actions = asyncio.run(_resolve_and_list())
    except Exception as e:  # noqa: BLE001
        logger.warning("Directed proposal resolve failed for %s: %s", command, e)
        return False
    if not node_id:
        return False  # no node in the household advertises the command → refused
    # Resolve the parsed command to the node's advertised proposable action.
    if cb_name:
        action = next((a for a in (actions or [])
                       if a.get("command") == cmd_name and a.get("callback") == cb_name), None)
    else:
        action = next((a for a in (actions or []) if a.get("command") == cmd_name), None)
    command = cmd_name
    if not action:
        return False  # not advertised by the resolved node → refused, no card

    try:
        callback = action.get("callback") or command
        declared = {p.get("name") for p in action.get("params", []) if p.get("name")}
        idem_param = action.get("idempotency_param")
        params = {
            k: v
            for k, v in (args or {}).items()
            if k in declared and k != idem_param
        }
        idem = proposal_matcher._stable_idempotency_key(
            {"items": [args or {}]}, command, callback
        )
        title = action.get("card_title") or f"Run {command}?"
        return emit_proposal_card(
            household_id=household_id,
            node_id=node_id,
            command=command,
            callback=callback,
            params=params,
            idempotency_key=idem,
            card_title=title,
            summary=title,
            source=source_key,
        )
    except Exception as e:  # noqa: BLE001 — best-effort; never break ingest
        logger.warning("Directed proposal emit failed for %s: %s", command, e)
        return False


# --- endpoint ---------------------------------------------------------------
@router.post("/signals", status_code=200)
def post_signal(
    body: SignalPostRequest,
    auth: SignalsAuthContext = Depends(_verify_signals_auth),
    db: Session = Depends(get_db),
) -> SignalPostResponse:
    """Ingest one Signal. Open mode persists for the matcher; directed mode also
    proposes the named command (if the node advertises it)."""
    household_id = body.household_id or auth.household_id
    if not household_id:
        raise HTTPException(
            status_code=400,
            detail="household_id required (provide in body or use node auth)",
        )
    if auth.auth_type == "node" and body.household_id and auth.household_id:
        if body.household_id != auth.household_id:
            raise HTTPException(
                status_code=403,
                detail="household_id does not match node's household",
            )

    if not _signals_enabled(household_id):
        raise HTTPException(status_code=409, detail="Signal bus is disabled for this household")
    if not _rate_ok(household_id, body.signal.source_agent):
        raise HTTPException(
            status_code=429, detail="Rate limit exceeded", headers={"Retry-After": "1"}
        )

    scope = body.signal.scope or {}
    # A producer may omit scope.node_id (an agent rarely knows its own node id). Fall
    # back to the authenticated node — the node that POSTed the signal is the natural
    # target for a node-scoped reaction (e.g. running get_drive_time for leave-by).
    node_id = scope.get("node_id") or auth.node_id

    service = SignalService(db)
    try:
        sig = service.save_signal(
            household_id=household_id,
            source_key=body.signal.source_key,
            kind=body.signal.kind,
            subject=body.signal.subject,
            summary=body.signal.summary,
            facts=body.data,
            user_id=scope.get("user_id"),
            node_id=node_id,
            room=scope.get("room"),
            ttl_seconds=body.signal.ttl_seconds,
            cacheable=body.signal.cacheable,
            salience=body.signal.salience,
            source_agent=body.signal.source_agent,
        )
    except CacheableSignalError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    proposed = False
    if body.command:
        proposed = _emit_directed_proposal(
            db, household_id, node_id, body.command, body.data, body.signal.source_key
        )

    # Signal Bus: a new/changed Signal is an edge — schedule a debounced proactive
    # reason pass (fire-and-forget; gated + off-by-default downstream). Adds no
    # latency to ingest and never raises.
    try:
        from app.services.situation_matcher_service import signal_situation_edge

        signal_situation_edge(household_id, node_id)
    except Exception:  # noqa: BLE001 — edge scheduling is best-effort
        pass

    # Signal Bus DETERMINISTIC reactions: fan the signal out to every reaction
    # registered for its kind (generic — ingest names no kind and no reaction).
    # Fire-and-forget; never blocks ingest. A no-op for kinds with no reaction.
    try:
        from app.services.signal_reaction_registry import (
            ReactionContext,
            schedule_signal_reactions,
        )

        schedule_signal_reactions(
            ReactionContext(
                household_id=household_id,
                node_id=node_id,
                user_id=scope.get("user_id"),
                kind=body.signal.kind,
                facts=body.data or {},
            )
        )
    except Exception:  # noqa: BLE001 — reaction scheduling is best-effort
        pass

    return SignalPostResponse(
        signal_id=sig.id,
        mode="directed" if body.command else "open",
        proposed=proposed,
    )
