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
from app.services.signal_service import CacheableSignalError, SignalService

logger = logging.getLogger("uvicorn")

router = APIRouter()


# --- auth -------------------------------------------------------------------
@dataclass
class SignalsAuthContext:
    """Auth result for the signals ingress."""
    auth_type: str            # "node" or "app"
    household_id: str | None = None


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
            return SignalsAuthContext(auth_type="node", household_id=node_ctx.household_id)
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

    Returns True if a proposal was emitted, False if the command isn't advertised
    (refused). The real proposable-card emission is wired with the matcher in
    Phase 3 — this is the patchable seam the ingress calls.
    """
    try:
        import asyncio

        from app.services import capability_registry

        action = asyncio.run(
            capability_registry.resolve_proposable_action(node_id, command)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Directed proposal resolve failed for %s: %s", command, e)
        return False
    if not action:
        return False
    logger.info("Directed proposal accepted: command=%s source_key=%s", command, source_key)
    return True  # Phase 3 posts the card


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
    node_id = scope.get("node_id")

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

    return SignalPostResponse(
        signal_id=sig.id,
        mode="directed" if body.command else "open",
        proposed=proposed,
    )
