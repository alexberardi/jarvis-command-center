"""Routines API — server-owned, per-household routines.

Three audiences share this router (mounted at /api/v0):
  - MOBILE (JWT, household-scoped via verify_provisioning_auth + require_household_access):
      CRUD under /households/{household_id}/routines  + /run-now (see routines_run.py wiring)
  - NODE (X-API-Key via verify_api_key):
      GET /nodes/{node_id}/routines — the pull-on-nudge source. Returns the household
      routine set as a slug-keyed map with step args FLATTENED to {k:v}.

On every mutation we publish a lightweight MQTT "routines changed" nudge to each
active household node; the node responds by pulling this endpoint and rewriting its
local routine store. The routine JSON is NOT sent over MQTT (it is plaintext
household data fetched over the node's authenticated HTTP channel).

Execution stays entirely on-node via the existing RoutineCommand engine.
"""

import asyncio
import json
import logging
import os
import re
import tempfile
import time
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import get_db, verify_api_key
from app.models import Node, Routine, RoutineExecution
from app.provisioning import (
    ProvisioningAuthContext,
    require_household_access,
    verify_provisioning_auth,
)

router = APIRouter()
logger = logging.getLogger("uvicorn")

VALID_RESPONSE_LENGTHS = {"short", "medium", "long"}

# Node POSTs routine results to /device-control-results/{request_id} (smart_home.py),
# which writes this dir. We poll it for the run-now / scheduled result.
_RESULT_DIR = os.path.join(tempfile.gettempdir(), "jarvis-device-control")


# ── Pydantic request/response shapes ─────────────────────────────────────────


class RoutineStepArg(BaseModel):
    key: str
    # The editor stores every value as a string. Array/object-typed params
    # (e.g. resolved_datetimes) are stored as a JSON string like '["today"]'
    # and parsed back at the node-pull boundary (_flatten_args).
    value: str = ""


class RoutineStep(BaseModel):
    command: str
    args: list[RoutineStepArg] = []
    label: str = ""


class RoutineSchedule(BaseModel):
    type: str  # "cron" | "interval"
    cron: str | None = None
    interval_seconds: int | None = None
    timezone: str = "UTC"
    # Which node a scheduled run fires on. Defaults to the household primary node
    # (smart_home.primary_node_id) when omitted.
    target_node_id: str | None = None
    enabled: bool = True
    last_fired_at: str | None = None


class RoutineCreate(BaseModel):
    name: str
    trigger_phrases: list[str] = []
    steps: list[RoutineStep] = []
    response_instruction: str = ""
    response_length: str = "short"
    schedule: RoutineSchedule | None = None
    enabled: bool = True


class RoutineUpdate(BaseModel):
    # All optional — PATCH semantics. Use model_dump(exclude_unset=True) to tell
    # "not provided" apart from "explicitly set to null" (e.g. clearing schedule).
    name: str | None = None
    trigger_phrases: list[str] | None = None
    steps: list[RoutineStep] | None = None
    response_instruction: str | None = None
    response_length: str | None = None
    schedule: RoutineSchedule | None = None
    enabled: bool | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", (name or "").strip().lower()).strip("_")
    return slug or "routine"


def _unique_slug(db: Session, household_id: str, name: str) -> str:
    base = _slugify(name)
    existing = {
        r.slug
        for r in db.query(Routine.slug).filter(Routine.household_id == household_id).all()
    }
    if base not in existing:
        return base
    i = 2
    while f"{base}_{i}" in existing:
        i += 1
    return f"{base}_{i}"


def _flatten_args(args_list: list) -> dict:
    """Convert mobile-native args [{key,value}] -> node-native {k: v}.

    Values are strings. A value that begins with '[' or '{' is parsed as JSON so
    array/object-typed params (resolved_datetimes -> ["today"]) round-trip
    correctly; plain scalars stay strings and the node coerces them at runtime.
    """
    out: dict = {}
    for pair in args_list or []:
        if not isinstance(pair, dict):
            continue
        key = pair.get("key")
        if not key:
            continue
        value = pair.get("value", "")
        if isinstance(value, str) and value[:1] in ("[", "{"):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        out[key] = value
    return out


def _routine_to_mobile(r: Routine) -> dict:
    """Mobile-facing shape: args stay as the [{key,value}] array."""
    return {
        "id": r.id,
        "slug": r.slug,
        "name": r.name,
        "trigger_phrases": json.loads(r.trigger_phrases or "[]"),
        "steps": json.loads(r.steps or "[]"),
        "response_instruction": r.response_instruction or "",
        "response_length": r.response_length or "short",
        "schedule": json.loads(r.schedule) if r.schedule else None,
        "enabled": r.enabled,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "updated_at": r.updated_at.isoformat() if r.updated_at else None,
    }


def _routine_to_node(r: Routine) -> dict:
    """Node-facing definition (matches _load_routines): args FLATTENED to {k:v}."""
    steps = json.loads(r.steps or "[]")
    node_steps = [
        {
            "command": s.get("command"),
            "args": _flatten_args(s.get("args", [])),
            "label": s.get("label", ""),
        }
        for s in steps
    ]
    return {
        "trigger_phrases": json.loads(r.trigger_phrases or "[]"),
        "steps": node_steps,
        "response_instruction": r.response_instruction or "",
        "response_length": r.response_length or "short",
    }


def _resolve_target_node_id(household_id: str, provided: str | None) -> str | None:
    if provided:
        return provided
    try:
        from app.services.settings_service import get_settings_service

        return get_settings_service().get(
            "smart_home.primary_node_id", household_id=household_id
        ) or None
    except Exception as exc:  # settings unavailable — leave unset, scheduler skips
        logger.warning("Could not resolve primary node for household %s: %s", household_id, exc)
        return None


def _validate_response_length(value: str) -> None:
    if value not in VALID_RESPONSE_LENGTHS:
        raise HTTPException(
            status_code=422,
            detail=f"response_length must be one of {sorted(VALID_RESPONSE_LENGTHS)}",
        )


def _validate_schedule(sched: RoutineSchedule) -> None:
    if sched.type not in ("cron", "interval"):
        raise HTTPException(status_code=422, detail="schedule.type must be 'cron' or 'interval'")
    if sched.type == "cron" and not (sched.cron and sched.cron.strip()):
        raise HTTPException(status_code=422, detail="cron schedule requires a 'cron' expression")
    if sched.type == "interval" and not (sched.interval_seconds and sched.interval_seconds > 0):
        raise HTTPException(status_code=422, detail="interval schedule requires interval_seconds > 0")


def _schedule_to_json(household_id: str, sched: RoutineSchedule | None) -> str | None:
    if sched is None:
        return None
    _validate_schedule(sched)
    data = sched.model_dump()
    data["target_node_id"] = _resolve_target_node_id(household_id, sched.target_node_id)
    return json.dumps(data)


def publish_routines_sync(household_id: str, db: Session) -> None:
    """Nudge every active household node to re-pull its routine set."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        logger.warning("MQTT unavailable — routines sync nudge skipped for %s", household_id)
        return
    nodes = (
        db.query(Node)
        .filter(Node.household_id == household_id, Node.is_active.is_(True))
        .all()
    )
    payload = json.dumps({"event": "routines_changed", "household_id": household_id})
    for n in nodes:
        try:
            client.publish(f"jarvis/nodes/{n.node_id}/routines/sync", payload)
        except Exception as exc:  # never fail the request on a publish error
            logger.warning("routines/sync publish failed for %s: %s", n.node_id, exc)
    logger.info("📤 routines/sync nudged %d node(s) in household %s", len(nodes), household_id)


# ── Mobile CRUD (JWT, household-scoped) ──────────────────────────────────────


@router.get("/households/{household_id}/routines")
def list_routines(
    household_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    require_household_access(household_id, auth)
    rows = (
        db.query(Routine)
        .filter(Routine.household_id == household_id)
        .order_by(Routine.created_at.asc())
        .all()
    )
    return {"routines": [_routine_to_mobile(r) for r in rows]}


@router.post("/households/{household_id}/routines", status_code=201)
def create_routine(
    household_id: str,
    body: RoutineCreate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    require_household_access(household_id, auth)
    if not body.name or not body.name.strip():
        raise HTTPException(status_code=422, detail="name is required")
    _validate_response_length(body.response_length)

    routine = Routine(
        id=str(uuid4()),
        household_id=household_id,
        slug=_unique_slug(db, household_id, body.name),
        name=body.name.strip(),
        trigger_phrases=json.dumps(body.trigger_phrases),
        steps=json.dumps([s.model_dump() for s in body.steps]),
        response_instruction=body.response_instruction or "",
        response_length=body.response_length,
        schedule=_schedule_to_json(household_id, body.schedule),
        enabled=body.enabled,
    )
    db.add(routine)
    db.commit()
    db.refresh(routine)
    publish_routines_sync(household_id, db)
    return _routine_to_mobile(routine)


def _get_or_404(db: Session, household_id: str, routine_id: str) -> Routine:
    routine = (
        db.query(Routine)
        .filter(Routine.id == routine_id, Routine.household_id == household_id)
        .first()
    )
    if not routine:
        raise HTTPException(status_code=404, detail="Routine not found")
    return routine


@router.get("/households/{household_id}/routines/{routine_id}")
def get_routine(
    household_id: str,
    routine_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    require_household_access(household_id, auth)
    return _routine_to_mobile(_get_or_404(db, household_id, routine_id))


@router.patch("/households/{household_id}/routines/{routine_id}")
def update_routine(
    household_id: str,
    routine_id: str,
    body: RoutineUpdate,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    require_household_access(household_id, auth)
    routine = _get_or_404(db, household_id, routine_id)

    provided = body.model_dump(exclude_unset=True)
    if "name" in provided and body.name:
        routine.name = body.name.strip()
    if "trigger_phrases" in provided and body.trigger_phrases is not None:
        routine.trigger_phrases = json.dumps(body.trigger_phrases)
    if "steps" in provided and body.steps is not None:
        routine.steps = json.dumps([s.model_dump() for s in body.steps])
    if "response_instruction" in provided and body.response_instruction is not None:
        routine.response_instruction = body.response_instruction
    if "response_length" in provided and body.response_length is not None:
        _validate_response_length(body.response_length)
        routine.response_length = body.response_length
    if "schedule" in provided:
        routine.schedule = _schedule_to_json(household_id, body.schedule)
    if "enabled" in provided and body.enabled is not None:
        routine.enabled = body.enabled

    db.commit()
    db.refresh(routine)
    publish_routines_sync(household_id, db)
    return _routine_to_mobile(routine)


@router.delete("/households/{household_id}/routines/{routine_id}", status_code=204)
def delete_routine(
    household_id: str,
    routine_id: str,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> None:
    require_household_access(household_id, auth)
    routine = _get_or_404(db, household_id, routine_id)
    db.delete(routine)
    db.commit()
    publish_routines_sync(household_id, db)


# ── Node pull (X-API-Key) ────────────────────────────────────────────────────


@router.get("/nodes/{node_id}/routines")
def node_pull_routines(
    node_id: str,
    node_context: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Pull-on-nudge source. Returns the household routine set as a slug-keyed
    map with args flattened to {k:v}. A node may only pull its own routines."""
    if node_context.node.node_id != node_id:
        raise HTTPException(status_code=403, detail="Node mismatch")

    household_id = node_context.household_id
    if not household_id:
        # Dev/container nodes with no household get an empty (but valid) set.
        return {"routines": {}}

    rows = (
        db.query(Routine)
        .filter(Routine.household_id == household_id, Routine.enabled.is_(True))
        .all()
    )
    return {"routines": {r.slug: _routine_to_node(r) for r in rows}}


# ── Run-now + scheduled execution ────────────────────────────────────────────


class RunNowRequest(BaseModel):
    # Which node to run on. Defaults to the household primary node when omitted.
    node_id: str | None = None


async def _wait_for_result_file(request_id: str, timeout: float = 20.0) -> dict | None:
    """Poll for the result file the node writes via /device-control-results/{id}."""
    os.makedirs(_RESULT_DIR, exist_ok=True)
    result_file = os.path.join(_RESULT_DIR, f"{request_id}.json")
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(result_file):
            try:
                with open(result_file) as f:
                    result = json.load(f)
                os.unlink(result_file)
                return result
            except (json.JSONDecodeError, OSError):
                pass
        await asyncio.sleep(0.1)
    try:
        os.unlink(result_file)
    except OSError:
        pass
    return None


def _record_execution(
    db: Session,
    routine: Routine,
    household_id: str,
    node_id: str,
    trigger: str,
    status: str,
    passed: int,
    failed: int,
    error: str | None,
    started_at: datetime,
) -> None:
    try:
        db.add(
            RoutineExecution(
                id=str(uuid4()),
                routine_id=routine.id,
                household_id=household_id,
                node_id=node_id,
                trigger=trigger,
                status=status,
                passed=passed,
                failed=failed,
                error=error,
                started_at=started_at,
                finished_at=datetime.utcnow(),
            )
        )
        db.commit()
    except Exception as exc:  # an audit-row failure must not fail the run
        logger.warning("Failed to record routine execution: %s", exc)
        db.rollback()


# Status → (icon, headline) for the completion card. Honest on EVERY terminal
# state: a routine you didn't trigger (scheduled/background) must tell you it
# finished — including when it timed out or partly failed — never stay silent.
_ROUTINE_STATUS_CARD = {
    "success": ("✅", "finished"),
    "partial": ("⚠️", "finished with issues"),
    "timeout": ("⏱️", "didn't finish"),
    "failed": ("⚠️", "couldn't run"),
}


def _notify_routine_complete(
    routine: Routine,
    household_id: str,
    result: dict,
    user_id: int | None = None,
) -> None:
    """Post ONE completion card + push for a detached routine run.

    Non-fatal by construction — a notification failure must never abort or fail
    the run (mirrors phone ``post_outcome_card``). Targets the triggering user
    when known, else the whole household (a scheduled run has no owner).
    """
    try:
        from app.services.inbox_notification_service import post_inbox_item_sync

        status = str(result.get("status") or "failed")
        icon, headline = _ROUTINE_STATUS_CARD.get(status, _ROUTINE_STATUS_CARD["failed"])
        message = result.get("message")
        if status in ("success", "partial") and message:
            summary = message
        elif status == "timeout":
            summary = "The node didn't respond in time."
        else:
            summary = message or result.get("error") or "The routine did not complete."

        post_inbox_item_sync(
            household_id=household_id,
            title=f"{icon} '{routine.name}' {headline}",
            summary=summary,
            body=summary,
            category="routine",
            metadata={
                "household_id": household_id,
                "routine_id": routine.id,
                "status": status,
            },
            user_id=user_id,
            target_type="user" if user_id is not None else "household",
            push=True,
        )
    except Exception as exc:  # noqa: BLE001 — a notification must never fail the run
        logger.warning("Routine completion notify failed for '%s': %s", routine.slug, exc)


async def execute_routine_on_node(
    db: Session,
    household_id: str,
    routine: Routine,
    node_id: str,
    trigger: str,
    *,
    notify_on_complete: bool = False,
    notify_user_id: int | None = None,
) -> dict:
    """Publish a 'routine' command to a node, await its result, record the run.

    The whole routine runs through the node's RoutineCommand engine (one command,
    not N tool_calls) so cross-step aggregation + the single composed reply are
    preserved. Returns {success, status, message, passed, failed}.

    When ``notify_on_complete`` is set, pushes one completion card on every
    terminal state (success/partial/timeout/failed) — used by detached runs
    (scheduler, background voice) where no caller is waiting on the return.
    ``notify_user_id`` targets that user; ``None`` fans out to the household.
    """
    from app.services.node_command_service import get_node_command_service

    request_id = str(uuid4())
    slug = routine.slug
    details = {
        "routine_name": slug,
        "reply_request_id": request_id,
        "tool_call_id": str(uuid4()),
        "trusted": True,
        "voice_command": f"routine: {slug}",
    }
    started = datetime.utcnow()
    os.makedirs(_RESULT_DIR, exist_ok=True)

    try:
        get_node_command_service().publish_command_with_id(node_id, "routine", details, request_id)
    except Exception as exc:
        logger.error("Run routine publish failed for %s on %s: %s", slug, node_id, exc)
        _record_execution(db, routine, household_id, node_id, trigger, "failed", 0, 0, str(exc), started)
        result = {"success": False, "status": "failed", "message": None, "passed": 0, "failed": 0, "error": str(exc)}
        if notify_on_complete:
            _notify_routine_complete(routine, household_id, result, notify_user_id)
        return result

    result = await _wait_for_result_file(request_id)
    if result is None:
        _record_execution(db, routine, household_id, node_id, trigger, "timeout", 0, 0, "node did not respond", started)
        result = {"success": False, "status": "timeout", "message": None, "passed": 0, "failed": 0}
        if notify_on_complete:
            _notify_routine_complete(routine, household_id, result, notify_user_id)
        return result

    output = result.get("output", {}) or {}
    passed = int(output.get("passed", 0) or 0)
    failed = int(output.get("failed", 0) or 0)
    success = bool(output.get("success", False))
    message = output.get("message")
    status = "failed" if not success else ("partial" if failed > 0 else "success")
    _record_execution(db, routine, household_id, node_id, trigger, status, passed, failed, output.get("error"), started)
    result = {"success": success, "status": status, "message": message, "passed": passed, "failed": failed}
    if notify_on_complete:
        _notify_routine_complete(routine, household_id, result, notify_user_id)
    return result


@router.post("/households/{household_id}/routines/{routine_id}/run-now")
async def run_routine_now(
    household_id: str,
    routine_id: str,
    body: RunNowRequest,
    auth: ProvisioningAuthContext = Depends(verify_provisioning_auth),
    db: Session = Depends(get_db),
) -> dict:
    require_household_access(household_id, auth)
    routine = _get_or_404(db, household_id, routine_id)
    node_id = body.node_id or _resolve_target_node_id(household_id, None)
    if not node_id:
        raise HTTPException(
            status_code=400,
            detail="No node specified and no household primary node configured",
        )
    return await execute_routine_on_node(db, household_id, routine, node_id, "run_now")


class RunBackgroundRequest(BaseModel):
    routine_slug: str
    speaker_user_id: int | None = None


async def _run_routine_background_task(
    household_id: str, routine_id: str, node_id: str, speaker_user_id: int | None
) -> None:
    """Detached background routine run (voice "… in the background").

    Owns its OWN DB session — the request's session is closed by the time this
    task runs — and always lands a completion card, even if the run raises
    (anti-vanishing, mirroring the phone-call fire-and-forget rule).
    """
    from app.db import get_session_local

    db = get_session_local()()
    try:
        routine = db.query(Routine).filter(Routine.id == routine_id).first()
        if routine is None:
            return
        await execute_routine_on_node(
            db, household_id, routine, node_id, "background",
            notify_on_complete=True, notify_user_id=speaker_user_id,
        )
    except Exception:
        logger.exception("Background routine run failed for %s", routine_id)
        try:
            routine = db.query(Routine).filter(Routine.id == routine_id).first()
            if routine is not None:
                _notify_routine_complete(
                    routine, household_id,
                    {"status": "failed", "message": None, "error": "the routine could not be started"},
                    speaker_user_id,
                )
        except Exception:  # noqa: BLE001 — the failure card is best-effort
            logger.warning("Background routine failure-card also failed for %s", routine_id)
    finally:
        db.close()


@router.post("/routines/run-background")
async def run_routine_in_background(
    body: RunBackgroundRequest,
    node_ctx: NodeContextProvider = Depends(verify_api_key),
    db: Session = Depends(get_db),
) -> dict:
    """Voice "run <routine> in the background" → dispatch a detached run on the
    calling node and return an instant ack; the completion card lands when done.

    Node-authenticated: household + executing node come from the validated node
    row, not the payload. The speaker (for notification targeting) is passed by
    the node; absent → the whole household is notified.
    """
    household_id = node_ctx.household_id
    node = getattr(node_ctx, "node", None)
    if not household_id or node is None:
        raise HTTPException(status_code=403, detail="Node is not associated with a household")

    routine = (
        db.query(Routine)
        .filter(Routine.slug == body.routine_slug, Routine.household_id == household_id)
        .first()
    )
    if routine is None:
        raise HTTPException(status_code=404, detail=f"Routine '{body.routine_slug}' not found")

    asyncio.create_task(
        _run_routine_background_task(household_id, routine.id, node.node_id, body.speaker_user_id)
    )
    logger.info("Dispatched background routine '%s' on node %s", routine.slug, node.node_id)
    return {"status": "dispatched", "routine": routine.name}
