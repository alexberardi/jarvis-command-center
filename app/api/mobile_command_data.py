"""Mobile command-data browser API.

Surfaces per-command JarvisStorage records on each node so users can
browse, edit, and delete reminders (and other future commands) from the
mobile app without SSH.

Auth: JWT (mobile user). Each request resolves the node, verifies it
belongs to a household the user has access to, then publishes a
correlation-id'd MQTT request and awaits the node's response on the
per-correlation response topic.

Pattern note: this is the first route in CC to use the subscribe-before-
publish MQTT request/response pattern (`MQTTClient.request_response`).
The existing mobile MQTT routes (bluetooth.py, node_commands.py,
node_tools.py) all use the file-polling callback pattern — that adds
~100ms polling latency that hurts interactive CRUD UX, where the user is
sitting and waiting. New routes for interactive surfaces should copy
this pattern; fire-and-forget operations stay on file/DB callbacks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.deps import (
    AuthenticatedUser,
    JARVIS_APP_ID,
    JARVIS_APP_KEY,
    _get_auth_base_url,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.models import Node

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-command-data"])

MQTT_TIMEOUT_SECONDS = 10.0
SCHEMA_CACHE_TTL_SECONDS = 600  # 10 minutes


# ── Schema cache ────────────────────────────────────────────────────────────


class _SchemaCache:
    """Per-process TTL cache of `(node_id, command_name)` → schema dict.

    Mirrors the shape of `app.core.conversation_cache.ConversationCache`:
    in-process dict, threading.Lock, 10-minute TTL. Schema rarely changes
    within a 10-minute window, so the cache eliminates a node round-trip
    when mobile opens an edit form right after a list view.

    No reconnect invalidation — there's no node-reconnect event in CC
    today (`MQTTClient._on_connect` only logs). A stale cache surfaces as
    a PATCH validation error from the node, which mobile can recover
    from by re-fetching the schema."""

    def __init__(self, ttl: float = SCHEMA_CACHE_TTL_SECONDS) -> None:
        self._store: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}
        self._ttl = ttl
        self._lock = threading.Lock()

    def get(self, node_id: str, command_name: str) -> Optional[dict[str, Any]]:
        key = (node_id, command_name)
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, ts = entry
            if now - ts > self._ttl:
                del self._store[key]
                return None
            return value

    def put(self, node_id: str, command_name: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._store[(node_id, command_name)] = (value, time.time())

    def invalidate(self, node_id: str, command_name: str) -> None:
        with self._lock:
            self._store.pop((node_id, command_name), None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_schema_cache = _SchemaCache()


# ── MQTT round-trip ─────────────────────────────────────────────────────────


async def _mqtt_request(
    node_id: str,
    op: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Subscribe-before-publish round-trip with the node.

    Generates a correlation_id (or reuses one in the payload), subscribes
    to `jarvis/nodes/{node_id}/command-data/{op}/response/{correlation_id}`,
    publishes the request, awaits one response within MQTT_TIMEOUT_SECONDS.
    Raises HTTPException(504) on timeout, 503 if MQTT is unavailable."""
    from app.node_settings import get_mqtt_client

    client = get_mqtt_client()
    if client is None:
        raise HTTPException(status_code=503, detail="MQTT client not available")

    correlation_id = payload.setdefault("correlation_id", str(uuid.uuid4()))
    request_topic = f"jarvis/nodes/{node_id}/command-data/{op}"
    response_topic = f"{request_topic}/response/{correlation_id}"

    raw = await asyncio.to_thread(
        client.request_response,
        request_topic,
        response_topic,
        json.dumps(payload),
        MQTT_TIMEOUT_SECONDS,
    )

    if raw is None:
        raise HTTPException(
            status_code=504,
            detail=f"Node {node_id} did not respond within {MQTT_TIMEOUT_SECONDS}s",
        )

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(
            "Node returned non-JSON command-data response",
            extra={"node_id": node_id, "op": op},
        )
        raise HTTPException(
            status_code=502, detail="Node returned invalid response"
        ) from exc


# ── Auth scoping ────────────────────────────────────────────────────────────


def _resolve_node_in_household(
    db: Session, node_id: str, user: AuthenticatedUser,
) -> Node:
    """Resolve the node, verify the user has access to its household.

    Centralizes a check that's currently inlined in several mobile routes
    (mobile_chat.py validates by household_id from the body;
    node_commands.py skips the check entirely). Future mobile routes
    should call this helper instead of rolling their own."""
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if node is None:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    if node.household_id:
        verify_household_role(user.user_id, node.household_id, required_role="member")
    return node


# ── user_ref resolution ─────────────────────────────────────────────────────


_USER_NAME_CACHE: dict[int, tuple[str, float]] = {}
_USER_NAME_CACHE_TTL = 300.0
_USER_NAME_CACHE_LOCK = threading.Lock()


async def _resolve_user_names(user_ids: set[int]) -> dict[int, str]:
    """Resolve a batch of user_ids to display names via jarvis-auth.

    Uses a small per-process TTL cache to avoid one auth call per
    record-row when the same owner is repeated. Falls back gracefully:
    unknown user_ids omitted from the result, mobile renders the raw id."""
    if not user_ids:
        return {}

    now = time.time()
    resolved: dict[int, str] = {}
    missing: list[int] = []
    with _USER_NAME_CACHE_LOCK:
        for uid in user_ids:
            entry = _USER_NAME_CACHE.get(uid)
            if entry is not None and now - entry[1] <= _USER_NAME_CACHE_TTL:
                resolved[uid] = entry[0]
            else:
                missing.append(uid)

    if not missing:
        return resolved

    if not JARVIS_APP_KEY:
        return resolved

    auth_url = _get_auth_base_url().rstrip("/")
    url = f"{auth_url}/internal/users/batch?user_ids={','.join(str(u) for u in missing)}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url,
                headers={
                    "X-Jarvis-App-Id": JARVIS_APP_ID,
                    "X-Jarvis-App-Key": JARVIS_APP_KEY,
                },
            )
    except httpx.RequestError as exc:
        logger.warning("Failed to batch-resolve user names: %s", exc)
        return resolved

    if resp.status_code != 200:
        logger.warning("Auth /internal/users/batch returned %s", resp.status_code)
        return resolved

    body = resp.json()
    users = body.get("users") if isinstance(body, dict) else None
    if not isinstance(users, dict):
        return resolved

    with _USER_NAME_CACHE_LOCK:
        for uid_str, name in users.items():
            try:
                uid = int(uid_str)
            except (TypeError, ValueError):
                continue
            if isinstance(name, str) and name:
                resolved[uid] = name
                _USER_NAME_CACHE[uid] = (name, now)
    return resolved


def _collect_user_ref_field_names(fields: list[dict[str, Any]]) -> list[str]:
    """Walk a schema's FieldSpec list (including nested object schemas) and
    return the names of every `user_ref` field. Used to know which field
    values to send to auth for display-name resolution."""
    names: list[str] = []
    for f in fields:
        if not isinstance(f, dict):
            continue
        if f.get("type") == "user_ref":
            name = f.get("name")
            if isinstance(name, str):
                names.append(name)
        nested = f.get("fields")
        if isinstance(nested, list):
            names.extend(_collect_user_ref_field_names(nested))
    return names


async def _enrich_records_with_user_names(
    records: list[dict[str, Any]],
    fields: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """For each user_ref field, attach a `{field_name}_display` value.

    The raw integer stays in the record (the form widget needs the id to
    PATCH back); the display name is added next to it so mobile can show
    "Alex" alongside "42" without a per-row auth call."""
    user_ref_names = _collect_user_ref_field_names(fields)
    if not user_ref_names:
        return records

    user_ids: set[int] = set()
    for r in records:
        data = r.get("data") if "data" in r and isinstance(r["data"], dict) else r
        for name in user_ref_names:
            value = data.get(name)
            if isinstance(value, int):
                user_ids.add(value)

    names = await _resolve_user_names(user_ids)
    if not names:
        return records

    for r in records:
        data = r.get("data") if "data" in r and isinstance(r["data"], dict) else r
        for name in user_ref_names:
            value = data.get(name)
            if isinstance(value, int) and value in names:
                data[f"{name}_display"] = names[value]
    return records


async def _enrich_single_record(
    record: dict[str, Any],
    fields: list[dict[str, Any]],
) -> dict[str, Any]:
    enriched = await _enrich_records_with_user_names([{"data": record}], fields)
    return enriched[0]["data"]


# ── Endpoints ───────────────────────────────────────────────────────────────


class _PatchBody(BaseModel):
    """Patch body for record updates. Plain dict overlay semantics."""
    patch: dict[str, Any] = Field(default_factory=dict)


@router.get("/command-data/nodes")
async def list_nodes(
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List nodes the requesting user can see, for the data-browser home picker."""
    # We don't have a per-user household roster here without an extra auth
    # call. Return every active node in any household the user can hit
    # `member` on. For now, list all active nodes and let the household
    # filter happen on each operation — mobile shows them all but each
    # subsequent call validates household access.
    nodes = db.query(Node).filter(Node.is_active.is_(True)).all()
    visible: list[dict[str, Any]] = []
    for node in nodes:
        if not node.household_id:
            continue
        try:
            verify_household_role(user.user_id, node.household_id, required_role="member")
        except HTTPException:
            continue
        visible.append(
            {
                "node_id": node.node_id,
                "household_id": node.household_id,
                "room": node.room,
            }
        )
    return {"nodes": visible}


@router.get("/command-data/nodes/{node_id}/commands")
async def list_commands(
    node_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List the visible commands on a node (those with non-disabled mode)."""
    _resolve_node_in_household(db, node_id, user)
    response = await _mqtt_request(node_id, "commands", {})
    return {"commands": response.get("commands", [])}


@router.get("/command-data/nodes/{node_id}/commands/{command_name}/schema")
async def get_schema(
    node_id: str,
    command_name: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return a command's FieldSpec list + mode. Cached for 10 minutes."""
    _resolve_node_in_household(db, node_id, user)

    cached = _schema_cache.get(node_id, command_name)
    if cached is not None:
        return cached

    response = await _mqtt_request(node_id, "schema", {"command_name": command_name})
    if not response.get("ok"):
        raise HTTPException(
            status_code=404,
            detail=response.get("error", {}).get("message", "command not found"),
        )
    out = {"mode": response["mode"], "fields": response["fields"]}
    _schema_cache.put(node_id, command_name, out)
    return out


@router.get("/command-data/nodes/{node_id}/commands/{command_name}/records")
async def list_records(
    node_id: str,
    command_name: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """List records for a command on a node.

    Records are filtered server-side (node-side) by `requesting_user_id`
    — the node returns only rows the caller owns (plus legacy rows with
    no `user_id`).

    Response shape:
        {
          "records": [{key, summary, data}],
          "truncated": bool,  # true if the node capped the response
          "count": int,
        }
    """
    _resolve_node_in_household(db, node_id, user)
    response = await _mqtt_request(node_id, "list", {
        "command_name": command_name,
        "requesting_user_id": user.user_id,
    })
    if not response.get("ok"):
        raise HTTPException(
            status_code=404,
            detail=response.get("error", {}).get("message", "command not found"),
        )

    # Enrich user_ref fields with display names. Pull schema (cache or
    # node) so we know which fields are user_refs.
    schema = _schema_cache.get(node_id, command_name)
    if schema is None:
        schema_resp = await _mqtt_request(node_id, "schema", {"command_name": command_name})
        if schema_resp.get("ok"):
            schema = {"mode": schema_resp["mode"], "fields": schema_resp["fields"]}
            _schema_cache.put(node_id, command_name, schema)

    records = response.get("records", [])
    if schema is not None:
        records = await _enrich_records_with_user_names(records, schema["fields"])

    return {
        "records": records,
        "truncated": bool(response.get("truncated")),
        "count": response.get("count", len(records)),
    }


@router.get(
    "/command-data/nodes/{node_id}/commands/{command_name}/records/{key}"
)
async def get_record(
    node_id: str,
    command_name: str,
    key: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Return one record's full data + the command's schema."""
    _resolve_node_in_household(db, node_id, user)
    response = await _mqtt_request(node_id, "get", {
        "command_name": command_name,
        "key": key,
        "requesting_user_id": user.user_id,
    })
    if not response.get("ok"):
        raise HTTPException(
            status_code=404,
            detail=response.get("error", {}).get("message", "record not found"),
        )

    schema = response.get("schema", {"mode": "enabled", "fields": []})
    _schema_cache.put(
        node_id,
        command_name,
        {"mode": schema.get("mode", "enabled"), "fields": schema.get("fields", [])},
    )
    record = response["record"]
    record = await _enrich_single_record(record, schema.get("fields", []))
    return {"record": record, "schema": schema}


@router.patch(
    "/command-data/nodes/{node_id}/commands/{command_name}/records/{key}"
)
async def update_record(
    node_id: str,
    command_name: str,
    key: str,
    body: _PatchBody = Body(...),
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Apply a patch to one record. Dict-overlay semantics."""
    _resolve_node_in_household(db, node_id, user)
    response = await _mqtt_request(node_id, "update", {
        "command_name": command_name,
        "key": key,
        "patch": body.patch,
        "requesting_user_id": user.user_id,
    })
    if not response.get("ok"):
        err = response.get("error", {})
        message = err.get("message", "update failed")
        if "not found" in message.lower() or "no editable" in message.lower():
            raise HTTPException(status_code=404, detail=message)
        if "read-only" in message.lower():
            raise HTTPException(status_code=403, detail=message)
        raise HTTPException(status_code=400, detail=message)

    return {"record": response.get("record")}


@router.delete(
    "/command-data/nodes/{node_id}/commands/{command_name}/records/{key}"
)
async def delete_record(
    node_id: str,
    command_name: str,
    key: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Delete one record."""
    _resolve_node_in_household(db, node_id, user)
    response = await _mqtt_request(node_id, "delete", {
        "command_name": command_name,
        "key": key,
        "requesting_user_id": user.user_id,
    })
    if not response.get("ok"):
        message = response.get("error", {}).get("message", "delete failed")
        if "read-only" in message.lower():
            raise HTTPException(status_code=403, detail=message)
        raise HTTPException(status_code=404, detail=message)
    return {"ok": True}
