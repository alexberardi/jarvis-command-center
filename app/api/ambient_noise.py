"""Ambient noise calibration endpoints.

The mobile "Set Automatically" button on the silence_threshold slider
triggers a brief ambient capture on the target node. Flow:

1. Mobile POSTs ``/nodes/{node_id}/ambient-noise-measurements`` (JWT auth)
2. CC publishes an ``measure_ambient_noise`` MQTT command to the node
3. Node records ~3s of audio, computes P95 RMS, posts the result back
4. Mobile polls ``GET /nodes/{node_id}/ambient-noise-measurements/{request_id}``
   until the result lands, then applies it to the slider

Results are kept in-process for 5 minutes — the mobile is expected to poll
within seconds. No DB row; this is calibration noise, not history.
"""
import logging
import threading
import time
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.context_providers.node_context_provider import NodeContextProvider
from app.deps import AuthenticatedUser, verify_api_key, verify_user_jwt
from app.services.node_command_service import get_node_command_service

logger = logging.getLogger("uvicorn")

router = APIRouter()

_RESULT_TTL_SECONDS = 300

# {request_id: {"node_id": str, "stored_at": float, "result": dict}}
_results: dict[str, dict[str, Any]] = {}
_results_lock = threading.Lock()


def _gc_expired() -> None:
    now = time.time()
    with _results_lock:
        expired = [rid for rid, entry in _results.items()
                   if now - entry["stored_at"] > _RESULT_TTL_SECONDS]
        for rid in expired:
            del _results[rid]


class TriggerRequest(BaseModel):
    duration_seconds: float | None = None


class TriggerResponse(BaseModel):
    request_id: str
    status: str


class ResultPayload(BaseModel):
    success: bool
    duration_seconds: float | None = None
    chunks: int | None = None
    p50_rms: float | None = None
    p75_rms: float | None = None
    p95_rms: float | None = None
    max_rms: float | None = None
    suggested_silence_threshold: int | None = None
    error: str | None = None


class ResultAck(BaseModel):
    status: str


class PollResponse(BaseModel):
    request_id: str
    status: str  # "pending" | "completed"
    completed_at: str | None = None
    result: ResultPayload | None = None


@router.post(
    "/nodes/{node_id}/ambient-noise-measurements",
    response_model=TriggerResponse,
)
def trigger_ambient_noise_measurement(
    node_id: str,
    body: TriggerRequest | None = None,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> TriggerResponse:
    """Ask a node to measure ambient noise and report back a suggested threshold.

    Fire-and-forget MQTT. Caller polls the GET endpoint for results.
    """
    duration = (body.duration_seconds if body and body.duration_seconds else 3.0)
    duration = max(1.0, min(10.0, float(duration)))

    service = get_node_command_service()
    request_id = service.publish_command(
        node_id,
        "measure_ambient_noise",
        {"duration_seconds": duration},
    )
    logger.info(
        "Ambient-noise measurement requested (node=%s request=%s duration=%.1fs)",
        node_id, request_id[:8], duration,
    )
    return TriggerResponse(request_id=request_id, status="sent")


@router.post(
    "/nodes/{node_id}/ambient-noise-measurements/{request_id}/result",
    response_model=ResultAck,
)
def post_ambient_noise_result(
    node_id: str,
    request_id: str,
    body: ResultPayload,
    node_context: NodeContextProvider = Depends(verify_api_key),
) -> ResultAck:
    """Receive the measurement result from the node. Node auth only."""
    if node_context.node.node_id != node_id:
        raise HTTPException(
            status_code=403,
            detail="Node may only post results for itself",
        )

    _gc_expired()
    with _results_lock:
        _results[request_id] = {
            "node_id": node_id,
            "stored_at": time.time(),
            "result": body.model_dump(),
            "completed_at": datetime.utcnow().isoformat() + "Z",
        }
    logger.info(
        "Ambient-noise result stored (node=%s request=%s success=%s suggested=%s)",
        node_id, request_id[:8], body.success, body.suggested_silence_threshold,
    )
    return ResultAck(status="ok")


@router.get(
    "/nodes/{node_id}/ambient-noise-measurements/{request_id}",
    response_model=PollResponse,
)
def get_ambient_noise_measurement(
    node_id: str,
    request_id: str,
    user: AuthenticatedUser = Depends(verify_user_jwt),
) -> PollResponse:
    """Poll for an ambient-noise measurement result."""
    _gc_expired()
    with _results_lock:
        entry = _results.get(request_id)

    if entry is None:
        return PollResponse(request_id=request_id, status="pending")

    if entry["node_id"] != node_id:
        raise HTTPException(status_code=404, detail="Not found")

    return PollResponse(
        request_id=request_id,
        status="completed",
        completed_at=entry["completed_at"],
        result=ResultPayload(**entry["result"]),
    )
