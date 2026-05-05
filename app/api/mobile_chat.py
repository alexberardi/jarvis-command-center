"""Mobile chat endpoint: SSE streaming for text-based Jarvis interaction.

Mobile sends text -> CC processes via LLM -> streams response back as SSE events.
Tool calls for client tools are routed to the selected node via MQTT.
"""

import asyncio
import json
import logging
import os
import tempfile as _tempfile
import time
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.conversation_cache import conversation_cache
from app.deps import get_db, get_model_service, verify_user_jwt, verify_household_role, AuthenticatedUser
from app.core.model_service import ModelService
from app.models import Node
from app.services.acknowledgment_service import generate_acknowledgment

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["mobile-chat"])

MAX_TOOL_ITERATIONS = 5
_RESULT_DIR = os.path.join(_tempfile.gettempdir(), "jarvis-device-control")


# ─── Request/Response Models ─────────────────────────────────────────────────


class MobileChatRequest(BaseModel):
    """Request body for mobile chat endpoint."""

    message: str = Field(..., min_length=1, max_length=5000)
    node_id: str
    household_id: str
    conversation_id: str | None = None
    timezone: str = "America/New_York"
    client_tools: list[dict[str, Any]] | None = None
    available_commands: list[dict[str, Any]] | None = None
    include_reasoning: bool = False


class WarmupRequest(BaseModel):
    """Request body for preemptive conversation warmup."""

    node_id: str
    household_id: str
    timezone: str = "America/New_York"
    client_tools: list[dict[str, Any]] | None = None
    available_commands: list[dict[str, Any]] | None = None


# ─── Shared Helpers ───────────────────────────────────────────────────────────


async def _log_mobile_transcript(
    conversation_id: str,
    user_message: str,
    assistant_message: str | None,
    tool_calls: list | None,
    user_id: int,
    household_id: str,
) -> None:
    """Fire-and-forget: log a mobile chat transcript for passive memory extraction."""
    try:
        from app.db import get_session_local
        from app.services.transcript_service import TranscriptService

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            TranscriptService(db).log_transcript(
                user_id=user_id,
                household_id=household_id,
                conversation_id=conversation_id,
                user_message=user_message,
                assistant_message=assistant_message,
                tool_calls=tool_calls,
            )
        finally:
            db.close()
    except Exception as e:
        logger.warning("Failed to log mobile transcript (non-fatal): %s", e)


def _sse_event(data: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _extract_intermediate_message(result: dict[str, Any]) -> str:
    """Extract the LLM's intermediate acknowledgment from a tool_calls response.

    The LLM is instructed to include a brief natural message alongside tool calls
    (e.g., "Let me check the weather for you."). Falls back to empty string if
    the LLM didn't include one.
    """
    return (result.get("assistant_message") or "").strip()


def _validate_node_in_household(
    db: Session, node_id: str, household_id: str,
) -> Node:
    """Validate that the node belongs to the household. Returns the node."""
    node = db.query(Node).filter(
        Node.node_id == node_id,
        Node.household_id == household_id,
    ).first()
    if not node:
        raise HTTPException(
            status_code=404,
            detail=f"Node {node_id} not found in household {household_id}",
        )
    return node


async def _resolve_tools(
    client_tools: list[dict[str, Any]] | None,
    available_commands: list[dict[str, Any]] | None,
    node_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve tools: MQTT fetch from node → request body fallback."""
    from app.api.node_tools import _request_tools_from_node

    # 1. Ask the node directly via MQTT (source of truth)
    try:
        fresh = await _request_tools_from_node(node_id, timeout=10.0)
        if fresh and fresh.get("client_tools"):
            return fresh["client_tools"], fresh["available_commands"]
    except Exception as e:
        logger.warning("MQTT tool fetch failed for %s: %s", node_id[:12], e)

    # 2. Fall back to what mobile sent (may be stale, but better than nothing)
    ct = client_tools or []
    ac = available_commands or []
    return ct, ac


async def _build_node_context(
    node: Node,
    user: AuthenticatedUser,
    household_id: str,
) -> dict[str, Any]:
    """Build the node_context dict with speaker name and room hierarchy."""
    node_context: dict[str, Any] = {
        "node_id": node.node_id,
        "room": node.room,
        "user": user.email or f"user-{user.user_id}",
        "household_id": household_id,
        "speaker_user_id": user.user_id,
        "voice_mode": node.voice_mode or "brief",
        "rich_response": True,
    }

    # Resolve speaker name for memory-aware prompts
    try:
        from app.core import service_config
        from app.core.utils.speaker_resolver import resolve_speaker_name
        auth_url = service_config.get_auth_url()
        speaker_name = await resolve_speaker_name(auth_url, user.user_id)
        if speaker_name:
            node_context["speaker_name"] = speaker_name
    except Exception as e:
        logger.warning("Failed to resolve speaker name for mobile chat: %s", e)

    # Load room hierarchy and device context for LLM
    try:
        from app.models import Device, Room as RoomModel
        from app.db import get_session_local
        session = get_session_local()()
        try:
            rooms = session.query(RoomModel).filter(
                RoomModel.household_id == household_id,
            ).all()
            has_hierarchy = any(r.parent_room_id for r in rooms)
            if has_hierarchy:
                node_context["room_hierarchy"] = [
                    {"id": r.id, "name": r.name, "parent_room_id": r.parent_room_id}
                    for r in rooms
                ]

            # Inject device data so the LLM knows available devices.
            # For voice flow, the node's DeviceDiscoveryAgent provides this
            # via agent context. For mobile chat, the CC loads it from DB.
            from sqlalchemy.orm import joinedload
            devices = session.query(Device).options(
                joinedload(Device.room),
            ).filter(
                Device.household_id == household_id,
                Device.is_active == True,  # noqa: E712
            ).all()
            if devices:
                device_controls: dict[str, list[dict]] = {}
                for d in devices:
                    domain = d.domain or "switch"
                    device_controls.setdefault(domain, []).append({
                        "entity_id": d.entity_id,
                        "name": d.name,
                        "area": d.room.name if d.room else "",
                        "state": "unknown",
                    })
                node_context.setdefault("agents", {})["home_assistant"] = {
                    "device_controls": device_controls,
                }
        finally:
            session.close()
    except Exception as e:
        logger.warning("Failed to load room/device context for mobile chat: %s", e, exc_info=True)

    return node_context


async def _do_warmup(
    model_service: ModelService,
    conversation_id: str,
    node: Node,
    user: AuthenticatedUser,
    household_id: str,
    timezone: str,
    client_tools: list[dict[str, Any]] | None,
    available_commands: list[dict[str, Any]] | None,
) -> None:
    """Perform conversation warmup: build context, resolve tools, call model_service."""
    node_context = await _build_node_context(node, user, household_id)
    ct, ac = await _resolve_tools(client_tools, available_commands, str(node.node_id))

    await model_service.warmup_conversation_with_tools(
        node_context=node_context,
        conversation_id=conversation_id,
        timezone=timezone,
        client_tools=ct,
        available_commands=ac,
        skip_warmup_inference=False,
    )


# ─── MQTT Tool Call Routing ───────────────────────────────────────────────────


async def _wait_for_result_file(request_id: str, timeout: float = 10.0) -> dict[str, Any] | None:
    """Poll for a result file written by the node callback. Non-blocking."""
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


async def _route_tool_call_to_node(
    node_id: str, tool_call: dict[str, Any],
    user_id: int | None = None,
) -> dict[str, Any]:
    """Route a single tool call to a node via MQTT and wait for the result."""
    from app.services.node_command_service import get_node_command_service

    func = tool_call.get("function", {})
    tool_call_id = tool_call.get("id", str(uuid4()))

    request_id = str(uuid4())
    os.makedirs(_RESULT_DIR, exist_ok=True)

    details: dict[str, Any] = {
        "command_name": func.get("name"),
        "arguments": func.get("arguments", {}),
        "tool_call_id": tool_call_id,
        "reply_request_id": request_id,
        "trusted": True,
    }
    if user_id is not None:
        details["user_id"] = user_id

    service = get_node_command_service()
    service.publish_command_with_id(node_id, "tool_call", details, request_id)

    result = await _wait_for_result_file(request_id)
    if result is None:
        return {
            "tool_call_id": tool_call_id,
            "output": {"error": "Node timed out executing command"},
        }

    return {
        "tool_call_id": tool_call_id,
        "output": result.get("output", result),
    }


# ─── SSE Chat Stream ─────────────────────────────────────────────────────────


async def _chat_stream(
    request: MobileChatRequest,
    model_service: ModelService,
    node: Node,
    user: AuthenticatedUser,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for the chat stream."""
    from app.core.utils.latency_logger import latency_logger

    conversation_id = request.conversation_id or f"mobile-{uuid4().hex[:12]}"
    timing = latency_logger.start_request(conversation_id, "mobile_chat")
    timing.source = "mobile"
    timing.node_id = node.node_id
    timing.household_id = request.household_id
    timing.user_command = request.message

    try:
        # Check if conversation is already warm (preemptive warmup or continuation)
        cached_tools = conversation_cache.get_tools(conversation_id)

        if cached_tools is None:
            # Cold start — need warmup
            yield _sse_event({"type": "status", "message": "Starting conversation..."})

            with timing.measure("warmup"):
                await _do_warmup(
                    model_service, conversation_id, node, user,
                    request.household_id, request.timezone,
                    request.client_tools, request.available_commands,
                )

        # Instant acknowledgment — keyword-matched, no LLM call.
        # Streamed before the pipeline starts so the user sees feedback immediately.
        ack_text = generate_acknowledgment(request.message)
        yield _sse_event({"type": "acknowledgment", "text": ack_text})

        with timing.measure("process_command"):
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.message,
                conversation_id=conversation_id,
            )

        # Track actions and preview extracted from tool results
        pending_actions: list[dict[str, Any]] = []
        action_context: dict[str, Any] = {}
        action_preview: str | None = None
        # Accumulate reasoning across iterations (initial call thinking
        # should be exposed even when the final response comes from a
        # formatting call that has no thinking of its own).
        accumulated_reasoning: str | None = None

        # Tool loop
        for _ in range(MAX_TOOL_ITERATIONS):
            stop_reason = result.get("stop_reason", "complete")
            if result.get("reasoning"):
                accumulated_reasoning = result["reasoning"]

            if stop_reason in ("complete", "server_tool_complete"):
                assistant_message = result.get("assistant_message", "")
                # Strip [Tool data: ...] prefix that's meant for history context, not display
                if assistant_message and "[Tool data:" in assistant_message:
                    idx = assistant_message.find("]\n\n")
                    if idx != -1:
                        assistant_message = assistant_message[idx + 3:].strip()
                if assistant_message:
                    words = assistant_message.split(" ")
                    for i, word in enumerate(words):
                        text = word if i == len(words) - 1 else word + " "
                        yield _sse_event({"type": "delta", "text": text})
                        if len(words) > 3:
                            await asyncio.sleep(0.02)

                done_event: dict[str, Any] = {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "full_text": assistant_message,
                    "stop_reason": "complete",
                    "trace_summary": timing.to_trace_summary(),
                }
                if pending_actions:
                    done_event["actions"] = pending_actions
                    done_event["action_context"] = action_context
                    if action_preview:
                        done_event["action_preview"] = action_preview
                if request.include_reasoning and accumulated_reasoning:
                    done_event["reasoning"] = accumulated_reasoning
                yield _sse_event(done_event)

                latency_logger.end_request(conversation_id)

                # Fire-and-forget: log transcript for passive memory extraction
                asyncio.create_task(_log_mobile_transcript(
                    conversation_id=conversation_id,
                    user_message=request.message,
                    assistant_message=assistant_message,
                    tool_calls=result.get("tool_calls"),
                    user_id=user.user_id,
                    household_id=request.household_id,
                ))
                return

            elif stop_reason == "validation_required":
                validation = result.get("validation_request", {})
                assistant_message = result.get("assistant_message", validation.get("question", ""))
                yield _sse_event({
                    "type": "done",
                    "conversation_id": conversation_id,
                    "full_text": assistant_message,
                    "stop_reason": "validation_required",
                    "validation": validation,
                    "trace_summary": timing.to_trace_summary(),
                })
                latency_logger.end_request(conversation_id)
                return

            elif stop_reason == "tool_calls":
                tool_calls = result.get("tool_calls", [])
                if not tool_calls:
                    break

                # LLM-generated intermediate message — stream immediately
                intermediate = _extract_intermediate_message(result)
                if intermediate:
                    yield _sse_event({"type": "delta", "text": intermediate + " "})

                yield _sse_event({
                    "type": "status",
                    "message": "Running command on node...",
                })

                tool_results: list[dict[str, Any]] = []
                for tc in tool_calls:
                    _tc_name = tc.get("function", {}).get("name", "?")
                    with timing.measure(f"mqtt_tool_{_tc_name}", service="node", metadata={"command": _tc_name}):
                        tc_result = await _route_tool_call_to_node(node.node_id, tc, user_id=user.user_id)
                    tool_results.append(tc_result)

                    # Extract actions from tool output
                    output = tc_result.get("output", {})
                    if isinstance(output, dict):
                        actions = output.get("actions")
                        if not actions:
                            ctx = output.get("context", {})
                            if isinstance(ctx, dict):
                                actions = ctx.get("actions")
                        if actions and isinstance(actions, list):
                            pending_actions = actions
                            func = tc.get("function", {})
                            raw_ctx = output.get("context", output)
                            action_context = {
                                "command_name": func.get("name", ""),
                                "context": raw_ctx,
                            }
                            if isinstance(raw_ctx, dict):
                                action_preview = (
                                    raw_ctx.get("preview")
                                    or raw_ctx.get("message")
                                    or None
                                )

                # Continue conversation with tool results.
                # NOTE: Previously used stream_final_response() for token-by-token
                # streaming, but that creates a separate formatting conversation
                # with a different system prompt, which evicts llama.cpp's KV
                # prefix cache and causes ~6s cache misses on follow-up messages.
                # Using continue_conversation keeps the same conversation context
                # so the prefix cache is preserved across turns.
                result = await model_service.continue_conversation_with_tool_results(
                    conversation_id=conversation_id,
                    tool_results=tool_results,
                )

            elif stop_reason == "error":
                error_msg = result.get("error", "An error occurred")
                timing.trace_status = "error"
                timing.error_message = error_msg
                yield _sse_event({
                    "type": "error",
                    "message": error_msg,
                    "conversation_id": conversation_id,
                    "trace_summary": timing.to_trace_summary(),
                })
                latency_logger.end_request(conversation_id)
                return

            else:
                assistant_message = result.get("assistant_message", "")
                done_event = {
                    "type": "done",
                    "conversation_id": conversation_id,
                    "full_text": assistant_message,
                    "stop_reason": stop_reason,
                    "trace_summary": timing.to_trace_summary(),
                }
                if request.include_reasoning and accumulated_reasoning:
                    done_event["reasoning"] = accumulated_reasoning
                yield _sse_event(done_event)
                latency_logger.end_request(conversation_id)
                return

        timing.trace_status = "error"
        timing.error_message = "Too many tool iterations"
        yield _sse_event({
            "type": "error",
            "message": "Too many tool iterations",
            "conversation_id": conversation_id,
            "trace_summary": timing.to_trace_summary(),
        })
        latency_logger.end_request(conversation_id)

    except Exception as e:
        logger.error("Mobile chat stream error: %s", e, exc_info=True)
        timing.trace_status = "error"
        timing.error_message = str(e)
        yield _sse_event({
            "type": "error",
            "message": str(e),
            "conversation_id": conversation_id,
            "trace_summary": timing.to_trace_summary(),
        })
        latency_logger.end_request(conversation_id)


# ─── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/chat/warmup")
async def mobile_chat_warmup(
    request: WarmupRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    model_service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
) -> dict[str, Any]:
    """Preemptively warm up a conversation for the mobile chat.

    Called when the app opens or a node is selected. By the time the user
    types their first message, the conversation is already warm with tools
    loaded, speaker identity resolved, and system prompt cached.

    Returns:
        {conversation_id: str, tools_loaded: int}
    """
    verify_household_role(user.user_id, request.household_id, required_role="member")
    node = _validate_node_in_household(db, request.node_id, request.household_id)

    conversation_id = f"mobile-{uuid4().hex[:12]}"

    await _do_warmup(
        model_service, conversation_id, node, user,
        request.household_id, request.timezone,
        request.client_tools, request.available_commands,
    )

    # Report how many *client* tools were loaded (exclude CC server tools)
    from app.core.conversation_handler import get_server_tool_names
    tools = conversation_cache.get_tools(conversation_id)
    server_tool_names = get_server_tool_names(tools)
    tools_loaded = (len(tools) - len(server_tool_names)) if tools else 0

    logger.info(
        "Mobile warmup complete: conversation=%s, tools=%d",
        conversation_id, tools_loaded,
    )

    return {
        "conversation_id": conversation_id,
        "tools_loaded": tools_loaded,
    }


@router.post("/chat")
async def mobile_chat(
    request: MobileChatRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    model_service: ModelService = Depends(get_model_service),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Chat with Jarvis via SSE stream.

    If conversation_id is provided from a prior warmup, skips the warmup step
    and goes straight to processing — much faster first response.

    SSE event types:
    - status: Processing update
    - delta: Incremental text chunk
    - done: Final response with conversation_id
    - error: Error occurred
    """
    verify_household_role(user.user_id, request.household_id, required_role="member")
    node = _validate_node_in_household(db, request.node_id, request.household_id)

    return StreamingResponse(
        _chat_stream(request, model_service, node, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
