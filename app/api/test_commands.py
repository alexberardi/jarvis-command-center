"""Test command endpoint for E2E command parsing tests.

Allows MCP tools and other services to test voice command parsing
without requiring a registered node. Uses app-to-app authentication.
"""

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.deps import require_app_auth, get_model_service
from app.core.model_service import ModelService

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/test", tags=["testing"])

MAX_VALIDATION_LOOPS = 3
MAX_CONCURRENT_TESTS = 3

_test_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    """Get or create the concurrency semaphore (lazy init for event loop safety)."""
    global _test_semaphore
    if _test_semaphore is None:
        _test_semaphore = asyncio.Semaphore(MAX_CONCURRENT_TESTS)
    return _test_semaphore


class CommandTestRequest(BaseModel):
    """Request model for test command endpoint."""

    voice_command: str = Field(..., min_length=1, max_length=2000)
    available_commands: list[dict[str, Any]] = Field(..., max_length=100)
    client_tools: list[dict[str, Any]] = Field(..., max_length=100)
    timezone: str = Field(default="America/New_York", max_length=100)
    node_context: dict[str, Any] | None = None
    # DEPRECATED / IGNORED: the warmup inference now ALWAYS runs and is never
    # skippable. Field kept only so existing callers that still send it don't
    # 422. CC does not act on this value.
    skip_warmup_inference: bool = True


class CommandTestResult(BaseModel):
    """Response model for test command endpoint."""

    voice_command: str
    stop_reason: str
    command_name: str | None = None
    parameters: dict[str, Any] | None = None
    assistant_message: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    validation_occurred: bool = False
    validation_count: int = 0
    elapsed_seconds: float = 0.0
    conversation_id: str | None = None
    error: str | None = None


async def _run_test_command(
    request: CommandTestRequest,
    model_service: ModelService,
) -> CommandTestResult:
    """
    Core logic for testing a voice command through the parsing pipeline.

    Separated from the endpoint for direct testing without FastAPI overhead.
    """
    start_time = time.monotonic()
    conversation_id = f"test-{uuid4().hex[:12]}"

    node_context = request.node_context or {
        "node_id": "test-node",
        "room": "test",
        "user": "test-user",
    }

    try:
        # Warmup inference ALWAYS runs (skip_warmup_inference is deprecated and
        # ignored — see CommandTestRequest).
        await model_service.warmup_conversation_with_tools(
            node_context=node_context,
            conversation_id=conversation_id,
            timezone=request.timezone,
            client_tools=request.client_tools,
            available_commands=request.available_commands,
        )

        result = await model_service.process_voice_command_with_tools(
            voice_command=request.voice_command,
            conversation_id=conversation_id,
        )

        # Handle validation loops
        validation_count = 0
        while (
            result.get("stop_reason") == "validation_required"
            and validation_count < MAX_VALIDATION_LOOPS
        ):
            validation_count += 1
            validation_req = result.get("validation_request", {})
            options = validation_req.get("options", [])
            selected = options[0] if options else "yes"

            result = await model_service.continue_conversation_with_tool_results(
                conversation_id=conversation_id,
                tool_results=[{"tool_call_id": "validation", "output": selected}],
            )

        # Extract command info from tool calls
        command_name = None
        parameters = None
        tool_calls_raw = result.get("tool_calls")

        if tool_calls_raw and len(tool_calls_raw) > 0:
            first_call = tool_calls_raw[0]
            func = first_call.get("function", {})
            command_name = func.get("name")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    parameters = json.loads(args)
                except json.JSONDecodeError:
                    parameters = {"_raw": args[:500]}
            else:
                parameters = args

        elapsed = time.monotonic() - start_time

        return CommandTestResult(
            voice_command=request.voice_command,
            stop_reason=result.get("stop_reason", "unknown"),
            command_name=command_name,
            parameters=parameters,
            assistant_message=result.get("assistant_message"),
            tool_calls=tool_calls_raw,
            validation_occurred=validation_count > 0,
            validation_count=validation_count,
            elapsed_seconds=round(elapsed, 3),
            conversation_id=conversation_id,
            error=result.get("error"),
        )

    except (RuntimeError, ValueError, KeyError) as e:
        elapsed = time.monotonic() - start_time
        error_msg = str(e)[:500]
        logger.error("Test command failed: %s", error_msg)
        return CommandTestResult(
            voice_command=request.voice_command,
            stop_reason="error",
            elapsed_seconds=round(elapsed, 3),
            conversation_id=conversation_id,
            error=error_msg,
        )
    finally:
        await model_service.cleanup_conversation(conversation_id)


@router.post("/command", response_model=CommandTestResult)
async def test_command(
    request: CommandTestRequest,
    _auth: None = Depends(require_app_auth),
    model_service: ModelService = Depends(get_model_service),
) -> CommandTestResult:
    """
    Test a voice command through the full parsing pipeline.

    Sends the command through warmup -> inference -> tool extraction
    without requiring a registered node. Handles validation loops
    by auto-selecting the first option.
    """
    async with _get_semaphore():
        return await _run_test_command(request, model_service)
