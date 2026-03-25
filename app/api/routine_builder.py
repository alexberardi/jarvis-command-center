"""Routine builder endpoint: conversational AI routine generation + testing.

Users describe what they want in natural language, the LLM generates routine
JSON using the node's installed commands, and the user can test/iterate before
saving.

Supports three providers:
- "jarvis" — local LLM via LLMProxyClient (no API key needed)
- "claude" — Anthropic API (BYOK)
- "openai" — OpenAI API (BYOK)
"""

import asyncio
import json
import logging
import os
import re
import tempfile as _tempfile
import time
from typing import Any, AsyncGenerator
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.conversation_cache import conversation_cache
from app.core.llm_proxy_client import LLMProxyClient
from app.deps import (
    AuthenticatedUser,
    get_db,
    verify_household_role,
    verify_user_jwt,
)
from app.models import Node

logger = logging.getLogger("uvicorn")

router = APIRouter(tags=["routine-builder"])

_RESULT_DIR = os.path.join(_tempfile.gettempdir(), "jarvis-device-control")

# ── Reference routines for the system prompt ─────────────────────────────────

REFERENCE_ROUTINES: list[dict[str, Any]] = [
    {
        "id": "good_morning",
        "name": "Good Morning",
        "trigger_phrases": ["good morning", "morning routine", "start my day"],
        "steps": [
            {"command": "control_device", "args": [{"key": "floor", "value": "Downstairs"}, {"key": "action", "value": "turn_on"}], "label": "lights"},
            {"command": "get_weather", "args": [{"key": "resolved_datetimes", "value": "today"}], "label": "weather"},
            {"command": "get_calendar_events", "args": [{"key": "resolved_datetimes", "value": "today"}], "label": "calendar"},
        ],
        "response_instruction": "Give a cheerful morning briefing with weather and calendar highlights.",
        "response_length": "short",
        "background": None,
    },
    {
        "id": "morning_briefing",
        "name": "Morning Briefing",
        "trigger_phrases": ["morning briefing", "daily briefing", "give me my briefing", "what's happening today", "catch me up"],
        "steps": [
            {"command": "get_weather", "args": [{"key": "resolved_datetimes", "value": "today"}], "label": "weather"},
            {"command": "get_calendar_events", "args": [{"key": "resolved_datetimes", "value": "today"}], "label": "calendar"},
            {"command": "get_news", "args": [{"key": "category", "value": "general"}, {"key": "count", "value": "3"}], "label": "news"},
        ],
        "response_instruction": "Deliver a morning briefing in a natural, flowing narrative style.",
        "response_length": "medium",
        "background": None,
    },
    {
        "id": "good_night",
        "name": "Good Night",
        "trigger_phrases": ["good night", "bedtime", "going to bed", "time for bed"],
        "steps": [
            {"command": "control_device", "args": [{"key": "floor", "value": "Downstairs"}, {"key": "action", "value": "turn_off"}], "label": "lights"},
            {"command": "get_calendar_events", "args": [{"key": "resolved_datetimes", "value": "tomorrow"}], "label": "tomorrow"},
        ],
        "response_instruction": "Brief goodnight with tomorrow's first appointment if any.",
        "response_length": "short",
        "background": None,
    },
]


# ── Request/Response Models ──────────────────────────────────────────────────


class RoutineChatRequest(BaseModel):
    """Request body for routine builder chat endpoint."""

    message: str = Field(..., min_length=1, max_length=5000)
    node_id: str
    household_id: str
    conversation_id: str | None = None
    provider: str = "jarvis"  # "jarvis", "claude", "openai"
    api_key: str | None = None  # Required for claude/openai
    timezone: str = "America/New_York"


class RoutineTestRequest(BaseModel):
    """Request body for routine test endpoint."""

    routine: dict[str, Any]
    node_id: str
    household_id: str


# ── Helpers ──────────────────────────────────────────────────────────────────


def _sse_event(data: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data)}\n\n"


def _validate_node_in_household(
    db: Session, node_id: str, household_id: str,
) -> Node:
    """Validate that the node belongs to the household."""
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


def _slugify(name: str) -> str:
    """Convert a display name to a slug ID."""
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def _format_commands(commands: list[dict[str, Any]]) -> str:
    """Format available commands into a readable section for the prompt."""
    if not commands:
        return "(No commands available)"

    parts: list[str] = []
    for cmd in commands:
        name: str = cmd.get("name", cmd.get("command_name", "unknown"))
        desc: str = cmd.get("description", "No description")
        keywords: list[str] = cmd.get("keywords", [])
        examples: list[str] = cmd.get("examples", [])
        params: list[dict[str, Any]] = cmd.get("parameters", [])

        lines: list[str] = [f"### {name}", f"Description: {desc}"]

        if keywords:
            lines.append(f"Keywords: {', '.join(keywords)}")

        if params:
            param_lines: list[str] = []
            for p in params:
                pname: str = p.get("name", "unknown")
                ptype: str = p.get("type", p.get("param_type", "string"))
                required: bool = p.get("required", False)
                pdesc: str = p.get("description", "")
                enum_vals: list[str] | None = p.get("enum_values")
                line_parts: list[str] = [f"  - `{pname}` ({ptype})"]
                if required:
                    line_parts.append("[required]")
                if pdesc:
                    line_parts.append(f"— {pdesc}")
                if enum_vals:
                    line_parts.append(f"(options: {', '.join(str(v) for v in enum_vals)})")
                param_lines.append(" ".join(line_parts))
            lines.append("Parameters:")
            lines.extend(param_lines)

        if examples:
            lines.append(f"Voice examples: {', '.join(repr(e) for e in examples[:3])}")

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _build_system_prompt(available_commands: list[dict[str, Any]]) -> str:
    """Build the routine builder system prompt."""
    commands_section = _format_commands(available_commands)
    reference_section = json.dumps(REFERENCE_ROUTINES, indent=2)

    return f"""\
You are a Jarvis routine builder — a conversational AI that helps users \
design multi-step voice routines by combining their installed commands.

## How to Interact

1. The user will describe what they want in natural language
2. If anything is unclear, ask a clarifying question (keep it brief)
3. When you have enough info, generate a routine as a JSON code block
4. If the user provides feedback or test results, refine the routine

## Routine Data Model

Each routine is a JSON object:

| Field | Type | Description |
|-------|------|-------------|
| id | string | Slugified name (lowercase, underscores) |
| name | string | Human-readable display name |
| trigger_phrases | string[] | 3+ natural voice phrases |
| steps | object[] | Ordered list of commands |
| response_instruction | string | How to format the response for TTS |
| response_length | string | "short", "medium", or "long" |
| background | null | Always null for on-demand routines |

### Step Object

| Field | Type | Description |
|-------|------|-------------|
| command | string | Must match an installed command |
| args | object[] | Array of {{"key": "...", "value": "..."}} objects |
| label | string | Short descriptive word |

## Installed Commands

{commands_section}

## Reference Routines

```json
{reference_section}
```

## Rules

1. `id` MUST be the slugified `name`
2. `trigger_phrases` MUST have at least 3 entries
3. Every step command MUST be from the installed commands above
4. `args` MUST be [{{"key": "...", "value": "..."}}] — NOT a dict
5. `response_length` MUST be "short", "medium", or "long"
6. `background` MUST be null
7. Only use commands from the installed list — never invent commands
8. Create routines that combine multiple commands — not single-command wrappers

## Output Format

When generating a routine, wrap the JSON in a code block:

```json
{{
  "id": "...",
  "name": "...",
  "trigger_phrases": ["...", "...", "..."],
  "steps": [...],
  "response_instruction": "...",
  "response_length": "short",
  "background": null
}}
```

Include a brief natural language explanation before or after the JSON block.

When the user reports test errors, analyze them and output a corrected routine."""


def _extract_routine_json(text: str) -> dict[str, Any] | None:
    """Extract routine JSON from LLM response text.

    Looks for JSON code blocks containing routine structure.
    """
    # Try ```json ... ``` blocks first
    json_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    for block in json_blocks:
        try:
            parsed = json.loads(block.strip())
            # Check if it looks like a routine (has steps and trigger_phrases)
            if isinstance(parsed, dict) and "steps" in parsed and "trigger_phrases" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    return None


def _validate_routine(
    routine: dict[str, Any], command_names: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a generated routine. Returns (routine_or_none, warnings)."""
    warnings: list[str] = []
    name: str = routine.get("name", "")

    # Check required fields
    trigger_phrases = routine.get("trigger_phrases")
    if not isinstance(trigger_phrases, list) or len(trigger_phrases) == 0:
        return None, [f"Routine '{name}': missing trigger_phrases"]

    steps = routine.get("steps")
    if not isinstance(steps, list) or len(steps) == 0:
        return None, [f"Routine '{name}': missing steps"]

    # Fix response_length
    if routine.get("response_length") not in ("short", "medium", "long"):
        warnings.append(f"Fixed invalid response_length, defaulting to 'medium'")
        routine["response_length"] = "medium"

    # Fix id
    expected_id = _slugify(name)
    if routine.get("id") != expected_id:
        routine["id"] = expected_id

    # Ensure background is null
    routine["background"] = None

    # Ensure response_instruction exists
    if not routine.get("response_instruction"):
        routine["response_instruction"] = ""

    # Check commands exist
    for step in steps:
        cmd_name: str = step.get("command", "")
        if cmd_name and cmd_name not in command_names:
            return None, [f"Unknown command '{cmd_name}' — not installed on this node"]

    return routine, warnings


# ── Provider Routing ─────────────────────────────────────────────────────────


async def _call_jarvis(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> AsyncGenerator[str, None]:
    """Stream from the local LLM proxy."""
    llm_client = LLMProxyClient()

    llm_messages = [{"role": "system", "content": system_prompt}] + messages

    async for event in llm_client.chat_completion_stream(
        messages=llm_messages,
        model="live",
        temperature=0.7,
    ):
        if event.get("delta"):
            yield event["delta"]
        elif event.get("done"):
            # Final event — content may have remaining text
            remaining = event.get("content", "")
            if remaining:
                # The stream already yielded deltas, content is the full text
                pass
            return


async def _call_claude_stream(
    system_prompt: str,
    messages: list[dict[str, str]],
    api_key: str,
) -> AsyncGenerator[str, None]:
    """Stream from Anthropic API."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 4096,
                "system": system_prompt,
                "messages": messages,
                "stream": True,
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"Claude API error: {resp.status_code} — {body.decode()}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    event = json.loads(data_str)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield delta.get("text", "")
                except json.JSONDecodeError:
                    continue


async def _call_openai_stream(
    system_prompt: str,
    messages: list[dict[str, str]],
    api_key: str,
) -> AsyncGenerator[str, None]:
    """Stream from OpenAI API."""
    openai_messages = [{"role": "system", "content": system_prompt}] + messages

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "max_tokens": 4096,
                "messages": openai_messages,
                "stream": True,
            },
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(f"OpenAI API error: {resp.status_code} — {body.decode()}")

            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    return
                try:
                    event = json.loads(data_str)
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError):
                    continue


# ── Chat Stream ──────────────────────────────────────────────────────────────


async def _routine_chat_stream(
    request: RoutineChatRequest,
    node: Node,
    user: AuthenticatedUser,
) -> AsyncGenerator[str, None]:
    """Async generator that yields SSE events for the routine builder chat."""
    conversation_id = request.conversation_id or f"routine-{uuid4().hex[:12]}"

    try:
        # Resolve the node's available commands
        yield _sse_event({"type": "status", "message": "Loading commands..."})

        from app.api.mobile_chat import _resolve_tools
        _, available_commands = await _resolve_tools(None, None, str(node.node_id))

        if not available_commands:
            yield _sse_event({
                "type": "error",
                "message": "No commands found on this node. Make sure it is online.",
                "conversation_id": conversation_id,
            })
            return

        command_names: set[str] = set()
        for cmd in available_commands:
            name = cmd.get("name", cmd.get("command_name", ""))
            if name:
                command_names.add(name)

        # Build system prompt
        system_prompt = _build_system_prompt(available_commands)

        # Load or init conversation history
        cached_messages = conversation_cache.get_messages(conversation_id)
        if cached_messages:
            # Continue existing conversation — extract user/assistant messages
            history = [
                m for m in cached_messages
                if m.get("role") in ("user", "assistant")
            ]
        else:
            history = []

        # Add the new user message
        history.append({"role": "user", "content": request.message})

        # Route to the selected provider
        yield _sse_event({"type": "status", "message": "Generating..."})

        full_text = ""
        if request.provider == "jarvis":
            async for token in _call_jarvis(system_prompt, history):
                full_text += token
                yield _sse_event({"type": "delta", "text": token})
        elif request.provider == "claude":
            if not request.api_key:
                yield _sse_event({
                    "type": "error",
                    "message": "API key is required for Claude",
                    "conversation_id": conversation_id,
                })
                return
            async for token in _call_claude_stream(system_prompt, history, request.api_key):
                full_text += token
                yield _sse_event({"type": "delta", "text": token})
        elif request.provider == "openai":
            if not request.api_key:
                yield _sse_event({
                    "type": "error",
                    "message": "API key is required for OpenAI",
                    "conversation_id": conversation_id,
                })
                return
            async for token in _call_openai_stream(system_prompt, history, request.api_key):
                full_text += token
                yield _sse_event({"type": "delta", "text": token})
        else:
            yield _sse_event({
                "type": "error",
                "message": f"Unknown provider: {request.provider}",
                "conversation_id": conversation_id,
            })
            return

        # Save conversation history
        history.append({"role": "assistant", "content": full_text})
        conversation_cache.set(
            conversation_id=conversation_id,
            messages=history,
            available_commands=available_commands,
            timezone=request.timezone,
            tools=[],
            node_context={"node_id": str(node.node_id)},
        )

        # Try to extract a routine from the response
        routine_json = _extract_routine_json(full_text)
        validated_routine: dict[str, Any] | None = None
        validation_warnings: list[str] = []

        if routine_json:
            validated_routine, validation_warnings = _validate_routine(
                routine_json, command_names,
            )

        done_event: dict[str, Any] = {
            "type": "done",
            "conversation_id": conversation_id,
            "full_text": full_text,
        }
        if validated_routine:
            done_event["routine"] = validated_routine
        if validation_warnings:
            done_event["validation_warnings"] = validation_warnings

        yield _sse_event(done_event)

    except Exception as e:
        logger.error("Routine builder stream error: %s", e, exc_info=True)
        yield _sse_event({
            "type": "error",
            "message": str(e),
            "conversation_id": conversation_id,
        })


# ── Test Stream ──────────────────────────────────────────────────────────────


async def _wait_for_result_file(
    request_id: str, timeout: float = 15.0,
) -> dict[str, Any] | None:
    """Poll for a result file written by the node callback."""
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


async def _routine_test_stream(
    request: RoutineTestRequest,
    node: Node,
) -> AsyncGenerator[str, None]:
    """Execute each step and stream results back."""
    from app.services.node_command_service import get_node_command_service

    routine = request.routine
    steps = routine.get("steps", [])
    routine_name = routine.get("name", "Untitled")

    if not steps:
        yield _sse_event({
            "type": "error",
            "message": "Routine has no steps to test",
        })
        return

    yield _sse_event({
        "type": "status",
        "message": f"Testing '{routine_name}' ({len(steps)} steps)...",
    })

    results: list[dict[str, Any]] = []
    os.makedirs(_RESULT_DIR, exist_ok=True)

    for i, step in enumerate(steps):
        command = step.get("command", "unknown")
        label = step.get("label", command)
        args_list = step.get("args", [])

        # Convert [{key, value}] to {key: value} dict for the tool call
        arguments: dict[str, str] = {}
        for arg in args_list:
            if isinstance(arg, dict) and "key" in arg:
                arguments[arg["key"]] = arg.get("value", "")

        yield _sse_event({
            "type": "step_start",
            "step_index": i,
            "command": command,
            "label": label,
        })

        request_id = str(uuid4())
        tool_call_id = str(uuid4())

        details = {
            "command_name": command,
            "arguments": arguments,
            "tool_call_id": tool_call_id,
            "reply_request_id": request_id,
            "trusted": True,
        }

        service = get_node_command_service()
        service.publish_command_with_id(str(node.node_id), "tool_call", details, request_id)

        result = await _wait_for_result_file(request_id)

        step_result: dict[str, Any] = {
            "step_index": i,
            "command": command,
            "label": label,
            "arguments": arguments,
        }

        if result is None:
            step_result["success"] = False
            step_result["error"] = "Node timed out executing command"
            step_result["output"] = None
        else:
            output = result.get("output", result)
            # Check for error in output
            if isinstance(output, dict) and output.get("error"):
                step_result["success"] = False
                step_result["error"] = output["error"]
            else:
                step_result["success"] = True
                step_result["error"] = None
            step_result["output"] = output

        results.append(step_result)

        yield _sse_event({
            "type": "step_result",
            **step_result,
        })

    # Summary
    passed = sum(1 for r in results if r["success"])
    yield _sse_event({
        "type": "test_done",
        "results": results,
        "passed": passed,
        "total": len(results),
    })


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.post("/chat")
async def routine_builder_chat(
    request: RoutineChatRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Chat with the routine builder via SSE stream.

    Conversational routine generation. The LLM uses the node's installed
    commands to generate routine JSON. Supports iterative refinement —
    send test results or feedback to get an updated routine.

    Providers:
    - "jarvis": Local LLM (no API key needed)
    - "claude": Anthropic API (requires api_key)
    - "openai": OpenAI API (requires api_key)

    SSE event types:
    - status: Processing update
    - delta: Incremental text chunk
    - done: Final response (may include `routine` and `validation_warnings`)
    - error: Error occurred
    """
    verify_household_role(user.user_id, request.household_id, required_role="member")
    node = _validate_node_in_household(db, request.node_id, request.household_id)

    return StreamingResponse(
        _routine_chat_stream(request, node, user),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/test")
async def routine_test(
    request: RoutineTestRequest,
    user: AuthenticatedUser = Depends(verify_user_jwt),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Test a routine by executing each step on the node.

    Streams step-by-step results so the user can see what works and what
    fails before saving. The user can mark false positives and send errors
    back to the builder chat for correction.

    SSE event types:
    - status: Starting test
    - step_start: About to execute a step
    - step_result: Result of a step (success/error + output)
    - test_done: All steps complete (summary with pass/fail counts)
    """
    verify_household_role(user.user_id, request.household_id, required_role="member")
    node = _validate_node_in_household(db, request.node_id, request.household_id)

    return StreamingResponse(
        _routine_test_stream(request, node),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
