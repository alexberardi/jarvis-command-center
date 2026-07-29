"""Errand planner — turns a natural-language goal into a reviewable step plan.

POC (prds/errand-runner.md §2-§3). A single LLM call maps a goal + a curated
menu of node commands into routine-shaped steps that the plan card renders and
the executor (execute_routine_on_node) runs. The plan is a DRAFT until the user
confirms it on the card — "LLM proposes, card disposes." The planner never
executes anything and validates every proposed command against the menu, so a
hallucinated command is dropped rather than run.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("uvicorn")


# FALLBACK menu — used only when the target node's real command set can't be
# sourced (offline / MQTT timeout). The live path builds the menu from the node's
# actually-installed commands via ``build_errand_menu`` (so a node plans over
# whatever it has, not this fixed list).
COMMAND_MENU: list[dict[str, Any]] = [
    {"command": "get_weather", "description": "Weather for the user's location on a given day.",
     "args": {"resolved_datetimes": 'list of day keywords, e.g. ["today"]'}},
    {"command": "get_news", "description": "Top news headlines by category.",
     "args": {"category": "one of general/business/technology/sports/entertainment/health"}},
    {"command": "get_calendar_events", "description": "The user's calendar events for a day.",
     "args": {"resolved_datetimes": 'list of day keywords, e.g. ["today"]'}},
    {"command": "set_reminder", "description": "Create a reminder for the user.",
     "args": {"text": "what to remind about", "when": "natural-language time, e.g. 'tomorrow at 9am'"}},
    {"command": "get_device_status", "description": "Read the state of a smart-home device.",
     "args": {"device_name": "the device's name"}},
    {"command": "control_device", "description": "Act on a smart-home device.",
     "args": {"device_name": "the device's name", "action": "turn_on / turn_off / lock / unlock"}},
]

# Node commands that make no sense as a HEADLESS, background errand step —
# conversational, live-speaker-dependent, or recursive. Filtered out of the
# node-derived menu (they never appear as errand steps).
ERRAND_MENU_DENY: frozenset[str] = frozenset({
    "routine",          # recursion — an errand routine running a "routine" step
    "chat",             # open-ended conversation
    "answer_question",  # conversational Q&A, needs a live turn
    "tell_joke",        # conversational
    "act_on_items",     # depends on the live conversation's referenced_items
    "send_link",        # needs a live speaker/device target
    "control_node",     # live on-device hardware control (volume, etc.)
})


def build_errand_menu(available_commands: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Turn a node's ``available_commands`` (CommandDefinition dicts, as cached at
    conversation start or returned by ``node_tools._request_tools_from_node``)
    into the planner-menu shape ``[{command, description, args:{name: desc}}]``,
    dropping commands unfit for a headless background errand (``ERRAND_MENU_DENY``).

    Sourcing from ``available_commands`` (node-only) means server tools — including
    ``run_errand`` itself — are excluded by construction. Returns ``[]`` if there's
    nothing usable; the caller then falls back to ``COMMAND_MENU``.
    """
    menu: list[dict[str, Any]] = []
    for c in available_commands or []:
        name = (c.get("command_name") or "").strip()
        if not name or name in ERRAND_MENU_DENY:
            continue
        args = {
            p["name"]: (p.get("description") or p.get("type") or "")
            for p in (c.get("parameters") or [])
            if p.get("name")
        }
        menu.append({"command": name, "description": c.get("description") or "", "args": args})
    return menu


@dataclass
class PlanStep:
    command: str
    args: dict[str, Any]
    label: str

    def as_routine_step(self) -> dict[str, Any]:
        """The routine step shape the executor (RoutineCommand) consumes."""
        return {"command": self.command, "args": self.args, "label": self.label}


@dataclass
class ErrandPlan:
    summary: str
    steps: list[PlanStep] = field(default_factory=list)

    def routine_steps(self) -> list[dict[str, Any]]:
        return [s.as_routine_step() for s in self.steps]


def _build_prompt(goal: str, menu: list[dict[str, Any]]) -> str:
    menu_text = "\n".join(
        f"- {c['command']}: {c['description']} | args: {json.dumps(c['args'])}" for c in menu
    )
    return (
        "You are Jarvis's errand planner. Turn the user's goal into an ordered "
        "list of steps, choosing ONLY from the available commands below. Fill each "
        "step's args from that command's arg spec, using the user's own words. Keep "
        "the plan minimal — only the steps the goal actually needs — and give each "
        "step a short human-readable label.\n\n"
        f"Available commands:\n{menu_text}\n\n"
        f"User's goal: {goal}\n\n"
        'Return ONLY a JSON object: {"summary": "<one-line plain-English plan>", '
        '"steps": [{"command": "<name>", "args": {...}, "label": "<short label>"}]}. '
        "No prose, no markdown.\n\n"
        # Planning is structured extraction, not reasoning — disable Qwen3's
        # <think> pass. Without this the model's variable-length reasoning
        # (~200-1900 tokens) can exhaust max_tokens before the JSON, yielding an
        # empty/truncated response. /no_think is a soft switch (harmless text to
        # non-Qwen models). Measured: completion drops to ~79 tokens, reliably.
        "/no_think"
    )


def _extract_content(response: Any) -> str:
    if isinstance(response, dict):
        choices = response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")
            if content:
                return content
        return response.get("message", "") or ""
    return ""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = "\n".join(
            ln for ln in text.split("\n") if not ln.strip().startswith("```")
        ).strip()
    # Be forgiving of leading prose before the object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


async def plan_errand(
    goal: str, llm_client: Any, menu: list[dict[str, Any]] | None = None
) -> ErrandPlan:
    """Plan an errand from a goal.

    Raises ``ValueError`` if the LLM output is empty, unparseable, or yields no
    step drawn from the menu — the caller turns that into a "couldn't plan that"
    card rather than a broken plan.
    """
    menu = menu or COMMAND_MENU
    allowed = {c["command"] for c in menu}

    response = await llm_client.chat_completion(
        messages=[{"role": "user", "content": _build_prompt(goal, menu)}],
        temperature=0,
        # Headroom so a model that ignores /no_think and reasons anyway still has
        # room for its <think> pass AND the JSON (measured up to ~1900 tokens).
        max_tokens=2000,
    )
    raw = _extract_content(response)
    if not raw:
        raise ValueError("planner returned an empty response")
    try:
        data = json.loads(_strip_fences(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"planner returned invalid JSON: {e}") from e

    steps: list[PlanStep] = []
    for s in data.get("steps", []) or []:
        cmd = (s.get("command") or "").strip()
        if cmd not in allowed:
            logger.warning("Planner proposed unknown command %r — dropping it", cmd)
            continue
        steps.append(
            PlanStep(command=cmd, args=dict(s.get("args") or {}), label=(s.get("label") or cmd))
        )
    if not steps:
        raise ValueError("planner produced no usable steps")

    return ErrandPlan(summary=(data.get("summary") or goal).strip(), steps=steps)
