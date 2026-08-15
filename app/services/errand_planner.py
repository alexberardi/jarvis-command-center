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

# Commands (node OR server) that make no sense as a HEADLESS, background errand
# step — conversational, live-speaker-dependent, or recursive. Filtered out of
# BOTH the node-derived and server-tool menus (they never appear as errand steps).
ERRAND_MENU_DENY: frozenset[str] = frozenset({
    "routine",          # recursion — an errand routine running a "routine" step
    "chat",             # open-ended conversation
    "answer_question",  # conversational Q&A, needs a live turn (also hard-disabled server-side)
    "tell_joke",        # conversational
    "act_on_items",     # depends on the live conversation's referenced_items
    "send_link",        # needs a live speaker/device target
    "control_node",     # live on-device hardware control (volume, etc.)
    # Conversation-runtime-only server tools — meaningless in a detached errand
    # (no live turn to ask / no audio / recursion). run_errand is the load-bearing
    # one: without it here the planner would plan an errand that runs an errand.
    "run_errand", "request_validation", "identify_speaker",
    "resolve_relative_date", "get_command_examples",
    # NB: outbound-to-a-third-party commands (email/send_message/make_phone_call)
    # are NO LONGER denied here — they now really execute (make_phone_call routes
    # to phone_call_service.create_call_plan with confirm-before-dial; other
    # channels are reviewed on the plan card before Run). Provenance/confirmation
    # rides on the real handler + the plan-card gate, not a menu denylist.
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
        menu.append({
            "command": name,
            "description": c.get("description") or "",
            "args": args,
            "is_risky": bool(c.get("is_risky", False)),
        })
    return menu


def _errand_bool_setting(key: str, household_id: str | None, default: bool) -> bool:
    """Read a per-household boolean setting the same way the voice path does.

    Value may come back as a real bool or a string ("true"/"1"/"yes"); on any
    error we return the caller's documented fail direction (open or closed) —
    mirroring conversation_handler's gating helpers so the errand menu offers
    EXACTLY the server tools the voice path would.
    """
    try:
        from app.services.settings_service import get_settings_service

        value = get_settings_service().get(key, household_id=str(household_id))
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes", "on")
        return default
    except Exception:  # noqa: BLE001 — settings unavailable → documented default
        return default


def enabled_errand_server_tools(
    household_id: str | None, speaker_user_id: int | None
) -> list[str]:
    """The server-tool names offered to the errand planner for this household.

    Mirrors the voice text-path allow-list (conversation_handler ``_safe_tool_names``)
    with its per-household gates and fail directions, with two deliberate errand
    differences: (1) ``make_phone_call`` is included only when ``phone_calls`` is
    actually enabled (the voice path always offers it and lets execute() refuse;
    an errand menu proposes actions, so a disabled capability shouldn't appear),
    and (2) ``run_errand`` is excluded (recursion — enforced again via
    ``ERRAND_MENU_DENY`` downstream). Names are candidates; ``build_server_tool_menu``
    drops any that aren't actually registered.
    """
    has_speaker = speaker_user_id is not None
    names: list[str] = []

    # Phone calls — the headline capability. Gated by the SAME predicate the tool
    # enforces at execute time (fails closed), so the menu never proposes a call
    # the household can't place.
    try:
        from app.services.phone_call_service import phone_calls_enabled

        if phone_calls_enabled(household_id):
            names.append("make_phone_call")
    except Exception:  # noqa: BLE001 — treat an unavailable gate as disabled
        pass

    # Web research — fails CLOSED (default off), like the voice path.
    if _errand_bool_setting("web_search.enabled", household_id, False):
        names.extend(["deep_research", "quick_search"])

    # Memory — fails OPEN (default on) but needs an identified speaker to scope to.
    if has_speaker and _errand_bool_setting("memory.enabled", household_id, True):
        names.extend(["remember", "forget"])
        if _errand_bool_setting("memory.recall_enabled", household_id, True):
            names.append("recall")

    return [n for n in names if n not in ERRAND_MENU_DENY]


def build_server_tool_menu(
    household_id: str | None, speaker_user_id: int | None
) -> list[dict[str, Any]]:
    """Enabled server tools as planner-menu entries ``[{command, description, args}]``.

    Source of truth is the ``tool_registry`` singleton (only globally-enabled
    ``IServerTool``s are present); we further gate by household via
    ``enabled_errand_server_tools`` and normalize each tool's JSON-Schema
    ``parameters.properties`` into the same ``{name: description}`` shape the node
    menu uses. A name that isn't registered (e.g. a tool file was removed) is
    silently skipped — same as the voice allow-list loop.
    """
    from app.core.tool_registry import tool_registry

    menu: list[dict[str, Any]] = []
    for name in enabled_errand_server_tools(household_id, speaker_user_id):
        tool = tool_registry.get_tool(name)
        if tool is None:
            continue
        fn = tool.to_openai_format().get("function", {})
        props = (fn.get("parameters") or {}).get("properties") or {}
        args = {
            key: (spec.get("description") or spec.get("type") or "")
            for key, spec in props.items()
            if key
        }
        menu.append({
            "command": fn.get("name") or name,
            "description": fn.get("description") or "",
            "args": args,
            "is_risky": bool(getattr(tool, "is_risky", False)),
        })
    return menu


def build_full_errand_menu(
    available_commands: list[dict[str, Any]] | None,
    household_id: str | None,
    speaker_user_id: int | None,
) -> list[dict[str, Any]]:
    """The planner's full building-block menu: the node's installed commands
    UNION the enabled server tools, normalized to one ``{command, description,
    args}`` shape. This is the general model — the planner sees every capability
    the errand can actually use (node command OR server tool), and the executor
    routes each chosen step to the right plane at run time via ``has_tool``.

    De-dup is server-precedence: if a node command and a server tool share a name,
    the executor's ``tool_registry.has_tool`` check would run the server tool, so
    the menu advertises the server entry to keep the plan honest.
    """
    server_menu = build_server_tool_menu(household_id, speaker_user_id)
    server_names = {c["command"] for c in server_menu}
    node_menu = [
        c for c in build_errand_menu(available_commands) if c["command"] not in server_names
    ]
    return node_menu + server_menu


# Control-flow commands the planner may emit that are NOT capabilities in the menu.
# They're always allowed through validation (the engine has a handler for each);
# they never contact anyone, so they're never risky.
#   request_replan — a mid-run checkpoint: pause here and plan the rest once the
#                    earlier steps' results are known (pause-and-replan, PRD §2.6).
CONTROL_COMMANDS: frozenset[str] = frozenset({"request_replan"})


@dataclass
class PlanStep:
    command: str
    args: dict[str, Any]
    label: str
    is_risky: bool = False

    def as_routine_step(self) -> dict[str, Any]:
        """The routine step shape the executor consumes. ``is_risky`` rides along so
        the mid-run envelope check (``widens_envelope``) can tell a benign added step
        from one that spends money / contacts a stranger — the stored plan is the only
        place that signal survives to run time."""
        return {"command": self.command, "args": self.args, "label": self.label,
                "is_risky": self.is_risky}


@dataclass
class ErrandPlan:
    summary: str
    steps: list[PlanStep] = field(default_factory=list)

    def routine_steps(self) -> list[dict[str, Any]]:
        return [s.as_routine_step() for s in self.steps]


# Provenance guardrail: don't fabricate SPECIFIC data. Crucially this is NOT a
# reason to drop a call/step the user asked for — naming who to call ("the
# pharmacy", a business) is the user's own intent, and every call is reviewed and
# confirmed on the plan card before it's placed. Only invented specifics (a
# made-up email address, phone number, or street address) are off-limits.
_FABRICATION_GUARDRAIL = (
    "Do NOT invent specific data you weren't given — a made-up email address, "
    "phone number, or mailing address. You MAY use the user's own words to name "
    "who to contact (e.g. call the business, or 'the pharmacy' they mentioned); "
    "the user reviews and confirms every call before it is placed, so keep the "
    "step. Only leave a step out if performing it would require inventing a "
    "specific contact detail the user never provided.\n\n"
)

# Shared decomposition rule — the planner must turn EVERY action the user names
# into its own step, in order. Without this the small model collapses multi-part
# instructions ("call the pharmacy first, then call the office") into a single
# step and silently drops the rest.
_DECOMPOSITION_RULE = (
    "Create a SEPARATE step for EACH distinct action the user asks for, and keep "
    "the order they gave. If they say 'first do X, then Y' — or list several "
    "things — that is MULTIPLE steps, one per action. Do not merge two actions "
    "into one step and do not drop an action. Only include actions the user "
    "actually asked for.\n\n"
)

# Closing directive. Decomposing a multi-part errand ("call the pharmacy, THEN
# the office") genuinely needs reasoning, so we let the model think rather than
# forcing /no_think (which collapses such instructions into one step). _run_planner
# strips the <think> block before parsing and max_tokens leaves room for both the
# reasoning and the JSON. The model still ends with the JSON object only.
_CLOSING_DIRECTIVE = (
    "Think step by step: list each action the user wants and its order, then output "
    "ONLY the final JSON object and nothing after it."
)


# Mid-run checkpoint rule — lets the planner defer the part of a plan that can't be
# decided until earlier steps run, instead of guessing. Deliberately CONSERVATIVE:
# the small model would over-insert checkpoints given half a reason, and every extra
# checkpoint is a wasted re-plan. Only outcome-dependent, can't-know-yet branches.
_REPLAN_CHECKPOINT_RULE = (
    "CHECKPOINT (rare): if — and ONLY if — a later action genuinely depends on the "
    "RESULT of an earlier step that cannot be known yet (e.g. 'check the weather and "
    "IF it's going to rain remind me to bring an umbrella', or 'call the pharmacy and "
    "then do the right follow-up based on what they say'), end the plan at that point "
    "with a single checkpoint step "
    '{"command": "request_replan", "args": {"reason": "<what to decide once the '
    'earlier steps have run>"}, "label": "<short label>"} and DO NOT try to guess the '
    "steps that come after it — they will be planned once the earlier results are "
    "known. Most goals do NOT need a checkpoint; only use one when the next actions "
    "truly cannot be chosen until the earlier steps have run.\n\n"
)


def _risky_by_cmd(menu: list[dict[str, Any]]) -> dict[str, bool]:
    """command → is_risky, from a planner menu (for stamping steps)."""
    return {c["command"]: bool(c.get("is_risky", False)) for c in menu}


def _build_prompt(goal: str, menu: list[dict[str, Any]]) -> str:
    menu_text = "\n".join(
        f"- {c['command']}: {c['description']} | args: {json.dumps(c['args'])}" for c in menu
    )
    return (
        "You are Jarvis's errand planner. Turn the user's goal into an ordered "
        "list of steps, choosing ONLY from the available commands below. Fill each "
        "step's args from that command's arg spec, using the user's own words, and "
        "give each step a short human-readable label.\n\n"
        + _DECOMPOSITION_RULE
        + _REPLAN_CHECKPOINT_RULE +
        f"Available commands:\n{menu_text}\n\n"
        f"User's goal: {goal}\n\n"
        + _FABRICATION_GUARDRAIL +
        'Return a JSON object: {"summary": "<one-line plain-English plan>", '
        '"steps": [{"command": "<name>", "args": {...}, "label": "<short label>"}]}. '
        "No markdown.\n\n"
        + _CLOSING_DIRECTIVE
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


def _build_refine_prompt(
    summary: str, current_steps: list[dict[str, Any]], instruction: str,
    menu: list[dict[str, Any]],
) -> str:
    """Prompt to REVISE an existing plan given a natural-language instruction —
    the conversational "tell it what to change" path (not a fresh re-plan)."""
    menu_text = "\n".join(
        f"- {c['command']}: {c['description']} | args: {json.dumps(c['args'])}" for c in menu
    )
    return (
        "You are Jarvis's errand planner revising an EXISTING plan. Apply the "
        "user's change to the current plan, keeping the parts they didn't ask to "
        "change and choosing ONLY from the available commands.\n\n"
        + _DECOMPOSITION_RULE +
        f"Current plan: {summary}\n"
        f"Current steps: {json.dumps(current_steps)}\n\n"
        f"Available commands:\n{menu_text}\n\n"
        f"The user wants this change: {instruction}\n\n"
        + _FABRICATION_GUARDRAIL +
        'Return the revised JSON object: {"summary": "<one-line plan>", '
        '"steps": [{"command": "<name>", "args": {...}, "label": "<short label>"}]}. '
        "No markdown.\n\n"
        + _CLOSING_DIRECTIVE
    )


def _strip_think(text: str) -> str:
    """Drop a Qwen3 <think>…</think> reasoning block (thinking is ON so the planner
    can decompose multi-step instructions). Keep only what follows the LAST
    </think>; an unclosed block (finish_reason=length) leaves the JSON absent, which
    _run_planner then reports as an empty/invalid response."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1]
    return text


async def _run_planner(
    prompt: str, llm_client: Any, *, extra_body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Single planner LLM call → parsed JSON dict. Raises ValueError on an empty
    or unparseable response. Thinking is ON by default (the model decomposes
    multi-part instructions), so we strip the <think> block and give max_tokens
    headroom for both the reasoning and the JSON. Callers whose task does NOT want
    thinking (e.g. the situation matcher, where Qwen3.5 reasons without bound) pass
    ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}``."""
    response = await llm_client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        # Room for the <think> reasoning pass (measured up to ~1900 tokens) AND the
        # JSON that follows. Too tight and the JSON is truncated (finish=length).
        max_tokens=3500,
        extra_body=extra_body,
    )
    raw = _extract_content(response)
    if not raw:
        # DIAGNOSTIC (Qwen3.5): the answer can land in reasoning_content, or content
        # can be empty when thinking runs past max_tokens — both surface here as an
        # empty planner response (seen live: a mid-run replan returned nothing → no
        # continuation steps got added). Log enough to tell which next time.
        try:
            ch = (response.get("choices") or [{}])[0] if isinstance(response, dict) else {}
            msg = ch.get("message", {}) if isinstance(ch, dict) else {}
            logger.warning(
                "planner EMPTY response: finish=%s content=%r reasoning_len=%d",
                ch.get("finish_reason"),
                (msg.get("content") if isinstance(msg, dict) else None),
                len((msg.get("reasoning_content") if isinstance(msg, dict) else "") or ""),
            )
        except Exception:  # noqa: BLE001 — logging must not mask the real error
            logger.warning("planner EMPTY response (uninspectable): %r", str(response)[:300])
        raise ValueError("planner returned an empty response")
    try:
        return json.loads(_strip_fences(_strip_think(raw)))
    except json.JSONDecodeError as e:
        logger.warning("planner UNPARSEABLE JSON (first 400 chars): %r", raw[:400])
        raise ValueError(f"planner returned invalid JSON: {e}") from e


def _plan_from_data(
    data: dict[str, Any], allowed: set[str], fallback_summary: str,
    risky_by_cmd: dict[str, bool] | None = None, *, require_nonempty: bool = True,
) -> ErrandPlan:
    """Validate the LLM's steps against the menu, dropping unknown commands and
    stamping each step's ``is_risky`` from the menu (control commands are never
    risky). ``allowed`` should already include ``CONTROL_COMMANDS``. Raises ValueError
    if ``require_nonempty`` and nothing usable survives; a cursor-aware replan passes
    ``require_nonempty=False`` because "nothing more to do" is a valid continuation."""
    risky_by_cmd = risky_by_cmd or {}
    steps: list[PlanStep] = []
    for s in data.get("steps", []) or []:
        cmd = (s.get("command") or "").strip()
        if cmd not in allowed:
            logger.warning("Planner proposed unknown command %r — dropping it", cmd)
            continue
        steps.append(PlanStep(
            command=cmd, args=dict(s.get("args") or {}), label=(s.get("label") or cmd),
            is_risky=bool(risky_by_cmd.get(cmd, False)),
        ))
    if not steps and require_nonempty:
        raise ValueError("planner produced no usable steps")
    return ErrandPlan(summary=(data.get("summary") or fallback_summary).strip(), steps=steps)


async def plan_errand(
    goal: str, llm_client: Any, menu: list[dict[str, Any]] | None = None
) -> ErrandPlan:
    """Plan an errand from a goal.

    Raises ``ValueError`` if the LLM output is empty, unparseable, or yields no
    step drawn from the menu — the caller turns that into a "couldn't plan that"
    card rather than a broken plan.
    """
    menu = menu or COMMAND_MENU
    data = await _run_planner(_build_prompt(goal, menu), llm_client)
    allowed = {c["command"] for c in menu} | CONTROL_COMMANDS
    return _plan_from_data(data, allowed, goal, _risky_by_cmd(menu))


async def refine_errand_plan(
    summary: str, current_steps: list[dict[str, Any]], instruction: str,
    llm_client: Any, menu: list[dict[str, Any]] | None = None,
) -> ErrandPlan:
    """Revise an existing plan from a natural-language instruction (the "tell it
    what to change" path). Same failure contract as ``plan_errand``."""
    menu = menu or COMMAND_MENU
    data = await _run_planner(
        _build_refine_prompt(summary, current_steps, instruction, menu), llm_client
    )
    # Never let a degraded menu (a transient node-fetch failure falls back to
    # COMMAND_MENU) strip steps the plan already had — union the menu's commands
    # with the current plan's, so a refine always preserves what it didn't change.
    allowed = {c["command"] for c in menu} | CONTROL_COMMANDS
    allowed |= {s["command"] for s in current_steps if s.get("command")}
    return _plan_from_data(data, allowed, summary, _risky_by_cmd(menu))


def _compact_result_data(data: Any, cap: int = 320) -> str:
    """A short string from a step's structured ``data`` payload — node commands return
    their real result THERE (get_weather's forecast, etc.), not in ``message``, so the
    cursor-aware replan MUST see it to make an outcome-dependent decision. Skips
    bookkeeping keys; caps length."""
    if not isinstance(data, dict):
        return ""
    skip = {"success", "error", "actions", "status", "message"}
    parts: list[str] = []
    for key, value in data.items():
        if key in skip:
            continue
        parts.append(f"{key}={value if isinstance(value, (str, int, float, bool)) else json.dumps(value)}")
        if sum(len(p) for p in parts) > cap:
            break
    return "; ".join(parts)[:cap]


def _summarize_progress(done_steps: list[dict[str, Any]], results: list[dict[str, Any]]) -> str:
    """A short "what happened so far" block for the cursor-aware replan prompt, pairing
    each already-run step with its outcome. Uses message / call-wrapup summary / the
    structured ``data`` payload (a node command's real result lives there) / error — so
    the replan actually sees e.g. the forecast and can branch on it."""
    lines: list[str] = []
    for i, step in enumerate(done_steps):
        label = step.get("label") or step.get("command") or f"step {i + 1}"
        r = results[i] if i < len(results) else {}
        detail = (r.get("message") or r.get("summary") or _compact_result_data(r.get("data"))
                  or r.get("error") or ("done" if r.get("success") else "no result"))
        lines.append(f"- {label}: {detail}")
    return "\n".join(lines) or "(nothing has run yet)"


def _build_replan_prompt(
    goal: str, done_steps: list[dict[str, Any]], progress: str, reason: str,
    menu: list[dict[str, Any]],
) -> str:
    """Prompt to CONTINUE an in-flight errand: given the goal, what already ran (+ its
    real results), and the checkpoint's reason, choose the NEXT steps. Returns only the
    additional steps — the done ones are not repeated."""
    menu_text = "\n".join(
        f"- {c['command']}: {c['description']} | args: {json.dumps(c['args'])}" for c in menu
    )
    return (
        "You are Jarvis's errand planner CONTINUING an errand that is already underway. "
        "Some steps have run — here is what ACTUALLY happened. Decide the NEXT steps "
        "needed to finish the goal BASED ON those results, choosing ONLY from the "
        "available commands. Return ONLY the steps still to run (do not repeat the ones "
        "already done).\n\n"
        + _DECOMPOSITION_RULE +
        f"Goal: {goal}\n"
        f"Steps already run: {json.dumps(done_steps)}\n"
        f"What happened:\n{progress}\n\n"
        f"Why we paused to re-plan: {reason}\n\n"
        f"Available commands:\n{menu_text}\n\n"
        + _FABRICATION_GUARDRAIL +
        "If the results mean nothing further is needed, return an EMPTY steps list. "
        'Return a JSON object: {"summary": "<one-line plan for the remaining work>", '
        '"steps": [{"command": "<name>", "args": {...}, "label": "<short label>"}]}. '
        "No markdown.\n\n"
        + _CLOSING_DIRECTIVE
    )


async def replan_from_progress(
    goal: str, done_steps: list[dict[str, Any]], results: list[dict[str, Any]],
    reason: str, llm_client: Any, menu: list[dict[str, Any]] | None = None,
) -> ErrandPlan:
    """Cursor-aware re-plan at a mid-run checkpoint: look at the goal, the steps that
    already ran, and their REAL results, and propose the continuation (the steps to add
    after the checkpoint). Unlike ``plan_errand`` this MAY return zero steps — "the
    results mean there's nothing more to do" is a valid continuation (the run just
    proceeds past the checkpoint). Same empty/unparseable-response failure contract."""
    menu = menu or COMMAND_MENU
    progress = _summarize_progress(done_steps, results)
    data = await _run_planner(_build_replan_prompt(goal, done_steps, progress, reason, menu), llm_client)
    allowed = {c["command"] for c in menu} | CONTROL_COMMANDS
    return _plan_from_data(
        data, allowed, (data.get("summary") or "continue"), _risky_by_cmd(menu),
        require_nonempty=False,
    )
