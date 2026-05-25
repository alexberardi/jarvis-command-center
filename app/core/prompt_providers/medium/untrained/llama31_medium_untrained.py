"""
Llama31MediumUntrained - Prompt provider for Meta Llama 3.1 8B Instruct.

Optimized for Llama 3.1 8B Instruct (Q4_K_M GGUF) with text-based tool calling.

Key features:
- Tools presented as JSON schemas with <function=name>{args}</function> call format
- Leverages Llama 3.1's fine-tuned function-calling format
- parse_response transforms <function=name>{args}</function> tags into Jarvis JSON
- supports_native_tools=False (text-based): model outputs function tags, not
  structured tool_calls via llama-cpp-python's tools parameter
- build_tools() ready for native path via ToolBuilder
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.interfaces.ijarvis_prompt_provider import IJarvisPromptProvider
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.core_rules import (
    ANTI_HALLUCINATION_MANDATE,
    build_fallback_line,
    build_rules_block,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt
from app.core.tool_builder import ToolBuilder

logger = logging.getLogger("uvicorn")

# Pattern for Llama 3.1 native function call format: <function=name>{...}</function>
# Also matches variant without '=' sign: <function>name{...}</function>
# Also matches malformed closing tag: <function=name>{...}<function> (missing slash)
_FUNCTION_CALL_RE = re.compile(
    r"<function[=>](\w+)>(.*?)</?\s*function>", re.DOTALL
)
# Variant: 3B model outputs <function=name {args}></function> (space, no > after name)
_FUNCTION_CALL_SPACE_RE = re.compile(
    r"<function[=>](\w+)\s+(\{.*?\})>(?:</?\s*function>)?", re.DOTALL
)
# Fallback: model emits <function=name>{...} without closing </function> tag
_FUNCTION_CALL_UNCLOSED_RE = re.compile(
    r"<function[=>](\w+)>(\{.*)", re.DOTALL
)
# Tolerant final fallback: handles adapter-induced drift where the model drops
# the '>' after the name (<function>name{args}...) or the final '>' on the
# closing tag (<function>name>{args}</function). Terminates the args at the
# first balanced closing brace via non-greedy match; trailing </function(...)
# is stripped downstream.
_FUNCTION_CALL_TOLERANT_RE = re.compile(
    r"<function[=>](\w+)>?\s*(\{.*?\})", re.DOTALL
)
# Observed drift variants: parens/brackets instead of angle brackets, e.g.
# "<function)name>{...}", "(function)name{...}", "(function:name>{...}",
# "{function=\"name\"}{...}". Single tolerant regex accepting common wrappers.
_FUNCTION_CALL_DRIFT_RE = re.compile(
    r"[<(\{]function[=>:)\"]?\s*(\w+)[>\"\}]?\s*(\{.*?\})", re.DOTALL
)
# Ultra-tolerant: <name>{args}... — model drops the "function=" prefix and
# uses the tool name directly as the tag (e.g., <music>{...}[/music]). Accepts
# any \w+ tag name and any closing delimiter style. Downstream validation
# rejects non-registered names.
_XML_WRAPPED_CALL_RE = re.compile(
    r"<(\w+)>\s*(\{.*?\})", re.DOTALL
)
# Split-tag variant: <function>name</function> {args} — model closes the tag
# before emitting arguments. Parser pairs the name from inside the tag with
# the JSON object that follows.
_SPLIT_TAG_CALL_RE = re.compile(
    r"<function>(\w+)</function>\s*(\{.*?\})", re.DOTALL
)
# No-args variant: <function=name /> or <function=name/> — self-closing tag
# with no arguments. Valid when the called tool has no required parameters
# (e.g. tell_joke with optional topic). Without this, the model's emitted
# tag falls through to plain-text wrapping and gets TTS'd to the user.
_FUNCTION_CALL_NOARGS_RE = re.compile(
    r"<function[=>](\w+)\s*/>", re.DOTALL
)
# No-args drift: <function)name></function>, <function:name></function>,
# <function)name</function>, <function=name></function> — bare tag with no
# arguments AND a paren/colon/equals drift separator. Hit on tools with all
# optional parameters (e.g. get_current_time with no location specified).
# Without this, the tag falls through to plain-text wrapping and TTS'd
# scaffolding reaches the user.
_FUNCTION_CALL_DRIFT_NOARGS_RE = re.compile(
    r"<function[=>:)\"]\s*(\w+)\s*>?\s*</\s*function\s*>", re.DOTALL
)

# Parameters that are always arrays — normalize string values to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})

# Strip patterns for sanitize_text — remove function-call scaffolding that
# leaks into post-tool follow-up prose. Small models (3B) hallucinate a
# second tool call inside the natural-language summary; if it isn't scrubbed,
# the raw <function=...>{...}</function> text reaches the user / TTS.
_STRIP_FUNCTION_CLOSED_RE = re.compile(
    r"<function[^>]*>.*?</?\s*function>", re.DOTALL
)
_STRIP_FUNCTION_UNCLOSED_RE = re.compile(
    r"<function[^>]*>\s*\{.*", re.DOTALL
)
_STRIP_SPLIT_TAG_RE = re.compile(
    r"<function>\w+</function>\s*\{.*?\}", re.DOTALL
)
_STRIP_FUNCTION_NOARGS_RE = re.compile(
    r"<function[^>/]*\s*/>", re.DOTALL
)
# Strip variant for the drift-noargs form (paren/colon separator, no args).
_STRIP_FUNCTION_DRIFT_NOARGS_RE = re.compile(
    r"<function[=>:)\"]\s*\w+\s*>?\s*</\s*function\s*>", re.DOTALL
)


class Llama31MediumUntrained(IJarvisPromptProvider):
    """
    Prompt provider for Meta Llama 3.1 8B Instruct (untrained).

    Strategy:
    - Tools as JSON schemas with <function=name>{args}</function> call format
    - Matches Llama 3.1's fine-tuned function-calling training
    - Agent context (HA devices) included for device awareness
    - Primary examples only to save context window
    - fastText classifier enabled for routing hints
    """

    @property
    def name(self) -> str:
        return "Llama31MediumUntrained"

    @property
    def use_tool_classifier(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        # Text-based <function=...> format is more reliable than
        # structured tool_calls via llama-cpp-python for this model.
        return False

    def build_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build OpenAI-format tool definitions using ToolBuilder."""
        return ToolBuilder.build(tools)

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt using Llama 3.1's native <function=...> format.

        Tools are presented as JSON schemas and the model is instructed to
        emit calls using <function=name>{"param": "value"}</function> tags,
        matching Llama 3.1's function-calling fine-tuning.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        # Read Direct Answer Policy directly from the tools list (each tool
        # has allow_direct_answer baked in via to_openai_tool_schema). This
        # is more robust than reading from available_commands, which goes
        # through an upstream merge that can drop flags for Pantry commands.
        _tool_flags = []
        for t in tools:
            fn = t.get("function") if isinstance(t.get("function"), dict) else None
            name = (fn.get("name") if fn else None) or t.get("name")
            if not name:
                continue
            _tool_flags.append({
                "command_name": name,
                "allow_direct_answer": t.get("allow_direct_answer"),
            })
        direct_answer_section: str = build_direct_answer_section(_tool_flags)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Tool descriptions with primary examples only (for intent guidance)
        tools_section: str = format_tools_for_prompt(
            tools, available_commands, primary_examples_only=True
        )

        # Build JSON schemas for tools. Compact dump (no indent) —
        # the JSON block is the single largest contributor to prompt
        # size (≈65% with a full tool set), and pretty-printing adds
        # ~1.4k tokens of whitespace for 23 tools without changing
        # what the model sees semantically.
        clean_tools: List[Dict[str, Any]] = ToolBuilder.build(tools)
        tool_json: str = json.dumps(clean_tools, separators=(",", ":")) if clean_tools else "[]"

        # Shared header
        identity: str = self.build_context_header(node_context)

        # Shared rules (Llama31: custom param_names_rule with NOT "param")
        rules: str = build_rules_block(
            param_names_rule='Use the actual parameter names from the function schema above \u2014 NOT "param".',
        )

        # Shared fallback
        fallback: str = build_fallback_line()

        system_prompt: str = f"""{identity}

You are a function calling AI model. You are provided with function signatures below. Always include all required parameters — use sensible defaults from context when the user does not state them explicitly. {ANTI_HALLUCINATION_MANDATE}

To call a function, respond with:
<function=function_name>{{"arg_name": "value"}}</function>

For example, to get weather: <function=get_weather>{{"city": "Miami"}}</function>

CRITICAL: Use ONLY the exact function names from the "Available functions" list below. Do NOT invent names (e.g., "calendar_today", "check_calendar") or use dotted service paths (e.g., "light.turn_off", "climate.set_temperature"). Pick the single best-matching function by name from the list and pass its parameters.

Available functions:
{tool_json}

{rules}
{direct_answer_section}
{agent_context_section}
{fallback}

Tools:
{tools_section}
"""

        logger.info(
            "Built Llama31MediumUntrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_response_format(self) -> Optional[Dict[str, Any]]:
        """Return text mode — Llama 3.1 outputs <function=...> tags, not JSON."""
        return {"type": "text"}

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Llama 3.1 native output into Jarvis JSON format.

        Llama 3.1 emits tool calls as <function=name>{"arg":"val"}</function>.
        This method:
        1. Extracts ALL <function=name>{...}</function> blocks and builds Jarvis JSON
        2. Wraps plain text responses as Jarvis JSON messages
        3. Returns None for content already in Jarvis JSON format (passthrough)

        Returns:
            Transformed Jarvis JSON string, or None if no transformation needed.
        """
        cleaned: str = raw_content.strip()

        # Extract ALL <function=name>{...}</function> blocks
        function_matches = _FUNCTION_CALL_RE.findall(cleaned)

        # Split-tag: <function>name</function> {args} (args outside closing tag)
        if not function_matches:
            function_matches = _SPLIT_TAG_CALL_RE.findall(cleaned)

        # Fallback: try <function=name {args}> variant (space between name and args)
        if not function_matches:
            function_matches = _FUNCTION_CALL_SPACE_RE.findall(cleaned)

        # Fallback: try unclosed <function=name>{...} (no </function> tag)
        if not function_matches:
            unclosed_match = _FUNCTION_CALL_UNCLOSED_RE.search(cleaned)
            if unclosed_match:
                function_matches = [unclosed_match.groups()]

        # Tolerant final fallback for adapter-induced drift (missing '>' chars)
        if not function_matches:
            function_matches = _FUNCTION_CALL_TOLERANT_RE.findall(cleaned)

        # Drift fallback: paren/brace wrappers like <function)name> or (function)name
        if not function_matches:
            function_matches = _FUNCTION_CALL_DRIFT_RE.findall(cleaned)

        # Ultra-tolerant: <name>{args}</name> — name-as-tag variant. Reject
        # generic "function" matches (handled by the patterns above) to avoid
        # false extraction of literal "function" as the tool name.
        if not function_matches:
            raw_matches = _XML_WRAPPED_CALL_RE.findall(cleaned)
            function_matches = [(n, a) for (n, a) in raw_matches if n != "function"]

        # No-args self-closing: <function=name /> — pair the name with an
        # empty JSON object so the existing arg-parsing loop yields {}.
        if not function_matches:
            noargs_names = _FUNCTION_CALL_NOARGS_RE.findall(cleaned)
            if noargs_names:
                function_matches = [(name, "{}") for name in noargs_names]

        # No-args drift: <function)name></function> and similar drift wrappers
        # with NO args block. Common for tools whose params are all optional
        # (get_current_time with no location).
        if not function_matches:
            drift_noargs_names = _FUNCTION_CALL_DRIFT_NOARGS_RE.findall(cleaned)
            if drift_noargs_names:
                function_matches = [(name, "{}") for name in drift_noargs_names]

        if function_matches:
            parsed_calls: list[Dict[str, Any]] = []
            for func_name, args_str in function_matches:
                try:
                    cleaned_args = args_str.strip().rstrip(">;\"'")
                    # Strip trailing </function> or </function (partial/unclosed) —
                    # the unclosed fallback captures everything after the args
                    # including the stray closing tag.
                    for tail in ("</function>", "</function", "</fn>", "</fn"):
                        if cleaned_args.endswith(tail):
                            cleaned_args = cleaned_args[: -len(tail)].rstrip()
                            break
                    # If the JSON is still truncated mid-object, try brace-matching
                    # to the last closing brace.
                    if cleaned_args.startswith("{") and not cleaned_args.endswith("}"):
                        last = cleaned_args.rfind("}")
                        if last > 0:
                            cleaned_args = cleaned_args[: last + 1]
                    # Empty args block (<function=name></function>) — treat as
                    # no-args call. Otherwise json.loads('') raises and we
                    # drop the call entirely, leaving scaffolding to leak.
                    if not cleaned_args:
                        arguments = {}
                    else:
                        try:
                            arguments = json.loads(cleaned_args)
                        except json.JSONDecodeError:
                            # Over-escaped JSON: model emits {\"key\": \"val\"} with
                            # literal backslash-quote. Strip and retry.
                            arguments = json.loads(cleaned_args.replace('\\"', '"'))
                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse function args for %s: %s",
                        func_name,
                        args_str[:100],
                    )
                    continue
                # Unwrap {"param": {...}} nesting — the model sometimes wraps
                # real arguments inside a literal "param" key from the format
                # example. Unwrap only when "param" is the sole key and its
                # value is a dict (the actual arguments).
                if (
                    isinstance(arguments, dict)
                    and len(arguments) == 1
                    and "param" in arguments
                    and isinstance(arguments["param"], dict)
                ):
                    arguments = arguments["param"]
                # Normalize array parameters: wrap string values in a list
                if isinstance(arguments, dict):
                    for key in _ARRAY_PARAMS:
                        if key in arguments and isinstance(arguments[key], str):
                            arguments[key] = [arguments[key]]
                parsed_calls.append({
                    "name": func_name,
                    "arguments": arguments,
                })
            if parsed_calls:
                jarvis_json: Dict[str, Any] = {
                    "message": "",
                    "tool_calls": parsed_calls,
                    "error": None,
                }
                return json.dumps(jarvis_json)

        # Check if content is already Jarvis JSON (passthrough)
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                return None
        except json.JSONDecodeError:
            pass

        # Plain text response — wrap as Jarvis message
        if cleaned and not cleaned.startswith("{"):
            return json.dumps({
                "message": cleaned,
                "tool_calls": [],
                "error": None,
            })

        return None

    def sanitize_text(self, text: str) -> str:
        """Strip Llama-style <function=...> tags from user-facing prose.

        Small models (3.2 3B) regularly hallucinate a second tool call
        inside post-tool follow-up summaries. The conversation_handler
        runs this on every formatted response, so model-specific scaffolding
        is scrubbed before TTS / chat rendering.
        """
        cleaned = _STRIP_FUNCTION_CLOSED_RE.sub("", text)
        cleaned = _STRIP_SPLIT_TAG_RE.sub("", cleaned)
        cleaned = _STRIP_FUNCTION_UNCLOSED_RE.sub("", cleaned)
        cleaned = _STRIP_FUNCTION_NOARGS_RE.sub("", cleaned)
        cleaned = _STRIP_FUNCTION_DRIFT_NOARGS_RE.sub("", cleaned)
        return cleaned.strip()

    def build_training_system_prompt(self) -> str:
        """System message for training — matches Llama 3.1's <function=name> format."""
        return (
            "You are a function calling AI model. "
            "To call a function, respond with:\n"
            '<function=function_name>{"arg_name": "value"}</function>'
        )

    def build_training_completion(self, tool_call: Dict[str, Any]) -> str:
        """Format as <function=name>{args}</function> matching Llama 3.1's output."""
        name: str = tool_call.get("name", "")
        args: dict = tool_call.get("arguments", {})
        return f" <function={name}>{json.dumps(args)}</function>"

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "llama",
            "size_tier": "medium",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }
