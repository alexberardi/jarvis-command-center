"""
HermesMediumMlx - Prompt provider for Hermes 3 via MLX 4-bit quantization.

Inherits from HermesMediumUntrained. Adds post-processing normalization to
fix type-format issues observed with MLX 4-bit inference:
- resolved_datetimes: model outputs "today" (string) instead of ["today"] (array)
- duration_seconds: model outputs "30" (string) instead of 30 (int)

These are the same normalizations already present in Gemma 2, Llama 3.1, Qwen 2.5,
and Mistral providers. The system prompt also includes a concrete example with
correct types to guide generation.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.hermes_medium_untrained import (
    HermesMediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)
from app.core.prompt_providers.shared.tool_formatters import format_tools_for_prompt

logger = logging.getLogger("uvicorn")

# Parameters that must be arrays — normalize bare strings to single-element lists
_ARRAY_PARAMS = frozenset({"resolved_datetimes"})

# Parameters that must be ints — coerce numeric strings
_INT_PARAMS = frozenset({"duration_seconds"})


class HermesMediumMlx(HermesMediumUntrained):
    """
    Prompt provider for Hermes 3 via MLX 4-bit quantization.

    Identical to HermesMediumUntrained except:
    - parse_response normalizes array and int argument types
    - System prompt includes concrete example with correct types
    - Stronger date rule emphasizing array format
    """

    @property
    def name(self) -> str:
        return "HermesMediumMlx"

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Build system prompt with stronger type hints for MLX inference.

        Same structure as HermesMediumUntrained, with two targeted changes:
        1. Date rule emphasizes resolved_datetimes MUST be a JSON array
        2. Format example uses concrete types instead of generic placeholders
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room: str = node_context.get("room", "unknown")
        user: str = node_context.get("user", "default")
        voice_mode: str = node_context.get("voice_mode", "brief")

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # Tool descriptions with primary examples only (for intent guidance)
        tools_section: str = format_tools_for_prompt(
            tools, available_commands, primary_examples_only=True
        )

        # Build <tools> XML block for Hermes's fine-tuned format
        tools_xml: str = HermesMediumUntrained._build_tools_xml(tools)

        system_prompt: str = f"""You are Jarvis, a function calling voice assistant.
Context: room={room}, user={user}, style={voice_mode}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.

{tools_xml}

Rules:
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- resolved_datetimes MUST be a JSON array: ["today"], ["tomorrow"], ["day_after_tomorrow"]. NEVER a bare string. NEVER ISO dates.
- duration_seconds MUST be an integer: 30, 60, 300. NEVER a string.
- Always populate required tool parameters from the user's request.
- For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "get_weather", "arguments": {{"city": "Miami", "resolved_datetimes": ["today"]}}}}
</tool_call>
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.

Tools:
{tools_section}
"""

        logger.info(
            "Built HermesMediumMlx system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def parse_response(self, raw_content: str) -> Optional[str]:
        """
        Transform Hermes output into Jarvis JSON, then normalize argument types.

        Calls the parent parse_response (handles <tool_call> extraction, scratch_pad
        stripping, plain text wrapping), then post-processes any tool_calls to fix:
        - resolved_datetimes: bare string → single-element array
        - duration_seconds: numeric string → int
        """
        result: Optional[str] = super().parse_response(raw_content)
        if result is None:
            return None

        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return result

        tool_calls = parsed.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            return result

        modified: bool = False
        for call in tool_calls:
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                continue

            # Normalize array parameters: wrap bare strings in a list
            for key in _ARRAY_PARAMS:
                if key in arguments and isinstance(arguments[key], str):
                    arguments[key] = [arguments[key]]
                    modified = True

            # Normalize int parameters: coerce numeric strings to int
            for key in _INT_PARAMS:
                if key in arguments and isinstance(arguments[key], str):
                    try:
                        arguments[key] = int(arguments[key])
                        modified = True
                    except ValueError:
                        pass

        if modified:
            return json.dumps(parsed)
        return result

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            **super().get_capabilities(),
            "provider_name": self.name,
            "training_tier": "untrained",
            "quantization": "mlx-4bit",
        }
