"""
Jarvis Adapter Model - Adapter-Tuned Model for Jarvis

This model uses the same tool execution loop as JarvisToolModel, with a
system prompt optimized for LoRA adapter usage:
- Full descriptions, antipatterns, and parameter info (rich context)
- Only is_primary examples (adapter trained on full example set)
- No fastText classifier (let LLM + adapter handle routing)
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.models.jarvis_tool_model import JarvisToolModel
from app.core.tool_call_parser import tool_call_parser
from app.core.general_context import get_general_context

logger = logging.getLogger("uvicorn")


class JarvisAdapterModel(JarvisToolModel):
    """
    Adapter-tuned model implementation for Jarvis.

    Uses rich prompt context combined with LoRA adapters trained on
    command schemas and examples for improved accuracy.
    """

    @property
    def name(self) -> str:
        return "JarvisAdapterModel"

    @property
    def use_tool_classifier(self) -> bool:
        """Disable fastText classifier - let LLM + adapter handle routing."""
        return False

    def _build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Build system prompt with full context but only primary examples.

        The adapter is trained on the full example set, so we only need
        primary examples in the prompt for disambiguation hints.
        """
        available_commands = available_commands or []
        node_context = node_context or {}

        room = node_context.get("room", "unknown")
        user = node_context.get("user", "default")
        voice_mode = node_context.get("voice_mode", "brief")

        # Get current date context for date-aware parameter extraction
        date_context_str = get_general_context(timezone)

        # Build direct answer policy section
        must_call_tools = []
        direct_answer_allowed = []
        for cmd in available_commands:
            cmd_name = cmd.get("command_name")
            if not cmd_name:
                continue
            allow_direct = cmd.get("allow_direct_answer")
            if allow_direct is False:
                must_call_tools.append(cmd_name)
            elif allow_direct is True:
                direct_answer_allowed.append(cmd_name)

        direct_answer_section = ""
        if must_call_tools or direct_answer_allowed:
            direct_answer_section = "\nDirect Answer Policy:\n"
            if must_call_tools:
                direct_answer_section += f"- MUST call tools for: {', '.join(sorted(set(must_call_tools)))}\n"
            if direct_answer_allowed:
                direct_answer_section += f"- Direct answers allowed for: {', '.join(sorted(set(direct_answer_allowed)))}\n"

        # Format tools with PRIMARY EXAMPLES ONLY
        tools_section = tool_call_parser.format_tools_for_prompt(
            tools,
            available_commands,
            primary_examples_only=True
        )

        system_prompt = f"""You are Jarvis, a voice assistant that uses tools.
Context: room={room}, user={user}, style={voice_mode}
{date_context_str}

Rules:
- Use tools to fulfill requests. Call ONE tool at a time.
- Select the tool that best matches the user's intent, keywords, or examples; if no tool is a clear match, use get_command_utterance_examples.
- Extract parameters directly from the user's utterance; only use request_validation if required parameters are missing or ambiguous.
{direct_answer_section}

Response format: JSON ONLY (no extra text).
- Tool call: {{"message":"brief ack","tool_calls":[{{"name":"<tool>","arguments":{{...}}}}],"error":null}}
- Final: {{"message":"<concise spoken reply>","tool_calls":[],"error":null}}

Tools:
{tools_section}
"""

        logger.info(
            "📝 Built adapter system prompt: %d characters, %d tools available (primary examples only)",
            len(system_prompt),
            len(tools)
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("📝 System prompt (full):\n%s", system_prompt)

        # Write system prompt to file for inspection
        repo_root = Path(__file__).resolve().parents[3]
        prompt_path = repo_root / "temp" / "adapter_system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(system_prompt, encoding="utf-8")
        logger.info("📝 System prompt written to %s", prompt_path)

        return system_prompt
