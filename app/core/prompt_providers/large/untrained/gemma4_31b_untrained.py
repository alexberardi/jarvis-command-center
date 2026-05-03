"""
Gemma4_31B_Untrained - Prompt provider for Google Gemma 4 31B Instruct (GGUF).

Forked from Gemma3Compressed (the best-performing Gemma provider). Uses the
same text-based <tool_call> format, inline rules, and compressed tools listing.

31B-specific tuning vs Gemma 3 12B:
- Strips Gemma 4's native control tokens that leak through llama-cpp-python:
  thinking channel (<|channel>thought...<channel|>), turn markers
  (<|turn>model, <turn|>), string delimiters (<|"|>), and ChatML tokens.
- Stronger DT_KEYS enforcement — the 31B model is more inclined to resolve
  dates and do math itself instead of passing symbolic keys.
- Language rule — larger multilingual models occasionally code-switch on
  ambiguous queries.
- Best-effort tool call preference — larger models are more likely to ask
  for clarification instead of making a tool call.
- No force_tool_calls — 31B is capable enough to follow Direct Answer
  guidance without the retry mechanism.

Inherits build_tools, _build_tools_xml, build_training_completion,
get_response_format, supports_native_tools, and use_tool_classifier from
Gemma3MediumUntrained. Overrides parse_response and sanitize_text to strip
Gemma 4 control tokens.

chat_format: Use "gemma4" — custom handler in llm-proxy that uses Gemma 4's
native template with thinking disabled. Prefills an empty thinking channel
(<|channel>thought\n<channel|>) so the model skips reasoning entirely (~2-5x
faster). Falls back to "chatml" (NOT "gemma") if gemma4 handler unavailable —
the gemma chat format drops system messages entirely.

Ref: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
Ref: https://ai.google.dev/gemma/docs/capabilities/thinking
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from app.core.prompt_providers.medium.untrained.gemma3_medium_untrained import (
    Gemma3MediumUntrained,
)
from app.core.prompt_providers.shared.context_builders import (
    build_agent_context_summary,
    build_direct_answer_section,
)

logger = logging.getLogger("uvicorn")

# --- Gemma 4 control token patterns ---
# These tokens are native to Gemma 4's template and leak through when
# using chatml chat_format with llama-cpp-python.

# Thinking channel: <|channel>thought\n...\n<channel|>
# Empty (thinking disabled) or full (model reasoned anyway).
_THINK_CHANNEL_RE = re.compile(
    r"<\|channel>thought\s*.*?<channel\|>\s*", re.DOTALL
)

# Turn markers: <|turn>model, <|turn>user, <turn|>
_TURN_MARKER_RE = re.compile(r"<\|turn>\w*\s*|<turn\|>\s*")

# Gemma 4 string delimiter: <|"|>  (used instead of quotes in native format)
_STRING_DELIM_RE = re.compile(r'<\|"\|>')

# ChatML tokens that may leak (parent handles these too, but belt-and-suspenders)
_CHATML_TOKEN_RE = re.compile(r"<\|im_(?:start|end)\|>\s*(?:assistant|system|user)?\s*")

# Thinking mode activation token
_THINK_ACTIVATE_RE = re.compile(r"<\|think\|>\s*")

# Comment-style thinking: lines starting with // (model uses as CoT)
_COMMENT_THINK_RE = re.compile(r"^\s*//[^\n]*$", re.MULTILINE)


def _strip_gemma4_tokens(text: str) -> str:
    """Remove all Gemma 4 control tokens and comment-style thinking from text."""
    text = _THINK_CHANNEL_RE.sub("", text)
    text = _TURN_MARKER_RE.sub("", text)
    text = _STRING_DELIM_RE.sub('"', text)
    text = _CHATML_TOKEN_RE.sub("", text)
    text = _THINK_ACTIVATE_RE.sub("", text)
    text = _COMMENT_THINK_RE.sub("", text)
    # Collapse multiple blank lines left by stripping
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


class Gemma4_31B_Untrained(Gemma3MediumUntrained):
    """
    Prompt provider for Google Gemma 4 31B Instruct (untrained, GGUF).

    Compressed prompt with inline rules, text-based <tool_call> format,
    and large-model tuning for DT_KEYS enforcement, language stability,
    and best-effort tool call preference. Strips Gemma 4's
    <|channel>thought...<channel|> thinking blocks.
    """

    @property
    def name(self) -> str:
        return "Gemma4_31B_Untrained"

    def parse_response(self, raw_content: str) -> Optional[str]:
        """Strip Gemma 4 control tokens, then delegate to Gemma 3 parser.

        Gemma 4 leaks native tokens through llama-cpp-python when using
        chatml format: thinking channel, turn markers, string delimiters,
        and <|think|> activation tokens. All are stripped before the parent
        parser extracts <tool_call> blocks or wraps plain text.
        """
        cleaned: str = _strip_gemma4_tokens(raw_content)
        return super().parse_response(cleaned)

    def sanitize_text(self, text: str) -> str:
        """Strip Gemma 4 control tokens from user-facing text.

        Defense-in-depth for direct answers, wake responses, and TTS paths.
        Covers thinking channel, turn markers, string delimiters, and any
        ChatML tokens that leak through.
        """
        return _strip_gemma4_tokens(text).strip()

    def build_system_prompt(
        self,
        node_context: Dict[str, Any],
        timezone: Optional[str],
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        available_commands = available_commands or []
        node_context = node_context or {}

        date_keys: list[str] = node_context.get("date_keys", [])

        identity: str = self.build_context_header(node_context)

        # Shared sections
        direct_answer_section: str = build_direct_answer_section(available_commands)
        agent_context_section: str = build_agent_context_summary(node_context)

        # <tools> XML block (inherited from Gemma3MediumUntrained)
        tools_xml: str = Gemma3MediumUntrained._build_tools_xml(tools)

        # DT_KEYS — strong enforcement for 31B (model aggressively resolves dates)
        dt_keys_line: str = ""
        if date_keys:
            dt_keys_line = (
                f"\nDT_KEYS: {'|'.join(date_keys)}\n"
                "CRITICAL — resolved_datetimes: You MUST use the symbolic key strings from DT_KEYS "
                "(e.g., \"today\", \"tomorrow\", \"this_weekend\", \"last_weekend\"). "
                "NEVER resolve dates to ISO timestamps like \"2026-03-07T05:00:00Z\" — the server handles resolution. "
                "NEVER compute relative dates (e.g., do NOT calculate \"3 days from now\" into a date). "
                "If the user omits a date, pass [\"today\"].\n"
            )

        system_prompt: str = f"""{identity}

You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You MUST call a function for any request that matches an available tool. Do not make assumptions about what values to plug into functions.

{tools_xml}

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{{"name": "<function-name>", "arguments": {{"<arg-name>": "<arg-value>"}}, "failure_message": "<brief spoken response if this call fails>"}}
</tool_call>

Example — if the user says "What's the weather in Miami?", respond ONLY with:
<tool_call>
{{"name": "get_weather", "arguments": {{"city": "Miami", "resolved_datetimes": ["today"]}}}}
</tool_call>

Rules:
- NEVER think out loud, reason, or include comments (// or otherwise) in your output. Respond ONLY with a tool call or a brief spoken answer. No preamble, no explanation, no commentary.
- You MUST call a tool for weather, sports, calendar, timers, music, device control, web search, and all other tool-covered domains. NEVER answer these from memory.
- Call ONE tool at a time to fulfill requests.
- Pick the tool that best matches intent; use get_command_utterance_examples if unsure.
- Extract parameters from the user's words; only request validation if required params are truly missing/ambiguous.
- Always populate required tool parameters from the user's request.
- ALWAYS respond in the language the user spoke. Prefer making a best-effort tool call over asking for clarification.
{dt_keys_line}
{direct_answer_section}
{agent_context_section}
For final answers with no tool needed, respond with a brief spoken reply.
"""

        logger.info(
            "Built Gemma4_31B_Untrained system prompt: %d chars, %d tools",
            len(system_prompt),
            len(tools),
        )

        if os.getenv("LOG_FULL_SYSTEM_PROMPT", "false").lower() in {"1", "true", "yes"}:
            logger.info("System prompt (full):\n%s", system_prompt)

        return system_prompt

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider_name": self.name,
            "model_family": "gemma",
            "size_tier": "large",
            "training_tier": "untrained",
            "use_tool_classifier": self.use_tool_classifier,
            "supports_native_tools": self.supports_native_tools,
        }
