"""
System prompt builder for Jarvis voice assistant.

This module provides utilities for building system prompts and response formats
for the LLM-based voice command processing.
"""

import os
from typing import Any, Dict, List, Optional

from app.core.tool_call_parser import tool_call_parser


def format_tools_for_prompt(
    tools: List[Dict[str, Any]],
    available_commands: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format tools for inclusion in the system prompt.

    Args:
        tools: List of tool definitions in OpenAI format
        available_commands: Optional list of CommandDefinition dicts for examples

    Returns:
        Formatted string describing tools
    """
    return tool_call_parser.format_tools_with_examples(tools, available_commands)


def get_response_format() -> Dict[str, Any]:
    """
    Get the JSON response format schema for LLM responses.

    Returns:
        Response format dict with JSON schema
    """
    return {
        "type": "json_object",
        "json_schema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "arguments": {"type": "object"},
                        },
                        "required": ["name", "arguments"],
                    },
                },
                "error": {
                    "type": ["object", "null"],
                    "properties": {
                        "type": {"type": "string"},
                        "message": {"type": "string"},
                    },
                    "required": ["type", "message"],
                },
            },
            "required": ["message", "tool_calls"],
            "additionalProperties": True,
        },
    }


def build_tool_system_message(
    node_context: Dict[str, Any],
    timezone: Optional[str],
    tools: List[Dict[str, Any]],
    available_commands: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Build system message for tool-based conversations.

    Args:
        node_context: Node context information
        timezone: User's timezone
        tools: Available tools
        available_commands: Optional available commands for examples

    Returns:
        System message string
    """
    prompt_style = os.getenv("JARVIS_PROMPT_STYLE", "compact").strip().lower()

    # Format tools for prompt with primary examples
    tools_text = format_tools_for_prompt(tools, available_commands)

    include_reason = os.getenv("JARVIS_INCLUDE_REASON", "").strip().lower() in {"1", "true", "yes"}
    reason_suffix_inline = ', "reason": "<optional>"' if include_reason else ""
    reason_suffix_block = '\n  "reason": "<optional>"' if include_reason else ""

    room = node_context.get("room", "unknown")
    speaker_name = node_context.get("speaker_name") or node_context.get("user", "default")
    voice_mode = node_context.get("voice_mode", "brief")
    user_memories = node_context.get("user_memories", "")

    # Build memory block if memories exist
    memory_block = ""
    if user_memories:
        memory_block = f"\nAbout {speaker_name}:\n{user_memories}\n"

    if prompt_style == "full":
        system_msg = f"""You are Jarvis, a voice-controlled assistant that operates BY CALLING TOOLS.

Node Context:
- Room: {room}
- User: {speaker_name}
- Voice Mode: {voice_mode}
{memory_block}
YOUR PRIMARY ROLE: You are a tool router and parameter extractor.
- Analyze the user's request
- Determine which available tool(s) to call
- Extract parameters from their message
- Call the appropriate tool(s)

CRITICAL - Tool Execution Order:
- Call tools ONE AT A TIME in the order they need to be executed.
- Do NOT select the final command/tool until all parameters have been resolved.
- If parameters need resolution (dates, examples, etc.), resolve them FIRST before calling the final tool.
- Read tool descriptions to understand their purpose and when to use them.

CRITICAL: You do NOT have direct access to information or the ability to perform actions yourself.
- When a user asks a question or makes a request, you MUST use the available tools
- DO NOT attempt to answer questions, provide information, or perform actions without calling a tool
- Review the available tools and select the one that best matches the user's intent
- After you receive tool results, THEN you format the data into a natural spoken response for the user

Think of yourself as operating in two phases:
1. TOOL CALLING: Route the request to the appropriate tool(s) - you cannot answer directly
2. RESPONSE FORMATTING: Once you receive tool results, format them into a conversational response for the user

EXCEPTION: If it's something you can absolutely figure out without a tool (like basic math or who you are), feel free to respond directly.

CRITICAL - Response Format (VALID JSON ONLY):
You MUST respond with valid JSON only. No comments, no explanations, no markdown, no code blocks.
- Comments (like # ...) are NOT allowed in JSON
- Only valid JSON syntax is accepted - any other text will cause errors
{"- You may include an optional 'reason' field explaining tool/parameter choices." if include_reason else ""}

When calling a tool (ONE at a time):
{{
  "message": "Brief acknowledgment",
  "tool_call": {{"name": "tool_name", "arguments": {{"param_name": "param_value"}}, "failure_message": "brief spoken response if this fails"}}{reason_suffix_block}
}}

When you have the final answer:
{{
  "message": "Your natural spoken response",
  "tool_call": null{reason_suffix_block}
}}

Available Tools:
{tools_text}

Key Guidelines:
- Extract parameters directly from user's message when they are clearly stated. If the user says "Seattle", extract "Seattle" as the city parameter - do not ask for validation when the information is already provided.
- Only ask for validation if a required parameter is truly missing or genuinely ambiguous after trying all other options.
- Be conversational and {voice_mode}"""
    else:
        system_msg = f"""You are Jarvis. Always call tools; respond with VALID JSON only.

Context: room={room}, speaker={speaker_name}, style={voice_mode}
{memory_block}
Rules:
- Call ONE tool at a time.
- Relative dates are resolved automatically; use natural terms like 'tomorrow' or 'next_week' in date parameters.
- Ask for validation only if required params are missing/ambiguous.
{"- You may include an optional 'reason' field explaining tool/parameter choices." if include_reason else ""}

Format:
- Tool call: {{"message":"brief ack","tool_call":{{"name":"tool_name","arguments":{{...}},"failure_message":"brief spoken response if this fails"}}{reason_suffix_inline}}}
- Final: {{"message":"concise reply","tool_call":null{reason_suffix_inline}}}

Tools:
{tools_text}"""

    return system_msg
