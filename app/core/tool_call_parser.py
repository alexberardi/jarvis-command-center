"""
Tool Call Parser for LLM Responses.

Parses tool calls from LLM JSON responses for models that don't have
native tool calling support (e.g., local Llama models via vLLM/llama.cpp).
"""

import json
import os
import uuid
import logging
import re
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("uvicorn")


class ToolCallParser:
    """Parser for extracting tool calls from LLM JSON responses."""
    
    @staticmethod
    def parse_response(llm_output: str) -> Tuple[str, List[Dict[str, Any]], str]:
        """
        Parse LLM response to extract tool calls.
        
        Expected LLM output format (OpenAI-style):
        {
            "message": "Response to the user",
            "tool_calls": [
                {"name": "tool_name", "arguments": {"param": "value"}}
            ]
        }
        
        Or for no tool calls (empty list):
        {
            "message": "Response to the user",
            "tool_calls": []
        }
        
        Args:
            llm_output: Raw output from LLM
            
        Returns:
            Tuple of (finish_reason, tool_calls, assistant_message)
            - finish_reason: "stop" or "tool_calls"
            - tool_calls: List of tool call dicts in internal format
            - assistant_message: The message content
        """
        logger.debug(f"Parsing LLM output: {llm_output[:200]}...")
        
        # Log full output for debugging
        logger.info(f"🔍 Raw LLM output (first 500 chars): {llm_output[:500]}")
        logger.info(f"🔍 Output length: {len(llm_output)} characters")
        
        try:
            # Clean JSON by removing comments (JSON doesn't support comments)
            cleaned_output = ToolCallParser._strip_json_comments(llm_output)
            
            # Try to parse as JSON
            parsed = json.loads(cleaned_output)
            
            # Extract message
            message = parsed.get("message", "")
            
            # Extract tool calls (array preferred)
            tool_calls_raw = parsed.get("tool_calls")
            formatted_calls = []

            if isinstance(tool_calls_raw, list) and tool_calls_raw:
                for tool_call_raw in tool_calls_raw:
                    formatted_call = ToolCallParser._format_tool_call(tool_call_raw)
                    if formatted_call:
                        formatted_calls.append(formatted_call)

            # Fallback: accept singular tool_call if provided
            if not formatted_calls:
                tool_call_raw = parsed.get("tool_call")
                if tool_call_raw:
                    formatted_call = ToolCallParser._format_tool_call(tool_call_raw)
                    if formatted_call:
                        formatted_calls.append(formatted_call)

            if formatted_calls:
                logger.info(f"✅ Parsed {len(formatted_calls)} tool call(s) from LLM response")
                return "tool_calls", formatted_calls, message
            
            else:
                # No tool call - conversation complete
                logger.info(f"✅ Parsed complete response (no tool call)")
                return "stop", [], message
        
        except json.JSONDecodeError as e:
            # Not valid JSON - treat as plain text response
            logger.warning(f"⚠️ Could not parse LLM output as JSON: {e}")
            logger.debug(f"Raw output: {llm_output}")
            
            # Attempt to extract JSON from text
            extracted = ToolCallParser._extract_json_from_text(llm_output)
            if extracted:
                logger.info("✅ Extracted JSON from text, retrying parse")
                return ToolCallParser.parse_response(extracted)
            
            # Give up, return as plain text
            return "stop", [], llm_output
        
        except Exception as e:
            logger.error(f"❌ Unexpected error parsing LLM output: {e}")
            logger.debug(f"Raw output: {llm_output}")
            return "stop", [], llm_output

    @staticmethod
    def _format_tool_call(tool_call_raw: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_call_raw, dict):
            return None
        name = tool_call_raw.get("name")
        arguments = tool_call_raw.get("arguments", {})

        # Support OpenAI format: {"function": {"name": ..., "arguments": ...}}
        if not name and "function" in tool_call_raw:
            function = tool_call_raw.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments", arguments)

        if not name:
            return None

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse tool arguments: {arguments}")

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            return None

        tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments)
            }
        }
    
    @staticmethod
    def _strip_json_comments(text: str) -> str:
        """
        Strip comments from JSON string.
        
        Removes:
        - Single-line comments starting with #
        - Inline comments after values (e.g., "value"  # comment)
        
        Args:
            text: JSON string potentially containing comments
            
        Returns:
            JSON string with comments removed
        """
        # Remove single-line comments (# ...)
        # Match # followed by any characters except newline, but preserve strings
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Check if line contains a comment outside of a string
            in_string = False
            escape_next = False
            result = []
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if escape_next:
                    result.append(char)
                    escape_next = False
                elif char == '\\':
                    result.append(char)
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                    result.append(char)
                elif char == '#' and not in_string:
                    # Found comment outside string - stop here
                    break
                else:
                    result.append(char)
                
                i += 1
            
            cleaned_lines.append(''.join(result))
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _extract_json_from_text(text: str) -> Optional[str]:
        """
        Attempt to extract JSON object from text that may contain markdown or other formatting.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Extracted JSON string or None
        """
        # First try to extract balanced JSON objects and return the first valid one.
        candidates = ToolCallParser._extract_balanced_json_objects(text)
        for candidate in candidates:
            cleaned_json = ToolCallParser._strip_json_comments(candidate)
            try:
                json.loads(cleaned_json)
                return cleaned_json
            except json.JSONDecodeError:
                continue

        # Fallback: simple slice between first { and last }
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = text[start_idx:end_idx + 1]
            cleaned_json = ToolCallParser._strip_json_comments(potential_json)
            try:
                json.loads(cleaned_json)
                return cleaned_json
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _extract_balanced_json_objects(text: str) -> List[str]:
        """
        Extract balanced JSON object substrings from text.
        Uses a simple brace matcher that ignores braces inside strings.
        """
        objects: List[str] = []
        in_string = False
        escape_next = False
        depth = 0
        start_idx = None

        for idx, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '{':
                if depth == 0:
                    start_idx = idx
                depth += 1
            elif char == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start_idx is not None:
                        objects.append(text[start_idx:idx + 1])
                        start_idx = None

        return objects
    
    @staticmethod
    def format_tools_for_prompt(
        tools: List[Dict[str, Any]],
        available_commands: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format tools as text for inclusion in system prompt.
        
        Args:
            tools: List of tool definitions in OpenAI format
            
        Returns:
            Formatted string describing tools
        """
        if not tools:
            return "No tools available."
        
        # Determine how many examples to include per command
        max_examples_env = os.getenv("JARVIS_EXAMPLES_PER_COMMAND", "").strip().lower()
        if not max_examples_env:
            max_examples = 3
        elif max_examples_env in {"all", "0", "-1"}:
            max_examples = None
        else:
            try:
                max_examples = max(1, int(max_examples_env))
            except ValueError:
                logger.warning("⚠️ Invalid JARVIS_EXAMPLES_PER_COMMAND=%r; defaulting to 3", max_examples_env)
                max_examples = 3

        # Build a map of command_name -> ordered example list
        command_examples = {}
        if available_commands:
            for cmd_dict in available_commands:
                cmd_name = cmd_dict.get("command_name")
                examples = cmd_dict.get("examples", [])
                if not cmd_name or not examples:
                    continue
                primary = [ex for ex in examples if ex.get("is_primary", False)]
                secondary = [ex for ex in examples if not ex.get("is_primary", False)]
                command_examples[cmd_name] = primary + secondary

        tool_descriptions = []
        available_names = {tool.get("function", {}).get("name") for tool in tools if isinstance(tool.get("function"), dict)}
        available_names.update({tool.get("name") for tool in tools if isinstance(tool.get("name"), str)})
        compact_prompt = os.getenv("JARVIS_PROMPT_COMPACT", "true").lower() in {"1", "true", "yes"}
        include_antipatterns = os.getenv("JARVIS_PROMPT_INCLUDE_ANTIPATTERNS", "true").lower() in {"1", "true", "yes"}
        if compact_prompt:
            max_keywords = int(os.getenv("JARVIS_PROMPT_MAX_KEYWORDS", "8"))
            include_param_descriptions = os.getenv("JARVIS_PROMPT_INCLUDE_PARAM_DESCRIPTIONS", "false").lower() in {"1", "true", "yes"}
            trim_descriptions = os.getenv("JARVIS_PROMPT_TRIM_TOOL_DESCRIPTIONS", "true").lower() in {"1", "true", "yes"}
            max_desc_chars = int(os.getenv("JARVIS_PROMPT_MAX_TOOL_DESC_CHARS", "180"))
        else:
            max_keywords = 999
            include_param_descriptions = True
            trim_descriptions = False
            max_desc_chars = 10_000
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            if trim_descriptions and isinstance(description, str):
                first_sentence = description.split(".")[0].strip()
                if first_sentence:
                    description = first_sentence + "."
                if len(description) > max_desc_chars:
                    description = description[: max_desc_chars - 3].rstrip() + "..."
            parameters = func.get("parameters", {})
            antipatterns = tool.get("antipatterns", []) or []
            keywords = tool.get("keywords") or []
            
            # Format parameters
            props = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                
                param_str = f"  - {param_name} ({param_type})"
                if is_required:
                    param_str += " [REQUIRED]"
                if include_param_descriptions and param_desc:
                    param_str += f": {param_desc}"
                
                param_list.append(param_str)

            antipattern_block = ""
            if include_antipatterns:
                antipattern_lines = []
                for ap in antipatterns:
                    if not isinstance(ap, dict):
                        continue
                    ap_cmd = ap.get("command_name")
                    ap_desc = ap.get("description")
                    if ap_cmd and ap_cmd not in available_names:
                        continue
                    if isinstance(ap_desc, str) and ap_desc.strip():
                        antipattern_lines.append(f"  - {ap_desc.strip()}")

                if antipattern_lines:
                    antipattern_block = f"\nAnti-patterns:\n{chr(10).join(antipattern_lines)}"

            keywords_block = ""
            if isinstance(keywords, list):
                keyword_list = [kw for kw in keywords if isinstance(kw, str) and kw.strip()]
                if keyword_list:
                    keywords_block = f"\nKeywords: {', '.join(keyword_list[:max_keywords])}"

            example_str = ""
            if name in command_examples:
                examples = command_examples[name]
                if max_examples is not None:
                    examples = examples[:max_examples]
                example_lines = []
                for example in examples:
                    voice_cmd = example.get("voice_command", "")
                    if voice_cmd:
                        example_lines.append(f"  - \"{voice_cmd}\"")
                if example_lines:
                    example_str = "\nExamples:\n" + "\n".join(example_lines)

            tool_str = f"""
Tool: {name}
Description: {description}
{antipattern_block}
{keywords_block}
Parameters:
{chr(10).join(param_list) if param_list else "  None"}{example_str}
"""
            tool_descriptions.append(tool_str)
        
        return "\n".join(tool_descriptions)
    
    @staticmethod
    def format_tools_with_examples(tools: List[Dict[str, Any]], available_commands: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Format tools as text with primary examples included.
        
        Args:
            tools: List of tool definitions in OpenAI format
            available_commands: Optional list of CommandDefinition dicts (from cache) to get examples from
            
        Returns:
            Formatted string describing tools with primary examples
        """
        if not tools:
            return "No tools available."
        
        # Determine how many examples to include per command
        max_examples_env = os.getenv("JARVIS_EXAMPLES_PER_COMMAND", "").strip().lower()
        if not max_examples_env:
            max_examples = 3
        elif max_examples_env in {"all", "0", "-1"}:
            max_examples = None
        else:
            try:
                max_examples = max(1, int(max_examples_env))
            except ValueError:
                logger.warning("⚠️ Invalid JARVIS_EXAMPLES_PER_COMMAND=%r; defaulting to 3", max_examples_env)
                max_examples = 3

        # Build a map of command_name -> ordered example list
        command_examples = {}
        if available_commands:
            for cmd_dict in available_commands:
                cmd_name = cmd_dict.get("command_name")
                examples = cmd_dict.get("examples", [])
                if not cmd_name or not examples:
                    continue
                primary = [ex for ex in examples if ex.get("is_primary", False)]
                secondary = [ex for ex in examples if not ex.get("is_primary", False)]
                ordered = primary + secondary
                command_examples[cmd_name] = ordered
        
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", {})
            name = func.get("name", "unknown")
            description = func.get("description", "No description")
            parameters = func.get("parameters", {})
            
            # Format parameters
            props = parameters.get("properties", {})
            required = parameters.get("required", [])
            
            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                
                param_str = f"  - {param_name} ({param_type})"
                if is_required:
                    param_str += " [REQUIRED]"
                if param_desc:
                    param_str += f": {param_desc}"
                
                param_list.append(param_str)
            
            # Add examples if available
            example_str = ""
            if name in command_examples:
                examples = command_examples[name]
                if max_examples is not None:
                    examples = examples[:max_examples]
                example_lines = []
                for example in examples:
                    voice_cmd = example.get("voice_command", "")
                    if voice_cmd:
                        example_lines.append(f"  - \"{voice_cmd}\"")
                if example_lines:
                    example_str = "\nExamples:\n" + "\n".join(example_lines)
            
            tool_str = f"""
Tool: {name}
Description: {description}
Parameters:
{chr(10).join(param_list) if param_list else "  None"}{example_str}
"""
            tool_descriptions.append(tool_str)
        
        return "\n".join(tool_descriptions)


# Global parser instance
tool_call_parser = ToolCallParser()

