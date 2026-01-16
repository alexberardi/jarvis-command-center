"""
Tool Call Parser for LLM Responses.

Parses tool calls from LLM JSON responses for models that don't have
native tool calling support (e.g., local Llama models via vLLM/llama.cpp).
"""

import json
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
        
        Expected LLM output format:
        {
            "message": "Response to the user",
            "tool_call": {"name": "tool_name", "arguments": {"param": "value"}}
        }
        
        Or for no tool calls:
        {
            "message": "Response to the user",
            "tool_call": null
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
            
            # Extract tool call (singular - one at a time)
            tool_call_raw = parsed.get("tool_call")
            
            if tool_call_raw and tool_call_raw is not None:
                # Convert to internal format with ID
                # Generate unique tool call ID
                tool_call_id = f"call_{uuid.uuid4().hex[:12]}"
                
                # Get arguments (may already be dict or need parsing)
                arguments = tool_call_raw.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse tool arguments: {arguments}")
                
                formatted_call = {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_call_raw.get("name", "unknown"),
                        "arguments": json.dumps(arguments)  # Store as JSON string
                    }
                }
                
                logger.info(f"✅ Parsed 1 tool call from LLM response: {tool_call_raw.get('name', 'unknown')}")
                return "tool_calls", [formatted_call], message
            
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
        # Look for JSON between curly braces
        start_idx = text.find('{')
        end_idx = text.rfind('}')
        
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            potential_json = text[start_idx:end_idx + 1]
            # Strip comments before trying to parse
            cleaned_json = ToolCallParser._strip_json_comments(potential_json)
            try:
                # Validate it's actually JSON
                json.loads(cleaned_json)
                return cleaned_json
            except json.JSONDecodeError:
                pass
        
        return None
    
    @staticmethod
    def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
        """
        Format tools as text for inclusion in system prompt.
        
        Args:
            tools: List of tool definitions in OpenAI format
            
        Returns:
            Formatted string describing tools
        """
        if not tools:
            return "No tools available."
        
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
            
            tool_str = f"""
Tool: {name}
Description: {description}
Parameters:
{chr(10).join(param_list) if param_list else "  None"}
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
        
        # Build a map of command_name -> primary example for quick lookup
        command_examples = {}
        if available_commands:
            for cmd_dict in available_commands:
                cmd_name = cmd_dict.get("command_name")
                examples = cmd_dict.get("examples", [])
                if examples:
                    # Find primary example
                    for example in examples:
                        if example.get("is_primary", False):
                            command_examples[cmd_name] = example
                            break
        
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
            
            # Add primary example if available
            example_str = ""
            if name in command_examples:
                example = command_examples[name]
                voice_cmd = example.get("voice_command", "")
                example_str = f"\n  Example: \"{voice_cmd}\""
            
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

