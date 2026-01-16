"""
Get Command Examples Tool.

Retrieves example utterances for one or more commands to help the LLM
make better command selection decisions.
"""

import logging
from typing import Dict, Any, Optional, List

from app.core.interfaces.iserver_tool import IServerTool
from app.core.conversation_cache import conversation_cache
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")


class GetCommandExamplesTool(IServerTool):
    """Tool for retrieving example utterances for commands."""
    
    @property
    def name(self) -> str:
        return "get_command_utterance_examples"
    
    @property
    def description(self) -> str:
        return (
            "Retrieve an extensive list of example voice commands for a command to confirm it is the correct choice."
            "This tool is also useful to inform your decision on how parameters should be extracted."
        )
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of CLIENT command/tool names to get examples for Can be a single command or multiple commands to compare."
                }
            },
            "required": ["command_names"]
        }
    
    def execute(
        self,
        command_names: List[str],
        conversation_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute the command examples retrieval.
        
        Args:
            command_names: List of command names to get examples for
            conversation_id: Conversation ID (passed by tool executor)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with examples for each requested command
        """
        logger.info(f"      📚 get_command_examples called for {len(command_names)} command(s): {command_names}")
        
        # Check for conversation_id
        if not conversation_id:
            logger.warning(f"      └─ ⚠️  Missing conversation_id")
            return {
                "error": "missing_conversation_id",
                "message": "Conversation ID is required but was not provided"
            }
        
        # Get available commands from cache
        available_commands = conversation_cache.get_available_commands(conversation_id)
        
        # Check if None (conversation not found) vs empty list (no commands provided)
        if available_commands is None:
            logger.warning(f"      └─ ⚠️  Conversation not found in cache for {conversation_id[:8]}")
            return {
                "error": "conversation_not_found",
                "message": f"Conversation {conversation_id[:8]} not found or expired"
            }
        
        if len(available_commands) == 0:
            logger.warning(f"      └─ ⚠️  No available commands stored for conversation {conversation_id[:8]}")
            return {
                "error": "no_commands_available",
                "message": f"Conversation {conversation_id[:8]} exists but no commands were provided during warmup"
            }
        
        logger.info(f"      └─ Found {len(available_commands)} available command(s) in cache")
        
        # Build result for each requested command
        results = []
        found_count = 0
        examples_count = 0
        
        for cmd_name in command_names:
            # Find matching command
            matching_cmd = None
            for cmd_dict in available_commands:
                if cmd_dict.get("command_name") == cmd_name:
                    matching_cmd = cmd_dict
                    break
            
            if not matching_cmd:
                logger.debug(f"      └─ Command '{cmd_name}' not found")
                results.append({
                    "command_name": cmd_name,
                    "examples": []
                })
                continue
            
            found_count += 1
            examples = matching_cmd.get("examples", [])
            
            # Convert examples to proper format
            example_list = []
            for example in examples:
                example_list.append({
                    "voice_command": example.get("voice_command", ""),
                    "expected_parameters": example.get("expected_parameters", {}),
                    "is_primary": example.get("is_primary", False)
                })
                examples_count += 1
            
            logger.info(f"      └─ Command '{cmd_name}': found {len(example_list)} example(s)")
            results.append({
                "command_name": cmd_name,
                "examples": example_list
            })
        
        logger.info(f"      └─ ✅ Returning examples: {found_count}/{len(command_names)} commands found, {examples_count} total examples")
        
        result = {
            "commands": results
        }
        
        # Log full result before returning
        import json
        result_json = json.dumps(result, indent=2)
        logger.info(f"      📤 Returning to LLM (first 2000 chars):\n{result_json[:2000]}")
        if len(result_json) > 2000:
            logger.info(f"      ... (truncated, total length: {len(result_json)} chars)")
        
        return result

