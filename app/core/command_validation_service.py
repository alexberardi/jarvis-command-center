"""
Command validation service for processing LLM command inference responses.
"""

import logging
from typing import List, Dict, Any, Tuple
from app.request_models.voice_command_request import CommandDefinition

logger = logging.getLogger("uvicorn")


class CommandValidationService:
    """Service for validating and processing commands from LLM inference responses."""
    
    def __init__(self):
        pass
    
    def validate_commands(
        self, 
        commands: List[Dict[str, Any]], 
        available_commands: List[CommandDefinition],
        conversation_id: str = None
    ) -> Tuple[List[CommandDefinition], List[Dict[str, Any]]]:
        """
        Validates commands and separates valid ones from broken ones.
        
        Args:
            commands: Raw commands from LLM inference response
            available_commands: Available commands from cache
            conversation_id: Conversation ID for logging
            
        Returns:
            Tuple of (valid_commands, broken_commands)
        """
        valid_commands = []
        broken_commands = []
        
        for i, command in enumerate(commands):
            command_name = command.get("n")
            if not command_name:
                # DO NOT DELETE - Broken command handling for future client-side support
                broken_command = {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {"type": "missing_command_name", "message": "Command name not found"}
                }
                broken_commands.append(broken_command)
                continue
            
            # Find the command definition
            selected_command = None
            for cmd in available_commands:
                if cmd.command_name == command_name:
                    selected_command = cmd
                    break
            
            if not selected_command:
                # DO NOT DELETE - Broken command handling for future client-side support
                broken_command = {
                    "success": False,
                    "command_name": command_name,
                    "parameters": None,
                    "errors": {"type": "unknown_command", "message": f"Command '{command_name}' not found"}
                }
                broken_commands.append(broken_command)
                continue
            
            # Command is valid, add to valid list
            valid_commands.append(selected_command)
        
        return valid_commands, broken_commands
