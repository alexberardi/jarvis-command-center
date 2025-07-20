import os
from app.core.interfaces.ijarvis_context_provider import ISystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition
from app.context_providers.command_filter import CommandFilter
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class StandardSystemPromptProvider(ISystemPromptProvider):
    def __init__(self):
        self.command_filter = CommandFilter()
        self.preprocessing_enabled = os.getenv("COMMAND_PREPROCESSING_ENABLED", "false").lower() == "true"
    
    @property
    def name(self) -> str:
        return "STANDARD"

    def build_system_prompt(self, node_context: dict, available_commands: List[CommandDefinition], voice_command: Optional[str] = None) -> str:
        # Apply command filtering if enabled and voice_command is provided
        if self.preprocessing_enabled and voice_command:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, available_commands)
            logger.info(f"Command filtering: {stats['total_commands']} -> {stats['filtered_commands']} commands ({stats['reduction_percentage']:.1f}% reduction, ~{stats['tokens_saved']} tokens saved)")
            commands_to_use = filtered_commands
        else:
            commands_to_use = available_commands
        
        context_str = f"Node context: {node_context}"
        commands_str = '\n'.join(
            f"{cmd.command_name}({', '.join([f'{param.name}: {param.type}' for param in cmd.parameters])}) - {cmd.description}"
            for cmd in commands_to_use
        )
        
        # Use actual node context for examples
        current_room = node_context.get('room', '')
        
        # Create dynamic examples based on context
        if current_room:
            room_example = f"User: 'turn on the lights' → {{'success': true, 'command_name': 'turn_on_lights', 'parameters': {{'room': '{current_room}'}}, 'errors': null}}"
            missing_param_example = f"User: 'make it 72 degrees' → {{'success': true, 'command_name': 'set_temperature', 'parameters': {{'room': '{current_room}', 'degrees': 72}}, 'errors': null}}"
        else:
            room_example = f"User: 'turn on the lights' → {{'success': false, 'command_name': 'turn_on_lights', 'parameters': {{}}, 'errors': {{'type': 'missing_parameters', 'message': 'I need to know which room', 'missing_parameters': ['room'], 'clarification_question': 'Which room would you like to turn on the lights in?'}}}}"
            missing_param_example = f"User: 'set the temperature' → {{'success': false, 'command_name': 'set_temperature', 'parameters': {{}}, 'errors': {{'type': 'missing_parameters', 'message': 'I need to know the room and temperature', 'missing_parameters': ['room', 'degrees'], 'clarification_question': 'Which room and what temperature would you like to set?'}}}}"
        
        return (
            "You are an intent-filtering and parameter-extracting LLM for a smart home system. "
            "Given a user's voice command, your job is to match it to one of the available commands and extract the required parameters.\n\n"
            f"{context_str}\n\n"
            f"Available commands:\n{commands_str}\n\n"
            "Return your response as JSON with the following structure:\n\n"
            "For single or multiple commands, always return a 'commands' array:\n"
            '{"commands": [{"success": true, "command_name": "command_name", "parameters": {"param1": "value1"}, "errors": null}]}\n\n'
            "For multiple commands (e.g., 'turn on lights and play music'):\n"
            '{"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "bedroom"}, "errors": null}, {"success": true, "command_name": "play_music", "parameters": {"song": "jazz"}, "errors": null}]}\n\n'
            "For failures:\n"
            '{"commands": [{"success": false, "command_name": null, "parameters": null, "errors": {"type": "error_type", "message": "description", "missing_parameters": ["param1"], "suggestions": ["option1", "option2"], "clarification_question": "What would you like to do?"}}]}\n\n'
            "Error types:\n"
            "- no_command_match: Could not identify any matching command\n"
            "- ambiguous_command: Multiple commands could match\n"
            "- missing_parameters: Command identified but missing required parameters\n"
            "- invalid_parameters: Parameters provided but invalid type/format\n\n"
            "CRITICAL PARAMETER EXTRACTION RULES:\n"
            "1. ROOM CONTEXT: If the node context contains a room (room != ''), ALWAYS use that room for commands that need a room parameter, even if the user doesn't mention it explicitly\n"
            "2. TEMPERATURE: Extract numbers from phrases like 'make it 68 degrees', 'set to 72', 'make it warmer' (use reasonable defaults)\n"
            "3. TIMER: Parse time expressions like '5 minutes' → duration=5, unit='minutes'; '2 hours' → duration=2, unit='hours'\n"
            "4. MUSIC: Accept genre names like 'jazz', 'rock', 'classical' as valid song parameters\n"
            "5. NATURAL LANGUAGE: Be flexible with phrasing - 'lights on', 'turn on lights', 'could you turn on the lights' all mean the same thing\n\n"
            "Examples:\n"
            f"User: 'turn on the lights' → {{'commands': [{room_example.split(' → ')[1]}]}}\n"
            f"User: 'set the temperature' → {{'commands': [{missing_param_example.split(' → ')[1]}]}}\n"
            "User: 'set timer for 5 minutes' → {'commands': [{'success': true, 'command_name': 'set_timer', 'parameters': {'duration': 5, 'unit': 'minutes'}, 'errors': null}]}\n"
            "User: 'play some jazz' → {'commands': [{'success': true, 'command_name': 'play_music', 'parameters': {'song': 'jazz'}, 'errors': null}]}\n"
            "User: 'make it 68 degrees' (with room context) → {'commands': [{'success': true, 'command_name': 'set_temperature', 'parameters': {'room': 'bedroom', 'degrees': 68}, 'errors': null}]}\n"
            "User: 'turn on lights and play music' → {'commands': [{'success': true, 'command_name': 'turn_on_lights', 'parameters': {'room': 'bedroom'}, 'errors': null}, {'success': true, 'command_name': 'play_music', 'parameters': {'song': 'jazz'}, 'errors': null}]}\n"
            "User: 'turn on something' → {'commands': [{'success': false, 'command_name': null, 'parameters': null, 'errors': {'type': 'no_command_match', 'message': 'Could not identify what you want to turn on', 'clarification_question': 'What would you like to turn on?'}}]}\n\n"
            "Guidelines:\n"
            "- MOST IMPORTANT: If node context has room != '', use that room for any command requiring a room parameter\n"
            "- Extract numeric values from natural language (e.g., 'five' → 5, 'seventy-two' → 72)\n"
            "- Be generous with music requests - genres, artists, and song names are all valid\n"
            "- Parse time expressions flexibly (minutes, mins, hours, hrs, seconds, secs)\n"
            "- Always return valid JSON\n"
            "- Only ask for clarification when truly ambiguous or when room context is empty and room is required"
        ) 