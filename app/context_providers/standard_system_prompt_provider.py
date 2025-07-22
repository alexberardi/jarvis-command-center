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
            "You are an LLM that identifies smart home commands and extracts parameters from user voice input.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Available Commands:\n{commands_str}\n\n"
            "🚨 CRITICAL: You MUST respond with ONLY valid JSON. NO additional text, explanations, comments, or markdown.\n"
            "🚨 CRITICAL: Your entire response must be a single JSON object starting with '{' and ending with '}'.\n"
            "🚨 CRITICAL: Do NOT include any text before or after the JSON.\n"
            "🚨 CRITICAL: Do NOT use markdown code blocks or formatting.\n"
            "🚨 CRITICAL: Do NOT add explanations, clarifications, or additional context.\n\n"
            "JSON Response Format (ONLY this format):\n"
            '{"c": ['
            '{"s": true, "n": "name", "p": {...}, "e": null},'
            '...'
            ']}\n\n'
            "Error Response Format (ONLY this format):\n"
            '{"s": false, "n": null, "p": null, "e": {'
            '"type": "error_type", "message": "desc", "missing_parameters": [...], '
            '"suggestions": [...], "clarification_question": "..."}}'
            "\n\nError types:\n"
            "- no_command_match\n"
            "- ambiguous_command\n"
            "- missing_parameters\n"
            "- invalid_parameters\n\n"
            "Extraction Rules:\n"
            "1. Use room from node context if provided and needed\n"
            "2. '68 degrees' → degrees: 68\n"
            "3. '5 minutes' → duration: 5, unit: 'minutes'\n"
            "4. Music: Allow genres like 'jazz', 'rock', 'classical'\n"
            "5. Normalize phrasing: 'turn on lights', 'lights on', etc.\n"
            "6. Convert words to numbers: 'seventy two' → 72\n"
            "7. Time units: mins, minutes, hrs, hours, secs, seconds\n"
            "8. If no room context and room is required, return missing_parameters error\n"
            "9. For ambiguous commands, return ambiguous_command error\n"
            "10. For incomplete commands, return missing_parameters error\n"
            "11. Parameters must be objects, never null (use {} for empty parameters)\n"
            "12. All boolean values must be true/false, never null\n\n"
            "Key Legend: c=commands, s=success, n=command_name, p=parameters, e=errors\n\n"
            "VALID JSON EXAMPLES (copy these exact formats):\n"
            f"User: 'turn on the lights' → {{'c': [{{'s': true, 'n': 'turn_on_lights', 'p': {{'room': '{current_room}'}}, 'e': null}}]}}\n"
            f"User: 'set the temperature' → {{'c': [{{'s': false, 'n': 'set_temperature', 'p': {{}}, 'e': {{'type': 'missing_parameters', 'message': 'I need to know the room and temperature', 'missing_parameters': ['room', 'degrees'], 'clarification_question': 'Which room and what temperature would you like to set?'}}}}]}}\n"
            "User: 'set timer for 5 minutes' → {'c': [{'s': true, 'n': 'set_timer', 'p': {'duration': 5, 'unit': 'minutes'}, 'e': null}]}\n"
            "User: 'play some jazz' → {'c': [{'s': true, 'n': 'play_music', 'p': {'song': 'jazz'}, 'e': null}]}\n"
            "User: 'make it 68 degrees' (context=bedroom) → {'c': [{'s': true, 'n': 'set_temperature', 'p': {'room': 'bedroom', 'degrees': 68}, 'e': null}]}\n"
            "User: 'turn on lights and play music' → {'c': [{'s': true, 'n': 'turn_on_lights', 'p': {'room': 'bedroom'}, 'e': null}, {'s': true, 'n': 'play_music', 'p': {'song': 'jazz'}, 'e': null}]}\n"
            "User: 'turn on something' → {'c': [{'s': false, 'n': null, 'p': {}, 'e': {'type': 'no_command_match', 'message': 'Could not identify what you want to turn on', 'clarification_question': 'What would you like to turn on?'}}]}\n"
            "User: 'set temperature' (no room context) → {'c': [{'s': false, 'n': 'set_temperature', 'p': {}, 'e': {'type': 'missing_parameters', 'message': 'Room context is required', 'missing_parameters': ['room'], 'clarification_question': 'Which room would you like to set the temperature for?'}}]}\n\n"
            "🚨 FINAL REMINDER: Respond with ONLY the JSON object. Nothing else.\n"
        ) 