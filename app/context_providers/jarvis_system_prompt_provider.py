from typing import List, Optional
import os
from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition
from app.context_providers.command_filter import CommandFilter
from typing import List, Optional

class JarvisCommandInferenceSystemPromptProvider(ICommandInferenceSystemPromptProvider):
    def __init__(self):
        self.command_filter = CommandFilter()
        self.preprocessing_enabled = os.getenv("COMMAND_PREPROCESSING_ENABLED", "false").lower() == "true"
    
    @property
    def name(self) -> str:
        return "JARVIS"

    def build_system_prompt(self, node_context: dict, available_commands: List[CommandDefinition], voice_command: Optional[str] = None) -> str:
        if self.preprocessing_enabled and voice_command:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, available_commands)
            commands_to_use = filtered_commands
        else:
            commands_to_use = available_commands
        
        context_str = f"{node_context}"
        commands_str = '\n'.join(f"{cmd.command_name}({', '.join([p.name for p in cmd.parameters])})" for cmd in commands_to_use)
        current_room = node_context.get('room', '')

        return (
            "Extract smart home commands and parameters. Respond ONLY with valid JSON.\n\n"
            f"Context: {context_str}\n\n"
            f"Available Commands:\n{commands_str}\n\n"
            "Rules:\n"
            "- Return {'c':[...]} with command objects: {'s':bool,'n':str,'p':{},'e':null or {}}\n"
            "- Use room from context if needed\n"
            "- If info missing, return error: {'type':'missing_parameters','missing_parameters':[...],'clarification_question':'...'}\n\n"
            "Examples:\n"
            f"User: 'turn on lights' → {{'c':[{{'s':true,'n':'turn_on_lights','p':{{'room':'{current_room}'}},'e':null}}]}}\n"
            f"User: 'set temperature' → {{'c':[{{'s':false,'n':'set_temperature','p':{{}},'e':{{'type':'missing_parameters','missing_parameters':['room','degrees'],'clarification_question':'Which room and what temperature?'}}}}]}}\n"
            "User: 'turn on lights and play music' → {'c':[{'s':true,'n':'turn_on_lights','p':{'room':'bedroom'},'e':null},{'s':true,'n':'play_music','p':{'song':'jazz'},'e':null}]}\n\n"
            "Output only JSON. No text, no markdown."
        )