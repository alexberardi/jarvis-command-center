from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition
from typing import List, Optional

class StandardCommandInferenceSystemPromptProvider(ICommandInferenceSystemPromptProvider):
    @property
    def name(self) -> str:
        return "STANDARD"

    def _filter_commands_by_context(self, available_commands: List[CommandDefinition], node_context: dict) -> List[CommandDefinition]:
        """Filter commands based on node context if needed."""
        # For now, return all commands - can be enhanced later for context filtering
        return available_commands

    def build_system_prompt(self, node_context: dict, available_commands: List[CommandDefinition], voice_command: Optional[str] = None) -> str:
        """Build system prompt for command inference."""
        import logging
        logger = logging.getLogger("uvicorn")
        
        # Filter commands based on node context
        commands_to_use = self._filter_commands_by_context(available_commands, node_context)
        
        logger.info(f"Building command inference prompt with {len(commands_to_use)} commands")
        
        if not commands_to_use:
            logger.warning("No commands available for this context")
            return "No commands available for this context."
        
        # Build commands string with descriptions and parameters
        commands_str = '\n'.join(
            f"- {cmd.command_name} - {cmd.description}\n" +
            (f"  Parameters: {', '.join([f'{p.name} ({p.type})' + ('*' if p.required else '') + (f' - enum: {p.enum_values}' if hasattr(p, 'enum_values') and p.enum_values else '') for p in cmd.parameters])}" if cmd.parameters else "  Parameters: None") +
            (f"\n  Example: {cmd.example}" if hasattr(cmd, 'example') and cmd.example else "")
            for cmd in commands_to_use
        )
        
        # Debug logging for enum values
        for cmd in commands_to_use:
            for param in cmd.parameters:
                if hasattr(param, 'enum_values') and param.enum_values:
                    logger.info(f"Found enum values for {cmd.command_name}.{param.name}: {param.enum_values}")
                else:
                    logger.info(f"No enum values for {cmd.command_name}.{param.name}")
        
        logger.info(f"Commands string length: {len(commands_str)}")
        
        system_prompt = f"""User said: "{voice_command}" """
        
        logger.info(f"Command inference prompt built, length: {len(system_prompt)}")
        return system_prompt 