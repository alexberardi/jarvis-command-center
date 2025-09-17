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

    def _format_keywords_for_command(self, cmd: CommandDefinition) -> str:
        """Format keywords as flexible hints for command detection."""
        if hasattr(cmd, 'keywords') and cmd.keywords:
            keywords_str = ', '.join(cmd.keywords)
            return f"\n  Keywords (hints): {keywords_str}"
        return ""

    def _format_examples_for_command(self, cmd: CommandDefinition) -> str:
        """Format examples for a command using structured format. Only show the first primary example."""
        if hasattr(cmd, 'examples') and cmd.examples:
            import json
            # Find the first example where is_primary is True
            primary_example = None
            for example in cmd.examples:
                if example.is_primary:
                    primary_example = example
                    break
            
            if primary_example:
                # Format the expected parameters as JSON output format
                params_json = json.dumps(primary_example.expected_parameters) if primary_example.expected_parameters else "{}"
                return f'\n  Example: "{primary_example.voice_command}" → {{"n": "{cmd.command_name}", "p": {params_json}}}'
        
        return ""

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
        
        # Build commands string with descriptions, parameters, keywords, and examples
        commands_str = '\n'.join(
            f"- {cmd.command_name} - {cmd.description}\n" +
            (f"  Parameters: {', '.join([f'{p.name} ({p.type})' + ('*' if p.required else '') + (f' - enum: {p.enum_values}' if hasattr(p, 'enum_values') and p.enum_values else '') for p in cmd.parameters])}" if cmd.parameters else "  Parameters: None") +
            self._format_keywords_for_command(cmd) +
            self._format_examples_for_command(cmd)
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
        
        system_prompt = f"""You are a voice assistant command classifier. Analyze the user's voice command and identify which command they want to execute.

AVAILABLE COMMANDS:
{commands_str}

INSTRUCTIONS:
- Keywords are provided as hints to help identify commands, but the user doesn't need to use these exact words
- Look for the intent and meaning behind the user's words, not just keyword matches
- Keywords help you understand what concepts each command handles
- Choose the command that best matches the user's intent

User said: "{voice_command}"

Respond with JSON: {{"s": true, "n": "command_name", "e": null}}"""
        
        logger.info(f"Command inference prompt built, length: {len(system_prompt)}")
        return system_prompt 