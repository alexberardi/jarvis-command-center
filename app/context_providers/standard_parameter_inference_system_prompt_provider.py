from app.core.interfaces.ijarvis_context_provider import IParameterInferenceSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition
from typing import Optional
from app.context_providers.general_prompt_provider import GeneralPromptProvider

class StandardParameterInferenceSystemPromptProvider(IParameterInferenceSystemPromptProvider):
    def __init__(self):
        self.general_provider = GeneralPromptProvider()
    
    @property
    def name(self) -> str:
        return "STANDARD"

    def build_parameter_inference(self, node_context: dict, selected_command: CommandDefinition, voice_command: Optional[str] = None) -> str:
        """Build complete parameter inference prompt with command details (implements interface)."""
        # Use the general prompt provider to build the base parameter inference prompt
        timezone = node_context.get('timezone', 'UTC')
        base_prompt = self.general_provider.build_parameter_inference_prompt(node_context, timezone, voice_command)
        
        # Add command details
        command_details = self.build_command_details_message(selected_command, voice_command)
        
        # Combine base prompt with command details
        complete_prompt = f"{base_prompt}\n\n{command_details}"
        
        return complete_prompt
    
    def build_base_parameter_inference_prompt(self, node_context: dict, conversation_id: str = None) -> str:
        """Build base system prompt for parameter inference (without command-specific details)."""
        # Use the general prompt provider to build the parameter inference prompt
        timezone = node_context.get('timezone', 'UTC')
        return self.general_provider.build_parameter_inference_prompt(node_context, timezone, conversation_id)

    def build_command_details_message(self, selected_command: CommandDefinition, voice_command: str) -> str:
        """Build the command details section for parameter inference."""
        import logging
        logger = logging.getLogger("uvicorn")
        
        # Get command details
        command_name = selected_command.command_name
        command_description = selected_command.description
        parameters = selected_command.parameters
        
        # Build parameter descriptions
        parameter_descriptions = []
        if parameters:
            for param_info in parameters:
                param_name = param_info.name
                param_type = param_info.type
                param_desc = param_info.description or 'No description available'
                param_required = param_info.required
                enum_values = param_info.enum_values or []
                
                # Build parameter line
                param_line = f"- {param_name} ({param_type})"
                if param_required:
                    param_line += " [REQUIRED]"
                param_line += f": {param_desc}"
                
                # Add enum values if available
                if enum_values:
                    param_line += f" [Allowed values: {', '.join(enum_values)}]"
                
                parameter_descriptions.append(param_line)
        
        # Build rules sections
        rules_str = ""
        if hasattr(selected_command, 'rules') and selected_command.rules:
            rules_str += f"\n\nRULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.rules)
        
        critical_rules_str = ""
        if hasattr(selected_command, 'critical_rules') and selected_command.critical_rules:
            critical_rules_str += f"\n\nCRITICAL RULES:\n" + "\n".join(f"- {rule}" for rule in selected_command.critical_rules)
        
        # Build examples
        example = selected_command.example if hasattr(selected_command, 'example') else None
        examples_str = ""
        if example:
            examples_str = f"\n\nEXAMPLES:\n{example}"
        
        # Build the complete command details message
        command_details = f"""COMMAND DETAILS:

Command: {command_name}
Description: {command_description}

PARAMETERS:
{chr(10).join(parameter_descriptions) if parameter_descriptions else "No parameters required"}{rules_str}{critical_rules_str}{examples_str}

RETURN ONLY VALID JSON

Voice Command: {voice_command}

NOTE: Date parameters (datetime, date, time, timedelta) will be extracted separately. Focus on extracting non-date parameters that you can confidently identify from the voice command."""
        
        logger.debug(f"Built command details for {command_name}: {len(command_details)} characters")
        return command_details
