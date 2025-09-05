from app.core.general_context import get_general_context
from typing import Dict, List
import logging

logger = logging.getLogger("uvicorn")

class GeneralPromptProvider:
    """Provider for shared prompt information used by both command and parameter inference."""
    
    @property
    def name(self) -> str:
        return "GENERAL"
    
    def build_shared_prompt_base(self, node_context: Dict[str, str], timezone: str = None, conversation_id: str = None) -> str:
        """Build the shared base prompt containing general context, node context, and persona."""
        general_context_str = get_general_context(timezone)
        
        # Build node context string
        node_context_str = '\n'.join(f"{k}: {v}" for k, v in node_context.items()) if node_context else "No specific context"
        
        # Add conversation ID at the beginning if provided
        conversation_prefix = f"Conversation ID: {conversation_id}\n\n" if conversation_id else ""
        
        shared_base = f"""{conversation_prefix}You are an inference engine for a voice assistant. Ultimately, your job is to match a command name and parameters from a voice command to a set of available commands. This prompt establishes the base context and rules.

{general_context_str}

Node Context: {node_context_str}"""
        
        return shared_base
    
    def build_command_inference_prompt(self, node_context: Dict[str, str], timezone: str = None, available_commands: List = None, conversation_id: str = None) -> str:
        """Build the complete command inference prompt using the shared base."""
        shared_base = self.build_shared_prompt_base(node_context, timezone, conversation_id)
        
        # Build available commands string
        available_commands_str = "No commands available"
        if available_commands:
            available_commands_str = '\n'.join(
                f"- {cmd.command_name} - {cmd.description}"
                for cmd in available_commands
            )
        
        command_prompt = f"""{shared_base}

TASK_COMMAND_INFERENCE: You are performing COMMAND INFERENCE. Identify which command the user wants to execute.

GENERAL RULES FOR COMMAND INFERENCE:
- Use room from context if available to help you identify the command
- Only extract the command name, do not attempt to extract parameters
- Choose only from the available commands listed below, if there is no match, return the command name as null and "e": {{"type": "no_command_match"}}

AVAILABLE COMMANDS:
{available_commands_str}


GENERAL JSON OUTPUT FORMAT RULES:
- Always use double quotes for property names and string values
- The "s" field indicates success (true) or failure (false)
- The "e" field is null for success, error object for failures {{ "type": "error_type", "message": "error message"}}
- The "n" field contains the command name (must match exactly from available commands)
- The "p" object contains parameters (empty {{}} for command inference, extracted parameters for parameter inference)

This warm-up establishes the context, rules, and available commands. 

WHEN YOU RECEIVE A USER COMMAND:
1. Analyze the voice command to understand what the user wants
2. Match it to one of the available commands listed above
3. Return the command name in the exact format specified below

REQUIRED OUTPUT FORMAT FOR COMMAND INFERENCE:
{{"s":true,"n":"command_name","p":{{}},"e":null}}

You will now receive a user command and must perform command inference only."""
        
        return command_prompt
    
    def build_parameter_inference_prompt(self, node_context: Dict[str, str], timezone: str = None, conversation_id: str = None) -> str:
        """Build the base parameter inference prompt using the shared base."""
        shared_base = self.build_shared_prompt_base(node_context, timezone, conversation_id)
        
        parameter_prompt = f"""TASK_PARAM_EXTRACTION: You are performing PARAMETER INFERENCE. Extract parameters from the voice command.

REQUIRED OUTPUT FORMAT FOR PARAMETER INFERENCE:
{{"s":true,"p":{{...}},"e":null}}

Return ONLY valid JSON. NO explanations, descriptions, or additional text.

JSON OUTPUT RULES:
- The "p" object contains the extracted parameters
- The "s" field indicates success (true) or failure (false)
- The "e" field is null for success, error object for failures
- Empty "p": {{}} is perfectly fine if no parameters are found

PARAMETER EXTRACTION RULES:
- Extract parameters that can be clearly inferred from the voice command
- Use ONLY parameter names listed in the Command Details
- ALWAYS use rules specified in the parameter descriptions when they exist
- Date parameters are handled separately - do NOT extract them here
- If a parameter has enum_values, use EXACTLY one of those values

EXAMPLE OUTPUT:
{{"s":true,"p":{{"color":"red","size":"large"}},"e":null}}

"""
        
        return parameter_prompt
    
    def build_complete_static_prompt(self, node_context: Dict[str, str], timezone: str = None, available_commands: List = None, conversation_id: str = None) -> str:
        """Build a complete static prompt containing all rules, commands, and examples that can be cached."""
        shared_base = self.build_shared_prompt_base(node_context, timezone, conversation_id)
        
        # Build available commands string
        available_commands_str = "No commands available"
        if available_commands:
            available_commands_str = '\n'.join(
                f"- {cmd.command_name} - {cmd.description}"
                for cmd in available_commands
            )
        

        
        complete_prompt = f"""{shared_base}

You are a voice assistant command classifier. Identify which command the user wants.

COMMANDS & EXAMPLES:

{available_commands_str}

Always respond with JSON: {{"s":true,"n":"command_name","e":null}}"""
        
        return complete_prompt
