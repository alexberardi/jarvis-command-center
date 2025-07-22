from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition

class DummyCustomSystemPromptProvider(ICommandInferenceSystemPromptProvider):
    @property
    def name(self) -> str:
        return "CUSTOM"

    def build_system_prompt(self, node_context: dict, available_commands: list[CommandDefinition]) -> str:
        return (
            "[CUSTOM PROVIDER] This is a dummy custom system prompt provider. "
            f"Node context: {node_context}\nAvailable commands: {available_commands}"
        ) 