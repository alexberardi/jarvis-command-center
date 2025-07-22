from typing import Protocol, Dict, Optional
from abc import ABC, abstractmethod

from app.request_models.voice_command_request import CommandDefinition

class IJarvisContextProvider(Protocol):
    key: str # key for the cache dictionary
    def get_context(self, user_text: str, context_map: Dict[str, str]) -> str:
        ...

class ICommandInferenceSystemPromptProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_system_prompt(self, node_context: Dict[str, str], available_commands: list[CommandDefinition], voice_command: Optional[str] = None) -> str:
        ...


class ITranscriptionCleanupSystemPromptProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def build_system_prompt(self, voice_command: str) -> str:
        ...

