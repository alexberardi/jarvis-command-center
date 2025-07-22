from app.core.interfaces.ijarvis_context_provider import ITranscriptionCleanupSystemPromptProvider

class NoOpTranscriptionCleanupProvider(ITranscriptionCleanupSystemPromptProvider):
    @property
    def name(self) -> str:
        return "NO_OP"

    def build_system_prompt(self, voice_command: str) -> str:
        # This provider is used when transcription cleanup is disabled
        # It simply returns the original command without any processing
        return voice_command 