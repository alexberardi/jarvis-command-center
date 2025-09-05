import pytest
import os
from unittest.mock import patch
from app.context_providers.standard_command_inference_prompt_provider import StandardCommandInferenceSystemPromptProvider
from app.context_providers.custom.dummy_custom_system_prompt_provider import DummyCustomSystemPromptProvider
from app.deps import get_command_inference_system_prompt_provider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestStandardCommandInferenceSystemPromptProvider:
    def test_name_property(self):
        """Test that the name property returns 'STANDARD'"""
        provider = StandardCommandInferenceSystemPromptProvider()
        assert provider.name == "STANDARD"
    
    def test_build_system_prompt_structure(self):
        """Test that build_system_prompt returns a properly structured prompt"""
        provider = StandardCommandInferenceSystemPromptProvider()
        
        # Mock data
        node_context = {"room": "kitchen", "node_id": "node-123"}
        commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")]
            ),
            CommandDefinition(
                command_name="set_temperature",
                description="Set temperature in a room",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="degrees", type="int")
                ]
            )
        ]
        
        result = provider.build_system_prompt(node_context, commands)
        
        # Check that essential components are present
        assert "You are an LLM that identifies smart home commands" in result
        assert "Node context: {'room': 'kitchen', 'node_id': 'node-123'}" in result
        assert "Available Commands:" in result
        assert "turn_on_lights(room: str) - Turn on lights in a specific room" in result
        assert "set_temperature(room: str, degrees: int) - Set temperature in a room" in result
        assert "JSON Response Format" in result
        assert "s" in result  # compressed format
        assert "n" in result  # compressed format
        assert "p" in result  # compressed format
        assert "e" in result  # compressed format
        
    def test_build_system_prompt_with_room_context(self):
        """Test that examples use the actual room context when room is present"""
        provider = StandardCommandInferenceSystemPromptProvider()
        
        node_context = {"room": "bedroom", "node_id": "node-456"}
        commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights",
                parameters=[CommandParameter(name="room", type="str")]
            )
        ]
        
        result = provider.build_system_prompt(node_context, commands)
        
        # Should use "bedroom" in examples when room is present
        assert "'room': 'bedroom'" in result
        assert "'s': true" in result  # compressed format
        
    def test_build_system_prompt_without_room_context(self):
        """Test that examples show missing parameter errors when room is empty"""
        provider = StandardCommandInferenceSystemPromptProvider()
        
        node_context = {"room": "", "node_id": "node-789"}
        commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights",
                parameters=[CommandParameter(name="room", type="str")]
            )
        ]
        
        result = provider.build_system_prompt(node_context, commands)
        
        # Should show failure examples when room is empty
        assert "'s': false" in result  # compressed format
        assert "missing_parameters" in result
        assert "clarification_question" in result


class TestGetSystemPromptProvider:
    @patch.dict(os.environ, {}, clear=True)
    def test_get_command_inference_system_prompt_provider_default(self):
        """Test that get_command_inference_system_prompt_provider returns StandardCommandInferenceSystemPromptProvider when env var is not set"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, StandardCommandInferenceSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "STANDARD"})
    def test_get_command_inference_system_prompt_provider_standard(self):
        """Test that get_command_inference_system_prompt_provider returns StandardCommandInferenceSystemPromptProvider when env var is 'STANDARD'"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, StandardCommandInferenceSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "standard"})
    def test_get_command_inference_system_prompt_provider_standard_lowercase(self):
        """Test that get_command_inference_system_prompt_provider handles case-insensitive matching"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, StandardCommandInferenceSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "CUSTOM"})
    def test_get_command_inference_system_prompt_provider_custom(self):
        """Test that get_command_inference_system_prompt_provider returns DummyCustomSystemPromptProvider when env var is 'CUSTOM'"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, DummyCustomSystemPromptProvider)
        assert provider.name == "CUSTOM"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "custom"})
    def test_get_command_inference_system_prompt_provider_custom_lowercase(self):
        """Test that get_command_inference_system_prompt_provider handles case-insensitive matching for custom provider"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, DummyCustomSystemPromptProvider)
        assert provider.name == "CUSTOM"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "NONEXISTENT"})
    def test_get_command_inference_system_prompt_provider_fallback(self):
        """Test that get_command_inference_system_prompt_provider falls back to StandardCommandInferenceSystemPromptProvider for unknown providers"""
        provider = get_command_inference_system_prompt_provider()
        assert isinstance(provider, StandardCommandInferenceSystemPromptProvider)
        assert provider.name == "STANDARD"


if __name__ == "__main__":
    pytest.main([__file__]) 