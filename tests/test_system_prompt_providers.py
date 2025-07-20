import pytest
import os
from unittest.mock import patch
from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider
from app.context_providers.custom.dummy_custom_system_prompt_provider import DummyCustomSystemPromptProvider
from app.deps import get_system_prompt_provider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestStandardSystemPromptProvider:
    def test_name_property(self):
        """Test that the name property returns 'STANDARD'"""
        provider = StandardSystemPromptProvider()
        assert provider.name == "STANDARD"
    
    def test_build_system_prompt_structure(self):
        """Test that build_system_prompt returns a properly structured prompt"""
        provider = StandardSystemPromptProvider()
        
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
        assert "You are an intent-filtering and parameter-extracting LLM" in result
        assert "Node context: {'room': 'kitchen', 'node_id': 'node-123'}" in result
        assert "Available commands:" in result
        assert "turn_on_lights(room: str) - Turn on lights in a specific room" in result
        assert "set_temperature(room: str, degrees: int) - Set temperature in a room" in result
        assert "Return your response as JSON" in result
        assert "success" in result
        assert "command_name" in result
        assert "parameters" in result
        assert "errors" in result
        
    def test_build_system_prompt_with_room_context(self):
        """Test that examples use the actual room context when room is present"""
        provider = StandardSystemPromptProvider()
        
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
        assert "success': true" in result
        
    def test_build_system_prompt_without_room_context(self):
        """Test that examples show missing parameter errors when room is empty"""
        provider = StandardSystemPromptProvider()
        
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
        assert "success': false" in result
        assert "missing_parameters" in result
        assert "clarification_question" in result


class TestGetSystemPromptProvider:
    @patch.dict(os.environ, {}, clear=True)
    def test_get_system_prompt_provider_default(self):
        """Test that get_system_prompt_provider returns StandardSystemPromptProvider when env var is not set"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, StandardSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "STANDARD"})
    def test_get_system_prompt_provider_standard(self):
        """Test that get_system_prompt_provider returns StandardSystemPromptProvider when env var is 'STANDARD'"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, StandardSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "standard"})
    def test_get_system_prompt_provider_standard_lowercase(self):
        """Test that get_system_prompt_provider handles case-insensitive matching"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, StandardSystemPromptProvider)
        assert provider.name == "STANDARD"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "CUSTOM"})
    def test_get_system_prompt_provider_custom(self):
        """Test that get_system_prompt_provider returns DummyCustomSystemPromptProvider when env var is 'CUSTOM'"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, DummyCustomSystemPromptProvider)
        assert provider.name == "CUSTOM"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "custom"})
    def test_get_system_prompt_provider_custom_lowercase(self):
        """Test that get_system_prompt_provider handles case-insensitive matching for custom provider"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, DummyCustomSystemPromptProvider)
        assert provider.name == "CUSTOM"
    
    @patch.dict(os.environ, {"JARVIS_SYSTEM_PROMPT_PROVIDER": "NONEXISTENT"})
    def test_get_system_prompt_provider_fallback(self):
        """Test that get_system_prompt_provider falls back to StandardSystemPromptProvider for unknown providers"""
        provider = get_system_prompt_provider()
        assert isinstance(provider, StandardSystemPromptProvider)
        assert provider.name == "STANDARD"


if __name__ == "__main__":
    pytest.main([__file__]) 