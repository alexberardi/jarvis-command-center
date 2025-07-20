#!/usr/bin/env python3
"""
Test script to verify the COMMAND_PREPROCESSING_ENABLED environment variable works correctly
Compatible with pytest framework
"""
import os
import pytest
from unittest.mock import patch
from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter

def create_test_commands():
    """Create test commands with keywords"""
    return [
        CommandDefinition(
            command_name="turn_on_lights",
            description="Turn on lights in a specific room",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["light", "lights", "on", "turn on"]
        ),
        CommandDefinition(
            command_name="set_temperature",
            description="Set temperature in a room",
            parameters=[
                CommandParameter(name="room", type="str"),
                CommandParameter(name="degrees", type="int")
            ],
            keywords=["temperature", "temp", "degrees", "hot", "cold"]
        ),
        CommandDefinition(
            command_name="play_music",
            description="Play music",
            parameters=[CommandParameter(name="song", type="str")],
            keywords=["play", "music", "song"]
        ),
        CommandDefinition(
            command_name="unrelated_command",
            description="An unrelated command",
            parameters=[],
            keywords=["unrelated", "other", "different"]
        )
    ]

class TestPreprocessingToggle:
    """Test the COMMAND_PREPROCESSING_ENABLED environment variable functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.commands = create_test_commands()
        self.node_context = {"room": "bedroom", "node_id": "test-node"}
        self.voice_command = "turn on the lights"
    
    def test_preprocessing_disabled_by_default(self):
        """Test that preprocessing is disabled by default"""
        # Clear the environment variable to test default behavior
        with patch.dict(os.environ, {}, clear=True):
            provider = StandardSystemPromptProvider()
            assert provider.preprocessing_enabled == False
    
    def test_preprocessing_enabled_explicitly(self):
        """Test that preprocessing can be enabled via environment variable"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            assert provider.preprocessing_enabled == True
    
    def test_preprocessing_disabled_explicitly(self):
        """Test that preprocessing can be disabled via environment variable"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider = StandardSystemPromptProvider()
            assert provider.preprocessing_enabled == False
    
    def test_preprocessing_case_insensitive(self):
        """Test that environment variable is case insensitive"""
        test_cases = ["true", "TRUE", "True", "false", "FALSE", "False"]
        
        for value in test_cases:
            with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": value}):
                provider = StandardSystemPromptProvider()
                expected = value.lower() == "true"
                assert provider.preprocessing_enabled == expected, f"Failed for value: {value}"
    
    def test_preprocessing_affects_prompt_content(self):
        """Test that preprocessing actually affects the prompt content"""
        # Test with preprocessing disabled
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider_no_preprocessing = StandardSystemPromptProvider()
            prompt_no_preprocessing = provider_no_preprocessing.build_system_prompt(
                self.node_context, self.commands, self.voice_command
            )
        
        # Test with preprocessing enabled
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider_with_preprocessing = StandardSystemPromptProvider()
            prompt_with_preprocessing = provider_with_preprocessing.build_system_prompt(
                self.node_context, self.commands, self.voice_command
            )
        
        # Should include relevant commands in both cases
        assert "turn_on_lights" in prompt_no_preprocessing
        assert "turn_on_lights" in prompt_with_preprocessing
        
        # Should include all commands when preprocessing is disabled
        assert "unrelated_command" in prompt_no_preprocessing
        
        # Should exclude unrelated commands when preprocessing is enabled
        assert "unrelated_command" not in prompt_with_preprocessing
        
        # Preprocessing should reduce prompt size
        assert len(prompt_with_preprocessing) < len(prompt_no_preprocessing)
    
    def test_preprocessing_without_voice_command(self):
        """Test that preprocessing is skipped when no voice command is provided"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            # No voice command provided
            prompt = provider.build_system_prompt(self.node_context, self.commands, None)
            
            # Should include all commands when no voice command is provided
            assert "turn_on_lights" in prompt
            assert "unrelated_command" in prompt
    
    def test_preprocessing_with_different_voice_commands(self):
        """Test preprocessing with different voice commands"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            # Test with lights command
            lights_prompt = provider.build_system_prompt(
                self.node_context, self.commands, "turn on the lights"
            )
            assert "turn_on_lights" in lights_prompt
            assert "unrelated_command" not in lights_prompt
            
            # Test with music command
            music_prompt = provider.build_system_prompt(
                self.node_context, self.commands, "play some music"
            )
            assert "play_music" in music_prompt
            assert "unrelated_command" not in music_prompt
            
            # Test with unrelated command (should fallback)
            unrelated_prompt = provider.build_system_prompt(
                self.node_context, self.commands, "something completely different"
            )
            # Should include fallback commands or all commands
            assert len(unrelated_prompt) > 0
    
    def test_preprocessing_statistics_logging(self):
        """Test that preprocessing statistics are logged when enabled"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            with patch('app.context_providers.standard_system_prompt_provider.logger') as mock_logger:
                provider.build_system_prompt(self.node_context, self.commands, self.voice_command)
                
                # Should log filtering statistics
                mock_logger.info.assert_called_once()
                log_message = mock_logger.info.call_args[0][0]
                assert "Command filtering:" in log_message
                assert "reduction" in log_message
                assert "tokens saved" in log_message
    
    def test_preprocessing_no_logging_when_disabled(self):
        """Test that no preprocessing statistics are logged when disabled"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider = StandardSystemPromptProvider()
            
            with patch('app.context_providers.standard_system_prompt_provider.logger') as mock_logger:
                provider.build_system_prompt(self.node_context, self.commands, self.voice_command)
                
                # Should not log filtering statistics
                mock_logger.info.assert_not_called()

def test_preprocessing_toggle_manual():
    """Manual test function for standalone execution"""
    commands = create_test_commands()
    node_context = {"room": "bedroom", "node_id": "test-node"}
    voice_command = "turn on the lights"
    
    print("Testing COMMAND_PREPROCESSING_ENABLED toggle:")
    print("=" * 60)
    
    # Test with preprocessing disabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "false"
    provider_no_preprocessing = StandardSystemPromptProvider()
    
    prompt_no_preprocessing = provider_no_preprocessing.build_system_prompt(
        node_context, commands, voice_command
    )
    
    print(f"🚫 Preprocessing DISABLED:")
    print(f"   Prompt length: {len(prompt_no_preprocessing)} characters")
    print(f"   Commands in prompt: {prompt_no_preprocessing.count('command_name')}")
    print(f"   Contains 'turn_on_lights': {'turn_on_lights' in prompt_no_preprocessing}")
    print(f"   Contains 'unrelated_command': {'unrelated_command' in prompt_no_preprocessing}")
    
    # Test with preprocessing enabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    provider_with_preprocessing = StandardSystemPromptProvider()
    
    prompt_with_preprocessing = provider_with_preprocessing.build_system_prompt(
        node_context, commands, voice_command
    )
    
    print(f"\n✅ Preprocessing ENABLED:")
    print(f"   Prompt length: {len(prompt_with_preprocessing)} characters")
    print(f"   Commands in prompt: {prompt_with_preprocessing.count('command_name')}")
    print(f"   Contains 'turn_on_lights': {'turn_on_lights' in prompt_with_preprocessing}")
    print(f"   Contains 'unrelated_command': {'unrelated_command' in prompt_with_preprocessing}")
    
    # Compare results
    length_diff = len(prompt_no_preprocessing) - len(prompt_with_preprocessing)
    reduction_percent = (length_diff / len(prompt_no_preprocessing)) * 100 if len(prompt_no_preprocessing) > 0 else 0
    
    print(f"\n📊 Comparison:")
    print(f"   Length difference: {length_diff} characters")
    print(f"   Reduction: {reduction_percent:.1f}%")
    print(f"   Preprocessing working: {'✅ YES' if length_diff > 0 else '❌ NO'}")
    
    # Test with voice command that should match different commands
    print(f"\n" + "=" * 60)
    print("Testing with different voice command:")
    
    voice_command2 = "play some music"
    
    # Disabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "false"
    provider_no_preprocessing2 = StandardSystemPromptProvider()
    prompt_no_preprocessing2 = provider_no_preprocessing2.build_system_prompt(
        node_context, commands, voice_command2
    )
    
    # Enabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    provider_with_preprocessing2 = StandardSystemPromptProvider()
    prompt_with_preprocessing2 = provider_with_preprocessing2.build_system_prompt(
        node_context, commands, voice_command2
    )
    
    print(f"Voice command: '{voice_command2}'")
    print(f"Without preprocessing - contains 'play_music': {'play_music' in prompt_no_preprocessing2}")
    print(f"With preprocessing - contains 'play_music': {'play_music' in prompt_with_preprocessing2}")
    print(f"Without preprocessing - contains 'turn_on_lights': {'turn_on_lights' in prompt_no_preprocessing2}")
    print(f"With preprocessing - contains 'turn_on_lights': {'turn_on_lights' in prompt_with_preprocessing2}")
    
    # Test with no voice command provided (should not filter)
    print(f"\n" + "=" * 60)
    print("Testing with no voice command (should not filter):")
    
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    provider_no_voice = StandardSystemPromptProvider()
    prompt_no_voice = provider_no_voice.build_system_prompt(
        node_context, commands, None  # No voice command
    )
    
    print(f"No voice command provided:")
    print(f"   Prompt length: {len(prompt_no_voice)} characters")
    print(f"   Commands in prompt: {prompt_no_voice.count('command_name')}")
    print(f"   Should be same as disabled: {'✅ YES' if len(prompt_no_voice) == len(prompt_no_preprocessing) else '❌ NO'}")

if __name__ == "__main__":
    # Run manual test if executed directly
    test_preprocessing_toggle_manual()
    
    # Also run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except:
        print("\nPytest not available, manual tests completed.") 