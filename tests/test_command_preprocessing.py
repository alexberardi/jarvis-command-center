"""
Tests for command preprocessing functionality including keyword-based filtering
"""
import os
import pytest
from unittest.mock import patch

from app.context_providers.command_filter import CommandFilter
from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestCommandFilter:
    """Test the CommandFilter class functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.command_filter = CommandFilter()
        self.sample_commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["light", "lights", "on", "turn on", "switch on", "illuminate", "brighten"]
            ),
            CommandDefinition(
                command_name="turn_off_lights",
                description="Turn off lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["light", "lights", "off", "turn off", "switch off", "dim", "darken"]
            ),
            CommandDefinition(
                command_name="set_temperature",
                description="Set temperature in a room",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="degrees", type="int")
                ],
                keywords=["temperature", "temp", "degrees", "hot", "cold", "warm", "cool", "heat", "heating", "cooling", "thermostat"]
            ),
            CommandDefinition(
                command_name="play_music",
                description="Play music",
                parameters=[CommandParameter(name="song", type="str")],
                keywords=["play", "music", "song", "track", "artist", "album", "jazz", "rock", "classical", "pop", "country"]
            ),
            CommandDefinition(
                command_name="stop_music",
                description="Stop playing music",
                parameters=[],
                keywords=["stop", "pause", "music", "song", "quiet", "silence"]
            ),
            CommandDefinition(
                command_name="set_timer",
                description="Set a timer",
                parameters=[
                    CommandParameter(name="duration", type="int"),
                    CommandParameter(name="unit", type="str")
                ],
                keywords=["timer", "alarm", "remind", "minutes", "hours", "seconds", "time", "countdown"]
            ),
            CommandDefinition(
                command_name="start_vacuum",
                description="Start vacuum cleaner",
                parameters=[],
                keywords=["vacuum", "clean", "start", "roomba"]
            ),
            CommandDefinition(
                command_name="no_keywords_command",
                description="A command with no keywords defined",
                parameters=[],
                keywords=None
            )
        ]

    def test_exact_keyword_matching(self):
        """Test that exact keyword matching works correctly"""
        voice_command = "turn on the lights"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert "turn_on_lights" in command_names
        assert "turn_off_lights" in command_names  # Also matches "lights"
        assert stats["total_commands"] == 8
        assert stats["filtered_commands"] <= 8
        assert stats["reduction_percentage"] > 0

    def test_single_command_filtering(self):
        """Test filtering for commands with unique keywords"""
        voice_command = "set timer for 5 minutes"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert "set_timer" in command_names
        assert stats["filtered_commands"] <= len(self.sample_commands)
        assert stats["tokens_saved"] > 0

    def test_multiple_command_detection(self):
        """Test detection of multiple commands in one voice command"""
        voice_command = "turn on lights and play music"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert "turn_on_lights" in command_names or "turn_off_lights" in command_names
        assert "play_music" in command_names or "stop_music" in command_names
        assert stats["has_multiple_commands"] == True

    def test_fallback_to_no_keywords_commands(self):
        """Test fallback behavior when few matches found"""
        voice_command = "completely unrelated command"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should return some commands (either matches or fallbacks)
        assert len(filtered_commands) > 0
        assert stats["filtered_commands"] >= 1

    def test_empty_voice_command(self):
        """Test handling of empty voice command"""
        voice_command = ""
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should return some commands (fallback behavior)
        assert len(filtered_commands) > 0
        assert stats["total_commands"] == 8

    def test_empty_commands_list(self):
        """Test handling of empty commands list"""
        voice_command = "turn on lights"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, [])
        
        assert filtered_commands == []
        assert stats["total_commands"] == 0
        assert stats["filtered_commands"] == 0
        assert stats["reduction_percentage"] == 0

    def test_command_limit(self):
        """Test that command filtering returns all strong matches without artificial limits"""
        # Create many commands that would all match
        many_commands = []
        for i in range(15):
            many_commands.append(CommandDefinition(
                command_name=f"light_command_{i}",
                description=f"Light command {i}",
                parameters=[],
                keywords=["light", "lights"]
            ))
        
        voice_command = "turn on lights"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, many_commands)
        
        # Should return ALL matching commands (no artificial limit)
        # Since all 15 commands have "light" or "lights" keywords, they should all match
        assert len(filtered_commands) == 15
        assert stats["filtered_commands"] == 15
        assert stats["strong_matches"] == 15
        assert stats["reduction_percentage"] == 0.0  # No reduction since all commands match


class TestStandardSystemPromptProviderPreprocessing:
    """Test the StandardSystemPromptProvider with preprocessing functionality"""

    def setup_method(self):
        """Set up test fixtures"""
        self.sample_commands = [
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
                keywords=["unrelated", "other"]
            )
        ]
        self.node_context = {"room": "bedroom", "node_id": "test-node"}

    def test_preprocessing_enabled(self):
        """Test that preprocessing works when enabled"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            voice_command = "turn on the lights"
            prompt = provider.build_system_prompt(self.node_context, self.sample_commands, voice_command)
            
            # Should include relevant commands
            assert "turn_on_lights" in prompt
            # Should exclude unrelated commands
            assert "unrelated_command" not in prompt

    def test_preprocessing_disabled(self):
        """Test that preprocessing is disabled when configured"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider = StandardSystemPromptProvider()
            
            voice_command = "turn on the lights"
            prompt = provider.build_system_prompt(self.node_context, self.sample_commands, voice_command)
            
            # Should include all commands when preprocessing is disabled
            assert "turn_on_lights" in prompt
            assert "unrelated_command" in prompt

    def test_preprocessing_without_voice_command(self):
        """Test that preprocessing is skipped when no voice command is provided"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            # No voice command provided
            prompt = provider.build_system_prompt(self.node_context, self.sample_commands, None)
            
            # Should include all commands when no voice command is provided
            assert "turn_on_lights" in prompt
            assert "unrelated_command" in prompt

    def test_preprocessing_reduces_prompt_size(self):
        """Test that preprocessing actually reduces prompt size"""
        voice_command = "turn on the lights"
        
        # Without preprocessing
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider_no_preprocessing = StandardSystemPromptProvider()
            prompt_no_preprocessing = provider_no_preprocessing.build_system_prompt(
                self.node_context, self.sample_commands, voice_command
            )
        
        # With preprocessing
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider_with_preprocessing = StandardSystemPromptProvider()
            prompt_with_preprocessing = provider_with_preprocessing.build_system_prompt(
                self.node_context, self.sample_commands, voice_command
            )
        
        # Preprocessing should reduce prompt size
        assert len(prompt_with_preprocessing) < len(prompt_no_preprocessing)

    def test_provider_name(self):
        """Test that the provider name is correct"""
        provider = StandardSystemPromptProvider()
        assert provider.name == "STANDARD"

    def test_logging_integration(self):
        """Test that filtering statistics are logged"""
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider = StandardSystemPromptProvider()
            
            with patch('app.context_providers.standard_system_prompt_provider.logger') as mock_logger:
                voice_command = "turn on the lights"
                provider.build_system_prompt(self.node_context, self.sample_commands, voice_command)
                
                # Should log filtering statistics
                mock_logger.info.assert_called_once()
                call_args = mock_logger.info.call_args[0][0]
                assert "Command filtering:" in call_args
                assert "reduction" in call_args
                assert "tokens saved" in call_args


class TestCommandDefinitionKeywords:
    """Test the CommandDefinition model with keywords"""

    def test_command_definition_with_keywords(self):
        """Test creating CommandDefinition with keywords"""
        cmd = CommandDefinition(
            command_name="test_command",
            description="Test command",
            parameters=[CommandParameter(name="param1", type="str")],
            keywords=["test", "keyword"]
        )
        
        assert cmd.command_name == "test_command"
        assert cmd.keywords == ["test", "keyword"]

    def test_command_definition_without_keywords(self):
        """Test creating CommandDefinition without keywords"""
        cmd = CommandDefinition(
            command_name="test_command",
            description="Test command",
            parameters=[CommandParameter(name="param1", type="str")]
        )
        
        assert cmd.command_name == "test_command"
        assert cmd.keywords is None

    def test_command_definition_empty_keywords(self):
        """Test creating CommandDefinition with empty keywords list"""
        cmd = CommandDefinition(
            command_name="test_command",
            description="Test command",
            parameters=[CommandParameter(name="param1", type="str")],
            keywords=[]
        )
        
        assert cmd.command_name == "test_command"
        assert cmd.keywords == []


class TestIntegrationWithExistingSystem:
    """Integration tests to ensure the new functionality works with existing system"""

    def test_interface_compatibility(self):
        """Test that the updated interface is compatible"""
        from app.core.interfaces.ijarvis_context_provider import ISystemPromptProvider
        from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider
        
        provider = StandardSystemPromptProvider()
        assert isinstance(provider, ISystemPromptProvider)
        
        # Test that the method signature is correct
        node_context = {"room": "test", "node_id": "test"}
        commands = [CommandDefinition(
            command_name="test",
            description="test",
            parameters=[],
            keywords=["test"]
        )]
        
        # Should work with voice_command parameter
        prompt1 = provider.build_system_prompt(node_context, commands, "test command")
        assert isinstance(prompt1, str)
        
        # Should work without voice_command parameter
        prompt2 = provider.build_system_prompt(node_context, commands, None)
        assert isinstance(prompt2, str)
        
        # Should work with positional arguments for backward compatibility
        prompt3 = provider.build_system_prompt(node_context, commands)
        assert isinstance(prompt3, str)

    def test_environment_variable_integration(self):
        """Test that environment variable integration works correctly"""
        # Test default behavior (should be disabled by default)
        provider = StandardSystemPromptProvider()
        assert provider.preprocessing_enabled == False
        
        # Test enabling via environment variable
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "true"}):
            provider_enabled = StandardSystemPromptProvider()
            assert provider_enabled.preprocessing_enabled == True
        
        # Test case insensitive
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "TRUE"}):
            provider_enabled_upper = StandardSystemPromptProvider()
            assert provider_enabled_upper.preprocessing_enabled == True
        
        # Test explicit disable
        with patch.dict(os.environ, {"COMMAND_PREPROCESSING_ENABLED": "false"}):
            provider_disabled = StandardSystemPromptProvider()
            assert provider_disabled.preprocessing_enabled == False


if __name__ == "__main__":
    pytest.main([__file__]) 