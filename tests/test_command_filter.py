#!/usr/bin/env python3
"""
Test script to verify command filtering works correctly
Compatible with pytest framework
"""
import pytest
from app.context_providers.command_filter import CommandFilter
from app.request_models.voice_command_request import CommandDefinition, CommandParameter

def create_sample_commands():
    """Create sample commands for testing"""
    return [
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
            command_name="open_garage",
            description="Open garage door",
            parameters=[],
            keywords=["garage", "door", "open"]
        ),
        CommandDefinition(
            command_name="lock_door",
            description="Lock a door",
            parameters=[CommandParameter(name="door", type="str")],
            keywords=["lock", "door", "secure"]
        ),
        CommandDefinition(
            command_name="unlock_door",
            description="Unlock a door",
            parameters=[CommandParameter(name="door", type="str")],
            keywords=["unlock", "door", "open"]
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

class TestCommandFilteringScenarios:
    """Test various command filtering scenarios"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.filter_obj = CommandFilter()
        self.commands = create_sample_commands()
    
    @pytest.mark.parametrize("voice_command,expected_commands", [
        ("turn on the lights", ["turn_on_lights"]),
        ("turn off lights in bedroom", ["turn_off_lights"]),
        ("set temperature to 72", ["set_temperature"]),
        ("play some jazz music", ["play_music"]),
        ("stop the music", ["stop_music"]),
        ("set timer for 5 minutes", ["set_timer"]),
        ("open garage door", ["open_garage"]),
        ("lock the front door", ["lock_door"]),
        ("unlock door", ["unlock_door"]),
        ("start vacuum", ["start_vacuum"]),
        ("turn on lights and play music", ["turn_on_lights", "play_music"]),
        ("set temperature and start timer", ["set_temperature", "set_timer"]),
    ])
    def test_specific_command_filtering(self, voice_command, expected_commands):
        """Test filtering for specific voice commands"""
        filtered_commands, stats = self.filter_obj.extract_relevant_commands(voice_command, self.commands)
        filtered_command_names = [cmd.command_name for cmd in filtered_commands]
        
        # Check if all expected commands are present
        for expected_cmd in expected_commands:
            assert expected_cmd in filtered_command_names, f"Expected command '{expected_cmd}' not found in filtered results for '{voice_command}'"
        
        # Verify statistics
        assert stats["total_commands"] == len(self.commands)
        assert stats["filtered_commands"] > 0
        assert stats["reduction_percentage"] >= 0
        assert stats["tokens_saved"] >= 0
    
    def test_fallback_behavior(self):
        """Test fallback behavior for unmatched commands"""
        voice_command = "random unrelated command"
        filtered_commands, stats = self.filter_obj.extract_relevant_commands(voice_command, self.commands)
        filtered_command_names = [cmd.command_name for cmd in filtered_commands]
        
        # Should return some commands (either matches or fallbacks)
        assert len(filtered_commands) > 0
        assert stats["filtered_commands"] > 0
    
    def test_empty_command_handling(self):
        """Test handling of empty voice command"""
        voice_command = ""
        filtered_commands, stats = self.filter_obj.extract_relevant_commands(voice_command, self.commands)
        
        # Should return some commands (fallback behavior)
        assert len(filtered_commands) > 0
        assert stats["total_commands"] == len(self.commands)
    
    def test_multiple_command_detection(self):
        """Test detection of multiple commands"""
        voice_command = "turn on lights and play music"
        filtered_commands, stats = self.filter_obj.extract_relevant_commands(voice_command, self.commands)
        
        # Should detect multiple commands
        assert stats["has_multiple_commands"] == True
        
        # Should include relevant commands for both actions
        filtered_command_names = [cmd.command_name for cmd in filtered_commands]
        light_commands = any(cmd in filtered_command_names for cmd in ["turn_on_lights", "turn_off_lights"])
        music_commands = any(cmd in filtered_command_names for cmd in ["play_music", "stop_music"])
        
        assert light_commands, "Should include light-related commands"
        assert music_commands, "Should include music-related commands"
    
    def test_filtering_statistics(self):
        """Test that filtering statistics are accurate"""
        voice_command = "turn on the lights"
        filtered_commands, stats = self.filter_obj.extract_relevant_commands(voice_command, self.commands)
        
        # Verify statistics consistency
        assert stats["total_commands"] == len(self.commands)
        assert stats["filtered_commands"] == len(filtered_commands)
        assert stats["filtered_commands"] <= stats["total_commands"]
        
        expected_reduction = ((stats["total_commands"] - stats["filtered_commands"]) / stats["total_commands"]) * 100
        assert abs(stats["reduction_percentage"] - expected_reduction) < 0.01  # Allow for floating point precision
        
        expected_tokens_saved = (stats["total_commands"] - stats["filtered_commands"]) * 15
        assert stats["tokens_saved"] == expected_tokens_saved

def test_command_filtering_manual():
    """Manual test function for standalone execution"""
    filter_obj = CommandFilter()
    commands = create_sample_commands()
    
    test_cases = [
        ("turn on the lights", ["turn_on_lights"]),
        ("turn off lights in bedroom", ["turn_off_lights"]),
        ("set temperature to 72", ["set_temperature"]),
        ("play some jazz music", ["play_music"]),
        ("stop the music", ["stop_music"]),
        ("set timer for 5 minutes", ["set_timer"]),
        ("open garage door", ["open_garage"]),
        ("lock the front door", ["lock_door"]),
        ("unlock door", ["unlock_door"]),
        ("start vacuum", ["start_vacuum"]),
        ("turn on lights and play music", ["turn_on_lights", "play_music"]),
        ("set temperature and start timer", ["set_temperature", "set_timer"]),
        ("random unrelated command", []),  # Should fall back to commands with no keywords
        ("", [])  # Empty command
    ]
    
    print("Testing Command Filtering:")
    print("=" * 50)
    
    for voice_command, expected_commands in test_cases:
        print(f"\nVoice Command: '{voice_command}'")
        
        filtered_commands, stats = filter_obj.extract_relevant_commands(voice_command, commands)
        filtered_command_names = [cmd.command_name for cmd in filtered_commands]
        
        print(f"Expected: {expected_commands}")
        print(f"Got: {filtered_command_names}")
        print(f"Stats: {stats}")
        
        # Check if we got the expected commands (order doesn't matter)
        if expected_commands:
            success = all(cmd in filtered_command_names for cmd in expected_commands)
        else:
            # For empty expected, check if we got fallback commands
            success = len(filtered_commands) > 0
        
        print(f"✓ PASS" if success else "✗ FAIL")

if __name__ == "__main__":
    # Run manual test if executed directly
    test_command_filtering_manual()
    
    # Also run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except:
        print("\nPytest not available, manual tests completed.") 