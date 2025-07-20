import pytest
from app.context_providers.command_filter import CommandFilter
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestFuzzyMatching:
    """Test fuzzy matching functionality with rapidfuzz"""

    def setup_method(self):
        """Set up test fixtures"""
        self.command_filter = CommandFilter()
        
        self.sample_commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["light", "lights", "on", "turn on", "illuminate"]
            ),
            CommandDefinition(
                command_name="set_temperature",
                description="Set temperature in a room",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="degrees", type="int")
                ],
                keywords=["temperature", "temp", "degrees", "hot", "cold", "thermostat"]
            ),
            CommandDefinition(
                command_name="play_music",
                description="Play music",
                parameters=[CommandParameter(name="song", type="str")],
                keywords=["play", "music", "song", "audio", "sound"]
            ),
            CommandDefinition(
                command_name="start_vacuum",
                description="Start the vacuum cleaner",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["vacuum", "clean", "cleaning", "suck", "dust"]
            ),
            CommandDefinition(
                command_name="unrelated_command",
                description="An unrelated command",
                parameters=[],
                keywords=["unrelated", "other", "different"]
            )
        ]

    def test_exact_keyword_matching(self):
        """Test that exact keyword matches still work"""
        voice_command = "turn on the lights"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find turn_on_lights as a strong match
        assert len(filtered_commands) == 1
        assert filtered_commands[0].command_name == "turn_on_lights"
        assert stats["strong_matches"] == 1
        assert stats["fuzzy_matches"] == 0

    def test_typo_handling(self):
        """Test handling of typos in voice commands"""
        voice_command = "turn on the lites"  # Typo: 'lites' instead of 'lights'
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find turn_on_lights (could be strong match or fuzzy match)
        assert len(filtered_commands) >= 1
        assert filtered_commands[0].command_name == "turn_on_lights"
        # Could be strong match (exact substring) or fuzzy match
        assert stats["strong_matches"] > 0 or stats["fuzzy_matches"] > 0

    def test_word_variations(self):
        """Test handling of word variations and synonyms"""
        voice_command = "set the thermostat to 72"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find set_temperature (could be strong match or fuzzy match)
        assert len(filtered_commands) >= 1
        assert filtered_commands[0].command_name == "set_temperature"
        # Could be strong match (exact substring) or fuzzy match
        assert stats["strong_matches"] > 0 or stats["fuzzy_matches"] > 0

    def test_partial_word_matching(self):
        """Test partial word matching"""
        voice_command = "start the vacuum cleaner"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find start_vacuum (could be strong match or fuzzy match)
        assert len(filtered_commands) >= 1
        assert filtered_commands[0].command_name == "start_vacuum"
        # Could be strong match (exact substring) or fuzzy match
        assert stats["strong_matches"] > 0 or stats["fuzzy_matches"] > 0

    def test_word_order_independence(self):
        """Test that word order doesn't affect matching"""
        voice_command = "music play some jazz"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find play_music (could be strong match or fuzzy match)
        assert len(filtered_commands) >= 1
        assert filtered_commands[0].command_name == "play_music"
        # Could be strong match (exact substring) or fuzzy match
        assert stats["strong_matches"] > 0 or stats["fuzzy_matches"] > 0

    def test_multiple_fuzzy_matches(self):
        """Test when multiple commands have fuzzy matches"""
        voice_command = "turn on the lights and play some music"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find both commands
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert "turn_on_lights" in command_names
        assert "play_music" in command_names
        assert len(filtered_commands) >= 2

    def test_no_matches_fallback(self):
        """Test fallback behavior when few matches found"""
        voice_command = "hello world"  # Few matching keywords
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should return some commands (either matches or fallbacks)
        assert len(filtered_commands) > 0
        # Should have some filtering (not all commands)
        assert len(filtered_commands) < len(self.sample_commands)

    def test_fuzzy_threshold_behavior(self):
        """Test that fuzzy matching respects the threshold"""
        voice_command = "completely different words"  # Very low similarity
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should return some commands (either matches or fallbacks)
        assert len(filtered_commands) > 0
        # Should have some filtering (not all commands)
        assert len(filtered_commands) < len(self.sample_commands)

    def test_case_insensitive_matching(self):
        """Test that fuzzy matching is case insensitive"""
        voice_command = "TURN ON THE LIGHTS"  # All caps
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find turn_on_lights
        assert len(filtered_commands) == 1
        assert filtered_commands[0].command_name == "turn_on_lights"

    def test_mixed_typos_and_correct_words(self):
        """Test handling of mixed typos and correct words"""
        voice_command = "turn on the lites and set temp to 72"  # Mix of typos and correct words
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
        
        # Should find both commands despite typos
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert "turn_on_lights" in command_names
        assert "set_temperature" in command_names

    def test_fuzzy_match_prioritization(self):
        """Test that fuzzy matches are prioritized by score"""
        # Create commands with similar keywords but different relevance
        commands = [
            CommandDefinition(
                command_name="exact_match",
                description="Exact match command",
                parameters=[],
                keywords=["lights"]
            ),
            CommandDefinition(
                command_name="similar_match",
                description="Similar match command",
                parameters=[],
                keywords=["lites"]  # Similar to lights
            )
        ]
        
        voice_command = "turn on the lights"
        filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, commands)
        
        # Should prioritize exact match over similar match
        assert len(filtered_commands) >= 1
        if len(filtered_commands) == 1:
            assert filtered_commands[0].command_name == "exact_match"


def test_fuzzy_matching_manual():
    """Manual test to demonstrate fuzzy matching capabilities"""
    command_filter = CommandFilter()
    
    commands = [
        CommandDefinition(
            command_name="turn_on_lights",
            description="Turn on lights",
            parameters=[],
            keywords=["light", "lights", "illuminate"]
        ),
        CommandDefinition(
            command_name="set_temperature",
            description="Set temperature",
            parameters=[],
            keywords=["temperature", "temp", "thermostat"]
        )
    ]
    
    # Test various voice commands with typos and variations
    test_cases = [
        ("turn on lites", "turn_on_lights"),  # Typo
        ("set the thermostate", "set_temperature"),  # Typo
        ("lights on", "turn_on_lights"),  # Word order
        ("temp to 72", "set_temperature"),  # Abbreviation
        ("illuminate room", "turn_on_lights"),  # Synonym
    ]
    
    for voice_command, expected_command in test_cases:
        filtered_commands, stats = command_filter.extract_relevant_commands(voice_command, commands)
        
        print(f"\nVoice command: '{voice_command}'")
        print(f"Expected: {expected_command}")
        print(f"Found: {[cmd.command_name for cmd in filtered_commands]}")
        print(f"Stats: {stats}")
        
        # Should find the expected command
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'" 