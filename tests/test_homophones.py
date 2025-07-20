import pytest
from app.context_providers.command_filter import CommandFilter
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestHomophonesAndSpeechRecognition:
    """Test handling of homophones and speech recognition artifacts from Whisper"""

    def setup_method(self):
        """Set up test fixtures"""
        self.command_filter = CommandFilter()
        
        self.sample_commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["light", "lights", "on", "turn on", "illuminate", "bright"]
            ),
            CommandDefinition(
                command_name="set_temperature",
                description="Set temperature in a room",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="degrees", type="int")
                ],
                keywords=["temperature", "temp", "degrees", "hot", "cold", "thermostat", "warm", "cool"]
            ),
            CommandDefinition(
                command_name="play_music",
                description="Play music",
                parameters=[CommandParameter(name="song", type="str")],
                keywords=["play", "music", "song", "audio", "sound", "tune"]
            ),
            CommandDefinition(
                command_name="start_vacuum",
                description="Start the vacuum cleaner",
                parameters=[CommandParameter(name="room", type="str")],
                keywords=["vacuum", "clean", "cleaning", "suck", "dust", "sweep"]
            ),
            CommandDefinition(
                command_name="lock_door",
                description="Lock a door",
                parameters=[CommandParameter(name="door", type="str")],
                keywords=["lock", "secure", "door", "entrance", "close"]
            ),
            CommandDefinition(
                command_name="unlock_door",
                description="Unlock a door",
                parameters=[CommandParameter(name="door", type="str")],
                keywords=["unlock", "door", "open", "unlock"]
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
                keywords=["garage", "door", "open", "gate"]
            )
        ]

    def test_light_lite_homophones(self):
        """Test light/lite homophones"""
        test_cases = [
            ("turn on the lite", "turn_on_lights"),
            ("turn on the lights", "turn_on_lights"),
            ("turn on the lites", "turn_on_lights"),
            ("turn on the light", "turn_on_lights"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nHomophone Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_temperature_thermostat_variations(self):
        """Test temperature/thermostat variations"""
        test_cases = [
            ("set the thermostat", "set_temperature"),
            ("set the thermostate", "set_temperature"),
            ("set the temperature", "set_temperature"),
            ("set the temp", "set_temperature"),
            ("set the term", "set_temperature"),  # Common STT error
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nTemperature Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_music_musical_variations(self):
        """Test music/musical variations"""
        test_cases = [
            ("play music", "play_music"),
            ("play musical", "play_music"),
            ("play a song", "play_music"),
            ("play some tunes", "play_music"),
            ("play audio", "play_music"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nMusic Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_vacuum_vacuuming_variations(self):
        """Test vacuum/vacuuming variations"""
        test_cases = [
            ("start the vacuum", "start_vacuum"),
            ("start vacuuming", "start_vacuum"),
            ("start the vacuum cleaner", "start_vacuum"),
            ("start cleaning", "start_vacuum"),
            ("start the cleaner", "start_vacuum"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nVacuum Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_door_lock_unlock_variations(self):
        """Test door lock/unlock variations"""
        test_cases = [
            ("lock the door", "lock_door"),
            ("lock the doors", "lock_door"),
            ("secure the door", "lock_door"),
            ("unlock the door", "unlock_door"),
            ("open the door", "unlock_door"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nDoor Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_timer_alarm_variations(self):
        """Test timer/alarm variations"""
        test_cases = [
            ("set a timer", "set_timer"),
            ("set an alarm", "set_timer"),
            ("set a reminder", "set_timer"),
            ("start a timer", "set_timer"),
            ("set the timer", "set_timer"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nTimer Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_garage_gate_variations(self):
        """Test garage/gate variations"""
        test_cases = [
            ("open the garage", "open_garage"),
            ("open the garage door", "open_garage"),
            ("open the gate", "open_garage"),
            ("open garage", "open_garage"),
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nGarage Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_common_stt_errors(self):
        """Test common speech-to-text transcription errors"""
        test_cases = [
            ("turn on the lites", "turn_on_lights"),  # lites instead of lights
            ("set the term", "set_temperature"),     # term instead of temp
            ("play musick", "play_music"),           # musick instead of music
            ("start the vacume", "start_vacuum"),    # vacume instead of vacuum
            ("lock the dore", "lock_door"),          # dore instead of door
            ("set a timmer", "set_timer"),           # timmer instead of timer
            ("open the garadge", "open_garage"),     # garadge instead of garage
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nSTT Error Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_phonetic_variations(self):
        """Test phonetic variations that Whisper might produce"""
        test_cases = [
            ("turn on the lites", "turn_on_lights"),      # lites/lights
            ("set the thermostate", "set_temperature"),   # thermostate/thermostat
            ("play musick", "play_music"),               # musick/music
            ("start the vacume", "start_vacuum"),        # vacume/vacuum
            ("lock the dore", "lock_door"),              # dore/door
            ("set a timmer", "set_timer"),               # timmer/timer
            ("open the garadge", "open_garage"),         # garadge/garage
            ("turn on the brights", "turn_on_lights"),   # brights/lights
            ("set the warm", "set_temperature"),         # warm/temperature
            ("play a tune", "play_music"),               # tune/music
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nPhonetic Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_accent_variations(self):
        """Test variations that might occur with different accents"""
        test_cases = [
            ("turn on the lites", "turn_on_lights"),      # Common accent variation
            ("set the temp", "set_temperature"),          # Abbreviation
            ("play some tunes", "play_music"),           # Synonym
            ("start cleaning", "start_vacuum"),          # Related action
            ("secure the door", "lock_door"),            # Synonym
            ("set an alarm", "set_timer"),               # Related concept
            ("open the gate", "open_garage"),            # Related concept
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nAccent Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"

    def test_whisper_artifacts(self):
        """Test common Whisper transcription artifacts"""
        test_cases = [
            ("turn on the lights", "turn_on_lights"),     # Perfect transcription
            ("turn on the lites", "turn_on_lights"),      # Common error
            ("turn on the light", "turn_on_lights"),      # Singular vs plural
            ("turn on lights", "turn_on_lights"),         # Missing "the"
            ("turn on the light's", "turn_on_lights"),    # Extra apostrophe
            ("turn on the lite's", "turn_on_lights"),     # Error + apostrophe
        ]
        
        for voice_command, expected_command in test_cases:
            filtered_commands, stats = self.command_filter.extract_relevant_commands(voice_command, self.sample_commands)
            command_names = [cmd.command_name for cmd in filtered_commands]
            
            print(f"\nWhisper Artifact Test: '{voice_command}'")
            print(f"Expected: {expected_command}")
            print(f"Found: {command_names}")
            print(f"Stats: {stats}")
            
            assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"


def test_homophones_manual():
    """Manual test to demonstrate homophone handling"""
    command_filter = CommandFilter()
    
    commands = [
        CommandDefinition(
            command_name="turn_on_lights",
            description="Turn on lights",
            parameters=[],
            keywords=["light", "lights", "bright", "illuminate"]
        ),
        CommandDefinition(
            command_name="set_temperature",
            description="Set temperature",
            parameters=[],
            keywords=["temperature", "temp", "thermostat", "warm", "cool"]
        ),
        CommandDefinition(
            command_name="play_music",
            description="Play music",
            parameters=[],
            keywords=["play", "music", "song", "tune", "audio"]
        )
    ]
    
    # Test various homophones and STT variations
    test_cases = [
        ("turn on the lites", "turn_on_lights"),      # lites/lights
        ("set the thermostate", "set_temperature"),   # thermostate/thermostat
        ("play musick", "play_music"),               # musick/music
        ("turn on the brights", "turn_on_lights"),   # brights/lights
        ("set the warm", "set_temperature"),         # warm/temperature
        ("play a tune", "play_music"),               # tune/music
    ]
    
    print("Homophone and STT Variation Testing:")
    print("=" * 50)
    
    for voice_command, expected_command in test_cases:
        filtered_commands, stats = command_filter.extract_relevant_commands(voice_command, commands)
        
        print(f"\nVoice command: '{voice_command}'")
        print(f"Expected: {expected_command}")
        print(f"Found: {[cmd.command_name for cmd in filtered_commands]}")
        print(f"Max fuzzy score: {stats['max_fuzzy_score']}")
        print(f"Confidence tier: {stats['confidence_tier']}")
        
        # Should find the expected command
        command_names = [cmd.command_name for cmd in filtered_commands]
        assert expected_command in command_names, f"Expected {expected_command} for '{voice_command}'"
        print("✅ PASS")
    
    print(f"\n🎉 All homophone tests passed!") 