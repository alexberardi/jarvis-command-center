#!/usr/bin/env python3
"""
Example usage of the new keyword-based command filtering system
"""
import os
from app.context_providers.standard_command_inference_prompt_provider import StandardCommandInferenceSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter

def create_example_commands():
    """Create example commands showing how to define keywords"""
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
            command_name="set_timer",
            description="Set a timer",
            parameters=[
                CommandParameter(name="duration", type="int"),
                CommandParameter(name="unit", type="str")
            ],
            keywords=["timer", "alarm", "remind", "minutes", "hours", "seconds", "time", "countdown"]
        ),
        CommandDefinition(
            command_name="check_weather",
            description="Check weather conditions",
            parameters=[],
            keywords=["weather", "temperature", "forecast", "rain", "sunny", "cloudy"]
        ),
        CommandDefinition(
            command_name="start_vacuum",
            description="Start vacuum cleaner",
            parameters=[],
            keywords=["vacuum", "clean", "start", "roomba"]
        ),
        CommandDefinition(
            command_name="lock_door",
            description="Lock a door",
            parameters=[CommandParameter(name="door", type="str")],
            keywords=["lock", "door", "secure"]
        ),
        CommandDefinition(
            command_name="open_garage",
            description="Open garage door",
            parameters=[],
            keywords=["garage", "door", "open"]
        ),
        CommandDefinition(
            command_name="water_plants",
            description="Water plants",
            parameters=[],
            keywords=["water", "plants", "garden", "irrigation"]
        ),
        # Example of a command with no keywords (will be included in fallback scenarios)
        CommandDefinition(
            command_name="emergency_shutdown",
            description="Emergency system shutdown",
            parameters=[],
            keywords=None  # No keywords - will be included when no other commands match
        )
    ]

def demonstrate_usage():
    """Demonstrate the keyword-based command filtering system"""
    print("🏠 Smart Home Command Filtering System")
    print("=" * 60)
    
    commands = create_example_commands()
    node_context = {"room": "living room", "node_id": "demo-node"}
    
    # Example 1: Enable preprocessing
    print("\n1️⃣ Enabling command preprocessing:")
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    provider = StandardCommandInferenceSystemPromptProvider()
    
    test_commands = [
        "turn on the lights",
        "set temperature to 72 degrees",
        "play some jazz music",
        "start the vacuum cleaner",
        "turn off lights and play music",  # Multiple commands
        "water the plants",
        "something completely random"  # Should fallback to commands with no keywords
    ]
    
    for voice_command in test_commands:
        print(f"\n🎤 Voice Command: '{voice_command}'")
        
        # Build system prompt with preprocessing
        system_prompt = provider.build_system_prompt(
            node_context, 
            commands, 
            voice_command
        )
        
        # Count how many commands are in the prompt
        command_count = system_prompt.count('command_name')
        total_commands = len(commands)
        reduction = ((total_commands - command_count) / total_commands) * 100
        
        print(f"   📊 Commands filtered: {total_commands} → {command_count} ({reduction:.1f}% reduction)")
        print(f"   📝 Prompt size: {len(system_prompt):,} characters")
        
        # Show which commands were included
        included_commands = []
        for cmd in commands:
            if cmd.command_name in system_prompt:
                included_commands.append(cmd.command_name)
        
        print(f"   ✅ Included commands: {', '.join(included_commands)}")
    
    # Example 2: Disable preprocessing
    print(f"\n" + "=" * 60)
    print("2️⃣ Disabling command preprocessing:")
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "false"
    provider_no_filter = StandardCommandInferenceSystemPromptProvider()
    
    voice_command = "turn on the lights"
    system_prompt_no_filter = provider_no_filter.build_system_prompt(
        node_context, 
        commands, 
        voice_command
    )
    
    print(f"\n🎤 Voice Command: '{voice_command}'")
    print(f"   📊 Commands included: {len(commands)} (all commands)")
    print(f"   📝 Prompt size: {len(system_prompt_no_filter):,} characters")
    
    # Compare with preprocessing enabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    provider_with_filter = StandardCommandInferenceSystemPromptProvider()
    system_prompt_with_filter = provider_with_filter.build_system_prompt(
        node_context, 
        commands, 
        voice_command
    )
    
    savings = len(system_prompt_no_filter) - len(system_prompt_with_filter)
    savings_percent = (savings / len(system_prompt_no_filter)) * 100
    
    print(f"\n📈 Comparison:")
    print(f"   Without preprocessing: {len(system_prompt_no_filter):,} characters")
    print(f"   With preprocessing: {len(system_prompt_with_filter):,} characters")
    print(f"   Savings: {savings:,} characters ({savings_percent:.1f}% reduction)")
    
    # Example 3: Show how to add custom keywords
    print(f"\n" + "=" * 60)
    print("3️⃣ Adding custom commands with keywords:")
    
    custom_command = CommandDefinition(
        command_name="brew_coffee",
        description="Brew coffee with specific settings",
        parameters=[
            CommandParameter(name="strength", type="str"),
            CommandParameter(name="size", type="str")
        ],
        keywords=["coffee", "brew", "espresso", "latte", "cappuccino", "caffeine", "drink"]
    )
    
    commands_with_custom = commands + [custom_command]
    
    voice_command = "brew me a strong coffee"
    system_prompt_custom = provider_with_filter.build_system_prompt(
        node_context, 
        commands_with_custom, 
        voice_command
    )
    
    print(f"\n🎤 Voice Command: '{voice_command}'")
    print(f"   📊 Total commands: {len(commands_with_custom)}")
    print(f"   📊 Commands in prompt: {system_prompt_custom.count('command_name')}")
    print(f"   ✅ Includes brew_coffee: {'brew_coffee' in system_prompt_custom}")
    print(f"   ❌ Excludes vacuum: {'start_vacuum' not in system_prompt_custom}")
    
    print(f"\n" + "=" * 60)
    print("✨ Summary:")
    print("• Add keywords to CommandDefinition objects to enable filtering")
    print("• Set COMMAND_PREPROCESSING_ENABLED=true to enable filtering")
    print("• Commands with no keywords serve as fallback for unmatched queries")
    print("• Multiple commands are detected using conjunctions (and, then, also)")
    print("• Filtering can reduce prompt size by 50-90% depending on the command set")

if __name__ == "__main__":
    demonstrate_usage() 