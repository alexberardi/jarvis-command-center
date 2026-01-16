#!/usr/bin/env python3
"""
Enhanced LLM Response Debug and Metrics Collection Script

This script provides comprehensive testing and metrics collection for real LLM API calls.
It can be used to:
1. Test only the real LLM API integration tests
2. Collect detailed metrics on request/response performance
3. Analyze LLM response quality and correctness
4. Compare different configurations and providers

Usage:
    python tests/debug_llm_responses.py --test-type=real-api
    python tests/debug_llm_responses.py --test-type=performance
    python tests/debug_llm_responses.py --test-type=quality
"""
import os
import json
import asyncio
import time
import argparse
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load .env file at module import
load_env_file()

from app.core.utils.rest_client import post
from app.deps import get_command_inference_system_prompt_provider, get_transcription_cleanup_system_prompt_provider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter

@dataclass
class LLMMetrics:
    """Comprehensive metrics for LLM API calls"""
    # Request metadata
    request_id: str
    timestamp: str
    voice_command: str
    node_context: Dict[str, Any]
    provider_name: str
    
    # Performance metrics
    response_time_ms: float
    total_latency_ms: float  # Including network overhead
    
    # Token metrics
    input_tokens: int
    output_tokens: int
    total_tokens: int
    system_prompt_tokens: int
    user_prompt_tokens: int
    
    # Cost metrics (estimated)
    estimated_cost_usd: float
    
    # Response quality metrics
    response_success: bool
    json_parse_success: bool
    command_extraction_success: bool
    expected_command_match: bool
    
    # Response content
    raw_response: str
    parsed_response: Optional[Dict[str, Any]]
    extracted_commands: List[Dict[str, Any]]
    
    # Error tracking
    errors: List[str]
    warnings: List[str]
    
    # Additional metrics
    system_prompt_length: int
    user_prompt_length: int
    response_length: int
    tokens_per_second: float
    cost_per_request_usd: float

def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (approximately 4 characters per token)"""
    return len(text) // 4

def estimate_cost(input_tokens: int, output_tokens: int, model: str = "gpt-4") -> float:
    """Estimate cost based on token usage"""
    # Rough cost estimates (adjust based on your actual pricing)
    costs = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},  # per 1K tokens
        "claude-3": {"input": 0.015, "output": 0.075},  # per 1K tokens
    }
    
    model_costs = costs.get(model, costs["gpt-4"])
    input_cost = (input_tokens / 1000) * model_costs["input"]
    output_cost = (output_tokens / 1000) * model_costs["output"]
    
    return input_cost + output_cost

def create_sample_commands() -> List[CommandDefinition]:
    """Create comprehensive sample commands for testing"""
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
            description="Stop music",
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
            command_name="dim_lights",
            description="Dim lights to a specific brightness",
            parameters=[
                CommandParameter(name="room", type="str"),
                CommandParameter(name="brightness", type="int")
            ],
            keywords=["dim", "brightness", "bright", "darker", "lighter", "dimmer"]
        ),
        CommandDefinition(
            command_name="open_blinds",
            description="Open blinds in a room",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["blinds", "curtains", "open", "window"]
        ),
        CommandDefinition(
            command_name="lock_door",
            description="Lock a door",
            parameters=[CommandParameter(name="door", type="str")],
            keywords=["lock", "door", "secure"]
        ),
        CommandDefinition(
            command_name="start_vacuum",
            description="Start vacuum cleaner",
            parameters=[],
            keywords=["vacuum", "clean", "start", "roomba"]
        ),
        CommandDefinition(
            command_name="check_weather",
            description="Check weather conditions",
            parameters=[],
            keywords=["weather", "temperature", "forecast", "rain", "sunny", "cloudy"]
        ),
        CommandDefinition(
            command_name="arm_security",
            description="Arm security system",
            parameters=[],
            keywords=["security", "alarm", "arm", "activate"]
        ),
        CommandDefinition(
            command_name="disarm_security",
            description="Disarm security system",
            parameters=[],
            keywords=["security", "alarm", "disarm", "deactivate"]
        ),
        CommandDefinition(
            command_name="set_scene",
            description="Set a lighting scene",
            parameters=[
                CommandParameter(name="scene", type="str"),
                CommandParameter(name="room", type="str")
            ],
            keywords=["scene", "mood", "lighting", "ambiance"]
        ),
        CommandDefinition(
            command_name="water_plants",
            description="Water plants",
            parameters=[],
            keywords=["water", "plants", "garden", "irrigation"]
        ),
        CommandDefinition(
            command_name="adjust_fan_speed",
            description="Adjust ceiling fan speed",
            parameters=[
                CommandParameter(name="room", type="str"),
                CommandParameter(name="speed", type="int")
            ],
            keywords=["fan", "ceiling", "speed", "faster", "slower", "breeze", "air"]
        ),
        CommandDefinition(
            command_name="close_blinds",
            description="Close blinds in a room",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["blinds", "curtains", "close", "shut", "window", "privacy"]
        ),
        CommandDefinition(
            command_name="unlock_door",
            description="Unlock a door",
            parameters=[CommandParameter(name="door", type="str")],
            keywords=["unlock", "door", "open", "access"]
        ),
        CommandDefinition(
            command_name="start_dishwasher",
            description="Start the dishwasher",
            parameters=[],
            keywords=["dishwasher", "dishes", "wash", "clean", "start"]
        ),
        CommandDefinition(
            command_name="stop_vacuum",
            description="Stop vacuum cleaner",
            parameters=[],
            keywords=["vacuum", "stop", "pause", "halt", "roomba"]
        ),
        CommandDefinition(
            command_name="set_alarm",
            description="Set an alarm",
            parameters=[
                CommandParameter(name="time", type="str"),
                CommandParameter(name="label", type="str")
            ],
            keywords=["alarm", "wake", "time", "morning", "remind", "clock"]
        ),
        CommandDefinition(
            command_name="turn_on_tv",
            description="Turn on television",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["tv", "television", "on", "turn on", "watch", "screen"]
        ),
        CommandDefinition(
            command_name="turn_off_tv",
            description="Turn off television",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["tv", "television", "off", "turn off", "screen"]
        ),
        CommandDefinition(
            command_name="change_channel",
            description="Change TV channel",
            parameters=[
                CommandParameter(name="room", type="str"),
                CommandParameter(name="channel", type="str")
            ],
            keywords=["channel", "tv", "change", "switch", "station"]
        ),
        CommandDefinition(
            command_name="adjust_volume",
            description="Adjust audio volume",
            parameters=[
                CommandParameter(name="room", type="str"),
                CommandParameter(name="volume", type="int")
            ],
            keywords=["volume", "loud", "quiet", "sound", "audio", "music", "tv"]
        ),
        CommandDefinition(
            command_name="start_coffee_maker",
            description="Start coffee maker",
            parameters=[],
            keywords=["coffee", "brew", "start", "maker", "espresso", "latte"]
        ),
        CommandDefinition(
            command_name="preheat_oven",
            description="Preheat oven to temperature",
            parameters=[CommandParameter(name="temperature", type="int")],
            keywords=["oven", "preheat", "heat", "temperature", "bake", "cook"]
        ),
        CommandDefinition(
            command_name="turn_on_air_conditioning",
            description="Turn on air conditioning",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["ac", "air conditioning", "cool", "cold", "cooling", "climate"]
        ),
        CommandDefinition(
            command_name="turn_off_air_conditioning",
            description="Turn off air conditioning",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["ac", "air conditioning", "off", "stop", "cooling"]
        ),
        CommandDefinition(
            command_name="open_garage_door",
            description="Open garage door",
            parameters=[],
            keywords=["garage", "door", "open", "car", "driveway"]
        ),
        CommandDefinition(
            command_name="close_garage_door",
            description="Close garage door",
            parameters=[],
            keywords=["garage", "door", "close", "shut", "secure"]
        ),
        CommandDefinition(
            command_name="turn_on_sprinklers",
            description="Turn on lawn sprinklers",
            parameters=[CommandParameter(name="zone", type="str")],
            keywords=["sprinklers", "water", "lawn", "garden", "irrigation", "zone"]
        ),
        CommandDefinition(
            command_name="turn_off_sprinklers",
            description="Turn off lawn sprinklers",
            parameters=[CommandParameter(name="zone", type="str")],
            keywords=["sprinklers", "stop", "off", "lawn", "water"]
        ),
        CommandDefinition(
            command_name="check_door_status",
            description="Check if doors are locked",
            parameters=[],
            keywords=["door", "status", "check", "locked", "secure", "safety"]
        ),
        CommandDefinition(
            command_name="check_window_status",
            description="Check if windows are open or closed",
            parameters=[],
            keywords=["window", "status", "check", "open", "closed"]
        ),
        CommandDefinition(
            command_name="turn_on_fireplace",
            description="Turn on fireplace",
            parameters=[],
            keywords=["fireplace", "fire", "warm", "heat", "cozy"]
        ),
        CommandDefinition(
            command_name="turn_off_fireplace",
            description="Turn off fireplace",
            parameters=[],
            keywords=["fireplace", "fire", "off", "extinguish"]
        ),
        CommandDefinition(
            command_name="set_pool_temperature",
            description="Set pool heater temperature",
            parameters=[CommandParameter(name="temperature", type="int")],
            keywords=["pool", "temperature", "heat", "warm", "degrees", "swim"]
        ),
        CommandDefinition(
            command_name="turn_on_pool_lights",
            description="Turn on pool lights",
            parameters=[],
            keywords=["pool", "lights", "on", "illuminate", "swim", "night"]
        ),
        CommandDefinition(
            command_name="turn_off_pool_lights",
            description="Turn off pool lights",
            parameters=[],
            keywords=["pool", "lights", "off", "dark"]
        ),
        CommandDefinition(
            command_name="start_meditation_mode",
            description="Start meditation mode with ambient lighting and sounds",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["meditation", "relax", "ambient", "calm", "zen", "peaceful"]
        ),
        CommandDefinition(
            command_name="start_party_mode",
            description="Start party mode with music and colorful lights",
            parameters=[CommandParameter(name="room", type="str")],
            keywords=["party", "music", "lights", "colorful", "fun", "celebration"]
        ),
        CommandDefinition(
            command_name="goodnight_routine",
            description="Execute goodnight routine (lights off, doors locked, etc.)",
            parameters=[],
            keywords=["goodnight", "sleep", "bedtime", "routine", "night"]
        ),
        CommandDefinition(
            command_name="good_morning_routine",
            description="Execute good morning routine (lights on, blinds open, etc.)",
            parameters=[],
            keywords=["good morning", "wake up", "morning", "routine", "sunrise"]
        )
    ]

async def test_llm_response_with_metrics(
    voice_command: str, 
    node_context: Dict[str, Any], 
    available_commands: List[CommandDefinition], 
    provider, 
    provider_name: str, 
    request_id: str,
    expected_command: Optional[str] = None
) -> LLMMetrics:
    """Test a single LLM response with comprehensive metrics collection"""
    
    start_time = time.time()
    errors = []
    warnings = []
    
    # Apply transcription cleanup if enabled
    transcription_cleanup_provider = get_transcription_cleanup_system_prompt_provider()
    cleaned_voice_command = voice_command
    
    if transcription_cleanup_provider.name != "NO_OP":
        try:
            cleanup_prompt = transcription_cleanup_provider.build_system_prompt(voice_command)
            
            # Create cleanup request
            cleanup_messages = [
                {"role": "system", "content": cleanup_prompt},
                {"role": "user", "content": voice_command}
            ]
            
            cleanup_request = {
                "model": "gpt-4",
                "messages": cleanup_messages,
                "temperature": 0.1,
                "max_tokens": 100,
                "stream": False
            }
            
            # Call LLM for cleanup
            llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
            cleanup_url = f"{llm_api_url.rstrip('/')}/v1/chat/completions"
            
            cleanup_response = await post(url=cleanup_url, json_data=cleanup_request)
            
            # Extract cleaned command
            if 'choices' in cleanup_response and cleanup_response['choices']:
                cleaned_voice_command = cleanup_response['choices'][0].get('message', {}).get('content', voice_command)
            elif 'content' in cleanup_response:
                cleaned_voice_command = cleanup_response['content']
            elif 'response' in cleanup_response:
                cleaned_voice_command = cleanup_response['response']
            
            # Strip quotes if present
            cleaned_voice_command = cleaned_voice_command.strip().strip('"').strip("'")
            
            if cleaned_voice_command != voice_command:
                warnings.append(f"Transcription cleaned: '{voice_command}' -> '{cleaned_voice_command}'")
                
        except Exception as e:
            errors.append(f"Transcription cleanup failed: {str(e)}")
            cleaned_voice_command = voice_command
    
    # Apply command preprocessing if enabled
    commands_to_use = available_commands
    preprocessing_enabled = os.getenv("COMMAND_PREPROCESSING_ENABLED", "false").lower() == "true"
    if preprocessing_enabled and cleaned_voice_command:
        from app.context_providers.command_filter import CommandFilter
        command_filter = CommandFilter()
        filtered_commands, stats = command_filter.extract_relevant_commands(cleaned_voice_command, available_commands)
        if stats['filtered_commands'] < stats['total_commands']:
            warnings.append(f"Command filtering: {stats['total_commands']} -> {stats['filtered_commands']} commands ({stats['reduction_percentage']:.1f}% reduction)")
        commands_to_use = filtered_commands

    # Build system prompt with preprocessed commands
    system_prompt = provider.build_system_prompt(node_context, commands_to_use, cleaned_voice_command)
    
    # Calculate token estimates
    system_prompt_tokens = estimate_tokens(system_prompt)
    user_prompt_tokens = estimate_tokens(cleaned_voice_command)
    input_tokens = system_prompt_tokens + user_prompt_tokens
    
    # Create OpenAI-style request
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": cleaned_voice_command}
    ]
    
    # OpenAI-style request payload
    request_payload = {
        "model": "gpt-4",  # Adjust based on your actual model
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1000,
        "stream": False
    }
    
    try:
        llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
        chat_url = f"{llm_api_url.rstrip('/')}/v1/chat/completions"
        
        # Make the API call
        llm_response = await post(url=chat_url, json_data=request_payload)
        
        end_time = time.time()
        response_time_ms = (end_time - start_time) * 1000
        total_latency_ms = response_time_ms  # Could be enhanced with network timing
        
        # Extract response content
        response_content = ""
        if 'choices' in llm_response and llm_response['choices']:
            response_content = llm_response['choices'][0].get('message', {}).get('content', '')
        elif 'content' in llm_response:
            response_content = llm_response['content']
        elif 'response' in llm_response:
            response_content = llm_response['response']
        elif isinstance(llm_response, str):
            response_content = llm_response
        
        # Calculate output metrics
        output_tokens = estimate_tokens(response_content)
        total_tokens = input_tokens + output_tokens
        tokens_per_second = total_tokens / (response_time_ms / 1000) if response_time_ms > 0 else 0
        
        # Estimate cost
        estimated_cost_usd = estimate_cost(input_tokens, output_tokens)
        
        # Parse response
        parsed_response = None
        json_parse_success = False
        command_extraction_success = False
        extracted_commands = []
        
        if response_content:
            try:
                # Clean up response content
                clean_content = response_content.strip()
                parsed_response = json.loads(clean_content)
                json_parse_success = True
                
                # Extract commands
                if 'commands' in parsed_response:
                    extracted_commands = parsed_response['commands']
                    command_extraction_success = len(extracted_commands) > 0
                else:
                    # Legacy format - single command
                    extracted_commands = [parsed_response]
                    command_extraction_success = True
                    
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error: {str(e)}")
                # Try to extract JSON from malformed response
                if response_content.strip().startswith('{'):
                    try:
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(response_content):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if json_end > 0 and json_end < len(response_content):
                            extra_content = response_content[json_end:].strip()
                            if extra_content:
                                warnings.append(f"Extra content after JSON: {repr(extra_content[:50])}")
                            
                            json_part = response_content[:json_end]
                            parsed_response = json.loads(json_part)
                            json_parse_success = True
                            
                            if 'commands' in parsed_response:
                                extracted_commands = parsed_response['commands']
                                command_extraction_success = len(extracted_commands) > 0
                            else:
                                extracted_commands = [parsed_response]
                                command_extraction_success = True
                    except:
                        pass
        else:
            errors.append("Empty response content")
        
        # Check response success
        response_success = json_parse_success and command_extraction_success and len(errors) == 0
        
        # Check expected command match
        expected_command_match = False
        if expected_command and extracted_commands:
            for cmd in extracted_commands:
                if cmd.get('command_name') == expected_command:
                    expected_command_match = True
                    break
        
        return LLMMetrics(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            voice_command=voice_command,
            node_context=node_context,
            provider_name=provider_name,
            response_time_ms=response_time_ms,
            total_latency_ms=total_latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            system_prompt_tokens=system_prompt_tokens,
            user_prompt_tokens=user_prompt_tokens,
            estimated_cost_usd=estimated_cost_usd,
            response_success=response_success,
            json_parse_success=json_parse_success,
            command_extraction_success=command_extraction_success,
            expected_command_match=expected_command_match,
            raw_response=response_content,
            parsed_response=parsed_response,
            extracted_commands=extracted_commands,
            errors=errors,
            warnings=warnings,
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(voice_command),
            response_length=len(response_content),
            tokens_per_second=tokens_per_second,
            cost_per_request_usd=estimated_cost_usd
        )
        
    except Exception as e:
        errors.append(f"API call error: {str(e)}")
        return LLMMetrics(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            voice_command=voice_command,
            node_context=node_context,
            provider_name=provider_name,
            response_time_ms=0,
            total_latency_ms=0,
            input_tokens=input_tokens,
            output_tokens=0,
            total_tokens=input_tokens,
            system_prompt_tokens=system_prompt_tokens,
            user_prompt_tokens=user_prompt_tokens,
            estimated_cost_usd=estimate_cost(input_tokens, 0),
            response_success=False,
            json_parse_success=False,
            command_extraction_success=False,
            expected_command_match=False,
            raw_response="",
            parsed_response=None,
            extracted_commands=[],
            errors=errors,
            warnings=warnings,
            system_prompt_length=len(system_prompt),
            user_prompt_length=len(voice_command),
            response_length=0,
            tokens_per_second=0,
            cost_per_request_usd=estimate_cost(input_tokens, 0)
        )

def print_metrics_summary(metrics_list: List[LLMMetrics]):
    """Print comprehensive metrics summary"""
    if not metrics_list:
        print("No metrics to summarize")
        return
    
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE METRICS SUMMARY")
    print(f"{'='*80}")
    
    # Performance metrics
    response_times = [m.response_time_ms for m in metrics_list]
    total_tokens = [m.total_tokens for m in metrics_list]
    input_tokens = [m.input_tokens for m in metrics_list]
    output_tokens = [m.output_tokens for m in metrics_list]
    costs = [m.estimated_cost_usd for m in metrics_list]
    
    print(f"📈 Performance Metrics:")
    print(f"   Total Requests: {len(metrics_list)}")
    print(f"   Average Response Time: {statistics.mean(response_times):.2f}ms")
    print(f"   Median Response Time: {statistics.median(response_times):.2f}ms")
    print(f"   Min Response Time: {min(response_times):.2f}ms")
    print(f"   Max Response Time: {max(response_times):.2f}ms")
    print(f"   Response Time Std Dev: {statistics.stdev(response_times):.2f}ms")
    
    print(f"\n🎯 Token Usage:")
    print(f"   Average Input Tokens: {statistics.mean(input_tokens):.1f}")
    print(f"   Average Output Tokens: {statistics.mean(output_tokens):.1f}")
    print(f"   Average Total Tokens: {statistics.mean(total_tokens):.1f}")
    print(f"   Total Tokens Used: {sum(total_tokens):,}")
    print(f"   Average Tokens/Second: {statistics.mean([m.tokens_per_second for m in metrics_list]):.1f}")
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Total Estimated Cost: ${sum(costs):.4f}")
    print(f"   Average Cost per Request: ${statistics.mean(costs):.4f}")
    print(f"   Cost per 1K Requests: ${sum(costs) * (1000 / len(metrics_list)):.2f}")
    
    # Success metrics
    successful_requests = [m for m in metrics_list if m.response_success]
    json_successful = [m for m in metrics_list if m.json_parse_success]
    command_successful = [m for m in metrics_list if m.command_extraction_success]
    
    print(f"\n✅ Success Metrics:")
    print(f"   Overall Success Rate: {len(successful_requests)/len(metrics_list)*100:.1f}%")
    print(f"   JSON Parse Success Rate: {len(json_successful)/len(metrics_list)*100:.1f}%")
    print(f"   Command Extraction Success Rate: {len(command_successful)/len(metrics_list)*100:.1f}%")
    
    # Error analysis
    all_errors = []
    for m in metrics_list:
        all_errors.extend(m.errors)
    
    if all_errors:
        print(f"\n❌ Error Analysis:")
        error_counts = {}
        for error in all_errors:
            error_type = error.split(":")[0] if ":" in error else error
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {error_type}: {count} occurrences")
    
    # Warning analysis
    all_warnings = []
    for m in metrics_list:
        all_warnings.extend(m.warnings)
    
    if all_warnings:
        print(f"\n⚠️  Warning Analysis:")
        warning_counts = {}
        for warning in all_warnings:
            warning_type = warning.split(":")[0] if ":" in warning else warning
            warning_counts[warning_type] = warning_counts.get(warning_type, 0) + 1
        
        for warning_type, count in sorted(warning_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {warning_type}: {count} occurrences")

async def run_real_api_tests():
    """Run tests that make real API calls to the LLM proxy"""
    print("🧪 Running Real LLM API Integration Tests")
    print("=" * 80)
    
    if not os.getenv("JARVIS_LLM_PROXY_API_URL"):
        print("❌ JARVIS_LLM_PROXY_API_URL environment variable not set!")
        print("   Set it to your LLM proxy API URL to run real API tests.")
        return []
    
    # Test cases with expected commands
    test_cases = [
        ("turn on the lights", {"room": "bedroom", "node_id": "node-1"}, "turn_on_lights"),
        ("set temperature to 72 degrees", {"room": "living room", "node_id": "node-2"}, "set_temperature"),
        ("play some jazz music", {"room": "kitchen", "node_id": "node-3"}, "play_music"),
        ("turn off lights and play music", {"room": "bedroom", "node_id": "node-1"}, None),  # Multiple commands
        ("start vacuum", {"room": "", "node_id": "node-4"}, "start_vacuum"),
        ("dim lights to 50 percent", {"room": "living room", "node_id": "node-2"}, "dim_lights"),
        ("lock the front door", {"room": "entryway", "node_id": "node-5"}, "lock_door"),
        ("set timer for 5 minutes", {"room": "kitchen", "node_id": "node-3"}, "set_timer"),
        ("open garage door", {"room": "garage", "node_id": "node-6"}, "open_garage_door"),
        ("check weather", {"room": "living room", "node_id": "node-1"}, "check_weather"),
    ]
    
    available_commands = create_sample_commands()
    provider = get_command_inference_system_prompt_provider()
    all_metrics = []
    
    for i, (voice_command, node_context, expected_command) in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(test_cases)}: '{voice_command}'")
        print(f"Room: {node_context.get('room', 'N/A')}")
        print(f"Expected Command: {expected_command or 'Multiple/Unknown'}")
        print(f"{'='*60}")
        
        metrics = await test_llm_response_with_metrics(
            voice_command=voice_command,
            node_context=node_context,
            available_commands=available_commands,
            provider=provider,
            provider_name=provider.name,
            request_id=f"Test-{i:02d}",
            expected_command=expected_command
        )
        
        all_metrics.append(metrics)
        
        # Print individual test results
        print(f"📊 Results:")
        print(f"   Response Time: {metrics.response_time_ms:.2f}ms")
        print(f"   Success: {metrics.response_success}")
        print(f"   Input Tokens: {metrics.input_tokens:,}")
        print(f"   Output Tokens: {metrics.output_tokens:,}")
        print(f"   Total Tokens: {metrics.total_tokens:,}")
        print(f"   Estimated Cost: ${metrics.estimated_cost_usd:.4f}")
        
        if metrics.extracted_commands:
            print(f"   Extracted Commands: {len(metrics.extracted_commands)}")
            for j, cmd in enumerate(metrics.extracted_commands, 1):
                print(f"     Command {j}: {cmd.get('command_name')} - Success: {cmd.get('success')}")
                if cmd.get('parameters'):
                    print(f"       Parameters: {cmd.get('parameters')}")
                if cmd.get('errors'):
                    print(f"       Errors: {cmd.get('errors')}")
        
        if metrics.errors:
            print(f"   Errors: {metrics.errors}")
        
        if metrics.warnings:
            print(f"   Warnings: {metrics.warnings}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    return all_metrics

async def run_performance_tests():
    """Run performance-focused tests"""
    print("🚀 Running Performance Tests")
    print("=" * 80)
    
    # Performance test cases
    test_cases = [
        ("turn on the lights", {"room": "bedroom", "node_id": "perf-1"}),
        ("set temperature to 72", {"room": "living room", "node_id": "perf-2"}),
        ("play jazz music", {"room": "kitchen", "node_id": "perf-3"}),
        ("start vacuum", {"room": "living room", "node_id": "perf-4"}),
        ("lock front door", {"room": "entryway", "node_id": "perf-5"}),
    ]
    
    available_commands = create_sample_commands()
    provider = get_command_inference_system_prompt_provider()
    all_metrics = []
    
    # Run multiple iterations for performance analysis
    iterations = 3
    
    for iteration in range(iterations):
        print(f"\n🔄 Performance Iteration {iteration + 1}/{iterations}")
        
        for i, (voice_command, node_context) in enumerate(test_cases, 1):
                    metrics = await test_llm_response_with_metrics(
            voice_command=voice_command,
            node_context=node_context,
            available_commands=available_commands,
            provider=provider,
            provider_name=provider.name,
            request_id=f"Perf-{iteration+1}-{i:02d}"
        )
        
        all_metrics.append(metrics)
        await asyncio.sleep(0.2)  # Shorter delay for performance tests
    
    return all_metrics

async def run_quality_tests():
    """Run quality-focused tests with edge cases"""
    print("🎯 Running Quality Tests")
    print("=" * 80)
    
    # Quality test cases with edge cases
    test_cases = [
        ("", {"room": "bedroom", "node_id": "qual-1"}),  # Empty command
        ("turn on the lights in the kitchen please", {"room": "kitchen", "node_id": "qual-2"}),  # Polite
        ("LIGHTS ON NOW!", {"room": "bedroom", "node_id": "qual-3"}),  # Shouting
        ("um turn on the like in the kitchen please", {"room": "kitchen", "node_id": "qual-4"}),  # Filler words
        ("set temp to seventy two degrees", {"room": "living room", "node_id": "qual-5"}),  # Word numbers
        ("turn on lights and play music and set temperature", {"room": "bedroom", "node_id": "qual-6"}),  # Multiple
        ("do something with the lights", {"room": "bedroom", "node_id": "qual-7"}),  # Ambiguous
        ("xyz123", {"room": "bedroom", "node_id": "qual-8"}),  # Nonsense
    ]
    
    available_commands = create_sample_commands()
    provider = get_command_inference_system_prompt_provider()
    all_metrics = []
    
    for i, (voice_command, node_context) in enumerate(test_cases, 1):
        print(f"\n🎯 Quality Test {i}/{len(test_cases)}: '{voice_command}'")
        
        metrics = await test_llm_response_with_metrics(
            voice_command=voice_command,
            node_context=node_context,
            available_commands=available_commands,
            provider=provider,
            provider_name=provider.name,
            request_id=f"Qual-{i:02d}"
        )
        
        all_metrics.append(metrics)
        
        # Print quality-specific results
        print(f"   Quality Score: {metrics.response_success}")
        print(f"   JSON Valid: {metrics.json_parse_success}")
        print(f"   Commands Extracted: {len(metrics.extracted_commands)}")
        print(f"   Errors: {len(metrics.errors)}")
        print(f"   Warnings: {len(metrics.warnings)}")
        
        await asyncio.sleep(0.5)
    
    return all_metrics

async def run_comprehensive_accuracy_tests():
    """Run comprehensive accuracy tests with 20 diverse voice commands"""
    print("🎯 Running Comprehensive Accuracy Tests")
    print("=" * 80)
    
    if not os.getenv("JARVIS_LLM_PROXY_API_URL"):
        print("❌ JARVIS_LLM_PROXY_API_URL environment variable not set!")
        print("   Set it to your LLM proxy API URL to run real API tests.")
        return []
    
    # Comprehensive test cases with diverse voice commands
    test_cases = [
        # Basic commands
        ("turn on the lights", {"room": "bedroom", "node_id": "comp-1"}, "turn_on_lights"),
        ("switch on the lights", {"room": "living room", "node_id": "comp-2"}, "turn_on_lights"),
        ("lights on", {"room": "kitchen", "node_id": "comp-3"}, "turn_on_lights"),
        ("turn off the lights", {"room": "bedroom", "node_id": "comp-4"}, "turn_off_lights"),
        ("shut off the lights", {"room": "living room", "node_id": "comp-5"}, "turn_off_lights"),
        
        # Temperature commands with various phrasings
        ("set temperature to 72 degrees", {"room": "living room", "node_id": "comp-6"}, "set_temperature"),
        ("change temperature to 68", {"room": "bedroom", "node_id": "comp-7"}, "set_temperature"),
        ("make it 75 degrees", {"room": "kitchen", "node_id": "comp-8"}, "set_temperature"),
        ("set the thermostat to 70", {"room": "living room", "node_id": "comp-9"}, "set_temperature"),
        ("temperature 65 degrees", {"room": "bedroom", "node_id": "comp-10"}, "set_temperature"),
        
        # Music commands
        ("play jazz music", {"room": "kitchen", "node_id": "comp-11"}, "play_music"),
        ("start playing rock music", {"room": "living room", "node_id": "comp-12"}, "play_music"),
        ("put on some classical", {"room": "bedroom", "node_id": "comp-13"}, "play_music"),
        ("play the blues", {"room": "kitchen", "node_id": "comp-14"}, "play_music"),
        ("turn on some music", {"room": "living room", "node_id": "comp-15"}, "play_music"),
        
        # Vacuum commands
        ("start the vacuum", {"room": "living room", "node_id": "comp-16"}, "start_vacuum"),
        ("turn on the vacuum cleaner", {"room": "bedroom", "node_id": "comp-17"}, "start_vacuum"),
        ("run the vacuum", {"room": "kitchen", "node_id": "comp-18"}, "start_vacuum"),
        
        # Door commands
        ("lock the front door", {"room": "entryway", "node_id": "comp-19"}, "lock_door"),
        ("unlock the back door", {"room": "kitchen", "node_id": "comp-20"}, "unlock_door"),
        
        # Timer commands
        ("set a timer for 5 minutes", {"room": "kitchen", "node_id": "comp-21"}, "set_timer"),
        ("start a 10 minute timer", {"room": "living room", "node_id": "comp-22"}, "set_timer"),
        ("timer 15 minutes", {"room": "bedroom", "node_id": "comp-23"}, "set_timer"),
        
        # Garage commands
        ("open the garage door", {"room": "garage", "node_id": "comp-24"}, "open_garage_door"),
        ("close the garage", {"room": "garage", "node_id": "comp-25"}, "close_garage_door"),
        
        # Weather commands
        ("what's the weather like", {"room": "living room", "node_id": "comp-26"}, "check_weather"),
        ("check the weather", {"room": "kitchen", "node_id": "comp-27"}, "check_weather"),
        ("weather forecast", {"room": "bedroom", "node_id": "comp-28"}, "check_weather"),
        
        # Light dimming
        ("dim the lights to 50 percent", {"room": "living room", "node_id": "comp-29"}, "dim_lights"),
        ("brighten the lights to 80 percent", {"room": "bedroom", "node_id": "comp-30"}, "dim_lights"),
        
        # Complex/multiple commands
        ("turn on lights and play music", {"room": "living room", "node_id": "comp-31"}, None),  # Multiple
        ("set temperature to 72 and lock the door", {"room": "bedroom", "node_id": "comp-32"}, None),  # Multiple
        ("start vacuum and turn off lights", {"room": "kitchen", "node_id": "comp-33"}, None),  # Multiple
        ("good morning routine", {"room": "bedroom", "node_id": "comp-34"}, "good_morning_routine"),
        ("goodnight routine", {"room": "bedroom", "node_id": "comp-35"}, "goodnight_routine"),
        
        # Edge cases and variations
        ("please turn on the lights", {"room": "living room", "node_id": "comp-36"}, "turn_on_lights"),  # Polite
        ("LIGHTS ON NOW!", {"room": "bedroom", "node_id": "comp-37"}, "turn_on_lights"),  # Shouting
        ("um turn on the like in the kitchen", {"room": "kitchen", "node_id": "comp-38"}, "turn_on_lights"),  # Filler words
        ("set temp to seventy two degrees", {"room": "living room", "node_id": "comp-39"}, "set_temperature"),  # Word numbers
        ("do something with the lights", {"room": "bedroom", "node_id": "comp-40"}, None),  # Ambiguous
    ]
    
    available_commands = create_sample_commands()
    provider = get_command_inference_system_prompt_provider()
    all_metrics = []
    
    print(f"🧪 Testing {len(test_cases)} diverse voice commands with JARVIS provider")
    print(f"🤖 Model: Fine-tuned Llama 3.1 8B")
    print(f"📊 Provider: {provider.name}")
    print()
    
    for i, (voice_command, node_context, expected_command) in enumerate(test_cases, 1):
        print(f"============================================================")
        print(f"Test {i}/{len(test_cases)}: '{voice_command}'")
        print(f"Room: {node_context.get('room', 'N/A')}")
        print(f"Expected Command: {expected_command or 'Multiple/Unknown'}")
        print(f"============================================================")
        
        metrics = await test_llm_response_with_metrics(
            voice_command=voice_command,
            node_context=node_context,
            available_commands=available_commands,
            provider=provider,
            provider_name=provider.name,
            request_id=f"Comp-{i:02d}",
            expected_command=expected_command
        )
        
        all_metrics.append(metrics)
        
        # Print detailed results
        print(f"📊 Results:")
        print(f"   Response Time: {metrics.response_time_ms:.2f}ms")
        print(f"   Success: {metrics.response_success}")
        print(f"   Input Tokens: {metrics.input_tokens}")
        print(f"   Output Tokens: {metrics.output_tokens}")
        print(f"   Total Tokens: {metrics.total_tokens}")
        print(f"   Estimated Cost: ${metrics.estimated_cost_usd:.4f}")
        print(f"   Extracted Commands: {len(metrics.extracted_commands)}")
        
        # Show extracted commands
        if metrics.extracted_commands:
            for j, cmd in enumerate(metrics.extracted_commands, 1):
                if 'c' in cmd and cmd['c']:
                    for k, command in enumerate(cmd['c'], 1):
                        success = command.get('s', False)
                        name = command.get('n', 'unknown')
                        params = command.get('p', {})
                        errors = command.get('e')
                        print(f"     Command {k}: {name} - Success: {success}")
                        if params:
                            print(f"       Parameters: {params}")
                        if errors:
                            print(f"       Errors: {errors}")
        
        # Show accuracy assessment
        if expected_command:
            if metrics.expected_command_match:
                print(f"   ✅ ACCURATE: Correctly identified '{expected_command}'")
            else:
                print(f"   ❌ INACCURATE: Expected '{expected_command}' but got different result")
        else:
            print(f"   🤔 MULTIPLE/AMBIGUOUS: No single expected command")
        
        print()
        await asyncio.sleep(0.3)  # Brief delay between requests
    
    return all_metrics

def save_metrics_to_file(metrics_list: List[LLMMetrics], filename: str):
    """Save metrics to JSON file for further analysis"""
    metrics_dicts = [asdict(m) for m in metrics_list]
    
    with open(filename, 'w') as f:
        json.dump(metrics_dicts, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {filename}")

async def main():
    """Main function to run different types of tests"""
    parser = argparse.ArgumentParser(description="LLM Response Debug and Metrics Collection")
    parser.add_argument(
        "--test-type", 
        choices=["real-api", "performance", "quality", "comprehensive_accuracy", "all"],
        default="real-api",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--save-metrics",
        action="store_true",
        help="Save metrics to JSON file"
    )
    parser.add_argument(
        "--output-file",
        default="llm_metrics.json",
        help="Output file for metrics (default: llm_metrics.json)"
    )
    
    args = parser.parse_args()
    
    all_metrics = []
    
    if args.test_type in ["real-api", "all"]:
        metrics = await run_real_api_tests()
        all_metrics.extend(metrics)
    
    if args.test_type in ["performance", "all"]:
        metrics = await run_performance_tests()
        all_metrics.extend(metrics)
    
    if args.test_type in ["quality", "all"]:
        metrics = await run_quality_tests()
        all_metrics.extend(metrics)
    
    if args.test_type in ["comprehensive_accuracy", "all"]:
        metrics = await run_comprehensive_accuracy_tests()
        all_metrics.extend(metrics)
    
    if all_metrics:
        print_metrics_summary(all_metrics)
        
        if args.save_metrics:
            save_metrics_to_file(all_metrics, args.output_file)
    else:
        print("No tests were run. Check your configuration and try again.")

if __name__ == "__main__":
    asyncio.run(main()) 