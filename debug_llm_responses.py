#!/usr/bin/env python3
"""
Debug script to capture actual LLM responses and analyze why tests are failing
"""
import os
import json
import asyncio
import time
from app.core.utils.rest_client import post
from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider
from app.request_models.voice_command_request import CommandDefinition, CommandParameter

def estimate_tokens(text):
    """Rough estimate of token count (approximately 4 characters per token)"""
    return len(text) // 4

def create_sample_commands():
    """Create the same sample commands used in tests"""
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

async def test_llm_response(voice_command, node_context, available_commands, provider, provider_name, request_id=None):
    """Test a single LLM response and measure performance"""
    start_time = time.time()
    model_issues = []
    
    # Build system prompt
    system_prompt = provider.build_system_prompt(node_context, available_commands, voice_command)
    
    prompt_tokens = estimate_tokens(system_prompt)
    command_tokens = estimate_tokens(voice_command)
    input_tokens = prompt_tokens + command_tokens
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": voice_command}
    ]
    
    try:
        llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
        chat_url = f"{llm_api_url}/chat"
        
        llm_response = await post(url=chat_url, json_data={"messages": messages})
        
        end_time = time.time()
        response_time = end_time - start_time
        
        response_content = llm_response.get('content', llm_response.get('response', ''))
        
        # If response_content is still empty, try to extract from other possible fields
        if not response_content:
            # Try other common response formats
            if 'choices' in llm_response and llm_response['choices']:
                response_content = llm_response['choices'][0].get('message', {}).get('content', '')
            elif 'text' in llm_response:
                response_content = llm_response['text']
            elif 'output' in llm_response:
                response_content = llm_response['output']
            elif isinstance(llm_response, str):
                response_content = llm_response
            else:
                # If it's a dict with a single key, try that
                if len(llm_response) == 1:
                    response_content = list(llm_response.values())[0]
        # Handle empty response
        if not response_content:
            print(f"\n⚠️  Warning: Empty response content!")
            model_issues.append("Empty response content")
            response_content = ""
        
        output_tokens = estimate_tokens(response_content)
        total_tokens = input_tokens + output_tokens
        
        print(f"\n{'='*60}")
        print(f"🎤 Voice Command: '{voice_command}'")
        print(f"🧠 Provider: {provider_name}")
        print(f"📊 Performance:")
        print(f"   Response Time: {response_time:.2f}s")
        print(f"   Input Tokens: {input_tokens:,} (System: {prompt_tokens:,}, Command: {command_tokens:,})")
        print(f"   Output Tokens: {output_tokens:,}")
        print(f"   Total Tokens: {total_tokens:,}")
        print(f"   Tokens/Second: {total_tokens/response_time:.1f}")
        
        print(f"\n🤖 LLM Response:")
        print(f"'{response_content}'" if response_content else "❌ EMPTY RESPONSE")
        
        # Try to parse JSON response
        if response_content:
            try:
                # Clean up response content
                clean_content = response_content.strip()
                
                # Parse the JSON response
                parsed_response = json.loads(clean_content)
                
                # Check if it's the new format with 'commands' array
                if 'commands' in parsed_response:
                    commands = parsed_response['commands']
                    print(f"\n✅ Valid JSON Response:")
                    print(f"   Total Commands: {len(commands)}")
                    
                    for i, cmd in enumerate(commands, 1):
                        print(f"   Command {i}:")
                        print(f"     Success: {cmd.get('success')}")
                        print(f"     Command: {cmd.get('command_name')}")
                        print(f"     Parameters: {cmd.get('parameters')}")
                        if cmd.get('errors'):
                            print(f"     Errors: {cmd.get('errors')}")
                else:
                    # Old format - single command
                    print(f"\n✅ Valid JSON Response (legacy format):")
                    print(f"   Success: {parsed_response.get('success')}")
                    print(f"   Command: {parsed_response.get('command_name')}")
                    print(f"   Parameters: {parsed_response.get('parameters')}")
                    if parsed_response.get('errors'):
                        print(f"   Errors: {parsed_response.get('errors')}")
                            
            except json.JSONDecodeError as e:
                print(f"\n❌ Invalid JSON Response: {e}")
                print(f"   Raw content: '{response_content[:200]}...' (truncated)")
                model_issues.append(f"Invalid JSON: {str(e)}")
                
                # Check if this is a case of extra content after JSON
                if response_content.strip().startswith('{'):
                    try:
                        # Try to extract just the JSON part
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
                                print(f"\n🚨 LLM MODEL ISSUE: Extra content detected after JSON!")
                                print(f"   Extra content: {repr(extra_content[:100])}{'...' if len(extra_content) > 100 else ''}")
                                print(f"   JSON portion: {json_end} chars, Total: {len(response_content)} chars")
                                model_issues.append(f"Extra content after JSON: {repr(extra_content[:50])}{'...' if len(extra_content) > 50 else ''}")
                                
                                # Try to parse just the JSON part
                                json_part = response_content[:json_end]
                                parsed_response = json.loads(json_part)
                                print(f"\n✅ Successfully extracted valid JSON from malformed response")
                                
                                # Display the recovered data
                                if "commands" in parsed_response:
                                    print(f"   Total Commands: {len(parsed_response['commands'])}")
                                    for i, cmd in enumerate(parsed_response["commands"], 1):
                                        print(f"   Command {i}:")
                                        print(f"     Success: {cmd.get('success')}")
                                        print(f"     Command: {cmd.get('command_name')}")
                                        print(f"     Parameters: {cmd.get('parameters')}")
                                        if cmd.get('errors'):
                                            print(f"     Errors: {cmd.get('errors')}")
                                else:
                                    print(f"   Success: {parsed_response.get('success')}")
                                    print(f"   Command: {parsed_response.get('command_name')}")
                                    print(f"   Parameters: {parsed_response.get('parameters')}")
                    except:
                        pass
        else:
            print(f"\n❌ Cannot parse JSON - empty response")
            model_issues.append("Cannot parse JSON - empty response")
        
        return {
            "provider": provider_name,
            "voice_command": voice_command,
            "response_time": response_time,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_content": response_content,
            "system_prompt_length": len(system_prompt),
            "system_prompt_tokens": prompt_tokens,
            "model_issues": model_issues,
            "request_id": request_id
        }
        
    except Exception as e:
        print(f"\n❌ Error testing '{voice_command}' with {provider_name}: {e}")
        return {
            "provider": provider_name,
            "voice_command": voice_command,
            "response_time": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "response_content": "",
            "system_prompt_length": 0,
            "system_prompt_tokens": 0,
            "model_issues": [f"Exception: {str(e)}"],
            "request_id": request_id
        }

async def compare_providers():
    """Compare standard provider with and without preprocessing"""
    print("🧪 LLM Response Analysis - Command Preprocessing Test")
    print("=" * 80)
    
    # Test commands
    test_cases = [
        ("turn on the lights", {"room": "bedroom", "node_id": "node-1"}),
        ("set temperature to 72", {"room": "living room", "node_id": "node-2"}),
        ("play some jazz music", {"room": "kitchen", "node_id": "node-3"}),
        ("turn off lights and play music", {"room": "bedroom", "node_id": "node-1"}),
        ("start vacuum", {"room": "", "node_id": "node-4"}),
    ]
    
    available_commands = create_sample_commands()
    
    # Test with preprocessing disabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "false"
    standard_provider = StandardSystemPromptProvider()
    
    # Test with preprocessing enabled
    os.environ["COMMAND_PREPROCESSING_ENABLED"] = "true"
    preprocessing_provider = StandardSystemPromptProvider()
    
    all_results = []
    request_counter = 0
    
    for voice_command, node_context in test_cases:
        print(f"\n{'='*80}")
        print(f"Testing: '{voice_command}' in room '{node_context.get('room', 'N/A')}'")
        print(f"{'='*80}")
        
        # Test without preprocessing
        request_counter += 1
        result1 = await test_llm_response(
            voice_command, 
            node_context, 
            available_commands, 
            standard_provider, 
            "STANDARD (no preprocessing)",
            request_id=f"Request {request_counter}"
        )
        
        # Test with preprocessing
        request_counter += 1
        result2 = await test_llm_response(
            voice_command, 
            node_context, 
            available_commands, 
            preprocessing_provider, 
            "STANDARD (with preprocessing)",
            request_id=f"Request {request_counter}"
        )
        
        if result1 and result2:
            all_results.extend([result1, result2])
            
            # Compare results
            time_diff = result1["response_time"] - result2["response_time"]
            token_diff = result1["input_tokens"] - result2["input_tokens"]
            prompt_length_diff = result1["system_prompt_length"] - result2["system_prompt_length"]
            prompt_token_diff = result1["system_prompt_tokens"] - result2["system_prompt_tokens"]
            
            print(f"\n📈 Comparison:")
            print(f"   Time Difference: {time_diff:+.2f}s ({'faster' if time_diff > 0 else 'slower'} with preprocessing)")
            print(f"   Total Token Difference: {token_diff:+,} ({'fewer' if token_diff > 0 else 'more'} with preprocessing)")
            print(f"   System Prompt Length: {prompt_length_diff:+,} chars ({'shorter' if prompt_length_diff > 0 else 'longer'} with preprocessing)")
            print(f"   System Prompt Tokens: {prompt_token_diff:+,} ({'fewer' if prompt_token_diff > 0 else 'more'} with preprocessing)")
            
            if token_diff > 0:
                print(f"   Total Token Reduction: {(token_diff/result1['input_tokens']*100):.1f}%")
            if prompt_token_diff > 0:
                print(f"   Prompt Token Reduction: {(prompt_token_diff/result1['system_prompt_tokens']*100):.1f}%")
            if prompt_length_diff > 0:
                print(f"   Prompt Length Reduction: {(prompt_length_diff/result1['system_prompt_length']*100):.1f}%")
    
    # Summary statistics
    if all_results:
        print(f"\n{'='*80}")
        print("📊 SUMMARY STATISTICS")
        print(f"{'='*80}")
        
        no_preprocessing = [r for r in all_results if "no preprocessing" in r["provider"]]
        with_preprocessing = [r for r in all_results if "with preprocessing" in r["provider"]]
        
        if no_preprocessing and with_preprocessing:
            # Calculate averages
            avg_time_no_prep = sum(r["response_time"] for r in no_preprocessing) / len(no_preprocessing)
            avg_time_with_prep = sum(r["response_time"] for r in with_preprocessing) / len(with_preprocessing)
            
            avg_tokens_no_prep = sum(r["input_tokens"] for r in no_preprocessing) / len(no_preprocessing)
            avg_tokens_with_prep = sum(r["input_tokens"] for r in with_preprocessing) / len(with_preprocessing)
            
            avg_prompt_length_no_prep = sum(r["system_prompt_length"] for r in no_preprocessing) / len(no_preprocessing)
            avg_prompt_length_with_prep = sum(r["system_prompt_length"] for r in with_preprocessing) / len(with_preprocessing)
            
            avg_prompt_tokens_no_prep = sum(r["system_prompt_tokens"] for r in no_preprocessing) / len(no_preprocessing)
            avg_prompt_tokens_with_prep = sum(r["system_prompt_tokens"] for r in with_preprocessing) / len(with_preprocessing)
            
            print(f"Average Response Time:")
            print(f"   Without Preprocessing: {avg_time_no_prep:.2f}s")
            print(f"   With Preprocessing: {avg_time_with_prep:.2f}s")
            print(f"   Improvement: {avg_time_no_prep - avg_time_with_prep:+.2f}s ({(avg_time_no_prep - avg_time_with_prep)/avg_time_no_prep*100:+.1f}%)")
            
            print(f"\nAverage Input Tokens:")
            print(f"   Without Preprocessing: {avg_tokens_no_prep:,.0f}")
            print(f"   With Preprocessing: {avg_tokens_with_prep:,.0f}")
            print(f"   Reduction: {avg_tokens_no_prep - avg_tokens_with_prep:+,.0f} ({(avg_tokens_no_prep - avg_tokens_with_prep)/avg_tokens_no_prep*100:.1f}%)")
            
            print(f"\nAverage System Prompt Length:")
            print(f"   Without Preprocessing: {avg_prompt_length_no_prep:,.0f} chars")
            print(f"   With Preprocessing: {avg_prompt_length_with_prep:,.0f} chars")
            print(f"   Reduction: {avg_prompt_length_no_prep - avg_prompt_length_with_prep:+,.0f} chars ({(avg_prompt_length_no_prep - avg_prompt_length_with_prep)/avg_prompt_length_no_prep*100:.1f}%)")
            
            print(f"\nAverage System Prompt Tokens:")
            print(f"   Without Preprocessing: {avg_prompt_tokens_no_prep:,.0f} tokens")
            print(f"   With Preprocessing: {avg_prompt_tokens_with_prep:,.0f} tokens")
            print(f"   Reduction: {avg_prompt_tokens_no_prep - avg_prompt_tokens_with_prep:+,.0f} tokens ({(avg_prompt_tokens_no_prep - avg_prompt_tokens_with_prep)/avg_prompt_tokens_no_prep*100:.1f}%)")
            
            # Calculate totals
            total_requests = len(no_preprocessing)
            total_time_no_prep = sum(r["response_time"] for r in no_preprocessing)
            total_time_with_prep = sum(r["response_time"] for r in with_preprocessing)
            total_tokens_no_prep = sum(r["input_tokens"] for r in no_preprocessing)
            total_tokens_with_prep = sum(r["input_tokens"] for r in with_preprocessing)
            total_prompt_chars_no_prep = sum(r["system_prompt_length"] for r in no_preprocessing)
            total_prompt_chars_with_prep = sum(r["system_prompt_length"] for r in with_preprocessing)
            total_prompt_tokens_no_prep = sum(r["system_prompt_tokens"] for r in no_preprocessing)
            total_prompt_tokens_with_prep = sum(r["system_prompt_tokens"] for r in with_preprocessing)
            
            print(f"\nTotal Efficiency Gains:")
            print(f"   Requests Processed: {total_requests}")
            print(f"   Total Time Saved: {total_time_no_prep - total_time_with_prep:.2f}s")
            print(f"   Total Tokens Saved: {total_tokens_no_prep - total_tokens_with_prep:,}")
            print(f"   Total Prompt Chars Saved: {total_prompt_chars_no_prep - total_prompt_chars_with_prep:,}")
            print(f"   Total Prompt Tokens Saved: {total_prompt_tokens_no_prep - total_prompt_tokens_with_prep:,}")
            
            print(f"\nThroughput Comparison:")
            print(f"   Average Tokens/Second (no prep): {total_tokens_no_prep/total_time_no_prep:.1f}")
            print(f"   Average Tokens/Second (with prep): {total_tokens_with_prep/total_time_with_prep:.1f}")
            print(f"   Throughput Improvement: {(total_tokens_with_prep/total_time_with_prep - total_tokens_no_prep/total_time_no_prep):+.1f} tokens/sec")
            
            # Calculate potential savings for larger scale
            print(f"\nProjected Savings (per 1000 requests):")
            time_saved_per_1000 = (total_time_no_prep - total_time_with_prep) * (1000 / total_requests)
            tokens_saved_per_1000 = (total_tokens_no_prep - total_tokens_with_prep) * (1000 / total_requests)
            prompt_tokens_saved_per_1000 = (total_prompt_tokens_no_prep - total_prompt_tokens_with_prep) * (1000 / total_requests)
            
            print(f"   Time Saved: {time_saved_per_1000:.1f}s ({time_saved_per_1000/60:.1f} minutes)")
            print(f"   Tokens Saved: {tokens_saved_per_1000:,.0f}")
            print(f"   Prompt Tokens Saved: {prompt_tokens_saved_per_1000:,.0f}")
            
            # Estimate cost savings (assuming typical token pricing)
            cost_per_1k_tokens = 0.002  # Rough estimate for input tokens
            cost_saved_per_1000_requests = (tokens_saved_per_1000 / 1000) * cost_per_1k_tokens
            print(f"   Estimated Cost Savings: ${cost_saved_per_1000_requests:.4f} (at $0.002/1K tokens)")
    
    # Summary of LLM model issues
    print(f"\n{'='*80}")
    print("🚨 LLM MODEL ISSUES SUMMARY")
    print(f"{'='*80}")
    
    # Collect all model issues
    all_issues = []
    for result in all_results:
        if result and result.get("model_issues"):
            for issue in result["model_issues"]:
                all_issues.append({
                    "request_id": result.get("request_id", "Unknown"),
                    "provider": result.get("provider", "Unknown"),
                    "voice_command": result.get("voice_command", "Unknown"),
                    "issue": issue
                })
    
    if all_issues:
        print(f"Found {len(all_issues)} LLM model issues:")
        
        # Group by request for cleaner output
        issues_by_request = {}
        for issue_data in all_issues:
            request_id = issue_data["request_id"]
            if request_id not in issues_by_request:
                issues_by_request[request_id] = []
            issues_by_request[request_id].append(issue_data)
        
        for request_id, issues in issues_by_request.items():
            provider = issues[0]["provider"]
            voice_command = issues[0]["voice_command"]
            print(f"\n   {request_id} ({provider}):")
            print(f"     Command: '{voice_command}'")
            for issue_data in issues:
                print(f"     Issue: {issue_data['issue']}")
        
        # Summary line
        affected_requests = list(issues_by_request.keys())
        print(f"\n🔍 Summary: {len(all_issues)} LLM model issues found in {len(affected_requests)} requests: {', '.join(affected_requests)}")
        
        # Issue type breakdown
        issue_types = {}
        for issue_data in all_issues:
            issue_type = issue_data["issue"].split(":")[0]  # Get the part before the colon
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        print(f"\n📊 Issue Types:")
        for issue_type, count in sorted(issue_types.items()):
            print(f"   {issue_type}: {count}")
    else:
        print("✅ No LLM model issues detected - all responses were well-formed!")
        print("   All LLM responses contained valid JSON in the expected format.")

if __name__ == "__main__":
    asyncio.run(compare_providers()) 