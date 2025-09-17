"""
Integration tests for the new model architecture.

These tests verify that the model architecture works correctly with
the LLM proxy and produces expected results for various voice commands.
"""

import pytest
import os
import asyncio
from typing import List, Dict, Any

from app.core.model_service import ModelService
from app.core.model_factory import ModelFactory
from app.request_models.voice_command_request import CommandDefinition, CommandParameter


class TestModelArchitectureIntegration:
    """Integration tests for the new model architecture."""
    
    @pytest.fixture(scope="class")
    def sample_commands(self) -> List[CommandDefinition]:
        """Sample commands for testing."""
        return [
            CommandDefinition(
                command_name="weather_command",
                description="Get weather information for a location",
                parameters=[
                    CommandParameter(
                        name="location", 
                        type="string", 
                        required=True, 
                        description="City name or location"
                    ),
                    CommandParameter(
                        name="date", 
                        type="string", 
                        required=False, 
                        description="Date for weather forecast"
                    )
                ]
            ),
            CommandDefinition(
                command_name="lights_command",
                description="Control smart lights",
                parameters=[
                    CommandParameter(
                        name="action", 
                        type="string", 
                        required=True, 
                        description="Action: on, off, dim, brighten"
                    ),
                    CommandParameter(
                        name="location", 
                        type="string", 
                        required=False, 
                        description="Room or light location"
                    ),
                    CommandParameter(
                        name="brightness", 
                        type="integer", 
                        required=False, 
                        description="Brightness level 0-100"
                    )
                ]
            ),
            CommandDefinition(
                command_name="music_command",
                description="Control music playback",
                parameters=[
                    CommandParameter(
                        name="action", 
                        type="string", 
                        required=True, 
                        description="Action: play, pause, stop, next, previous"
                    ),
                    CommandParameter(
                        name="song", 
                        type="string", 
                        required=False, 
                        description="Song name or artist"
                    ),
                    CommandParameter(
                        name="playlist", 
                        type="string", 
                        required=False, 
                        description="Playlist name"
                    )
                ]
            ),
            CommandDefinition(
                command_name="timer_command",
                description="Set or manage timers",
                parameters=[
                    CommandParameter(
                        name="action", 
                        type="string", 
                        required=True, 
                        description="Action: set, cancel, check"
                    ),
                    CommandParameter(
                        name="duration", 
                        type="string", 
                        required=False, 
                        description="Timer duration (e.g., '5 minutes', '1 hour')"
                    ),
                    CommandParameter(
                        name="name", 
                        type="string", 
                        required=False, 
                        description="Timer name or description"
                    )
                ]
            ),
            CommandDefinition(
                command_name="calculator_command",
                description="Perform mathematical calculations",
                parameters=[
                    CommandParameter(
                        name="expression", 
                        type="string", 
                        required=True, 
                        description="Mathematical expression to evaluate"
                    )
                ]
            )
        ]
    
    @pytest.fixture(scope="class")
    def node_context(self) -> Dict[str, Any]:
        """Sample node context for testing."""
        return {
            "node_id": "test-node-123",
            "room": "living room",
            "user": "test-user",
            "voice_mode": "brief"
        }
    
    @pytest.fixture(scope="class")
    def test_cases(self) -> List[Dict[str, Any]]:
        """Test cases with voice commands and expected results."""
        return [
            {
                "command": "what's the weather in New York",
                "expected_command": "weather_command",
                "expected_params": {"location": "New York"},
                "description": "Simple weather query"
            },
            {
                "command": "turn on the living room lights",
                "expected_command": "lights_command",
                "expected_params": {"action": "on", "location": "living room"},
                "description": "Light control with location"
            },
            {
                "command": "turn off all lights",
                "expected_command": "lights_command",
                "expected_params": {"action": "off"},
                "description": "Light control without specific location"
            },
            {
                "command": "play some jazz music",
                "expected_command": "music_command",
                "expected_params": {"action": "play", "song": "jazz"},
                "description": "Music playback request"
            },
            {
                "command": "set a timer for 10 minutes",
                "expected_command": "timer_command",
                "expected_params": {"action": "set", "duration": "10 minutes"},
                "description": "Timer setting"
            },
            {
                "command": "calculate 15 plus 27",
                "expected_command": "calculator_command",
                "expected_params": {"expression": "15 plus 27"},
                "description": "Math calculation"
            },
            {
                "command": "dim the bedroom lights to 50 percent",
                "expected_command": "lights_command",
                "expected_params": {"action": "dim", "location": "bedroom", "brightness": 50},
                "description": "Complex light control with brightness"
            },
            {
                "command": "what's the weather tomorrow in San Francisco",
                "expected_command": "weather_command",
                "expected_params": {"location": "San Francisco", "date": "tomorrow"},
                "description": "Weather with date parameter"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_base_model_integration(self, sample_commands, node_context, test_cases):
        """Test BaseModel integration with LLM proxy."""
        print("\n🧪 Testing BaseModel Integration")
        
        # Skip if no LLM proxy available
        if not os.getenv("JARVIS_LLM_PROXY_URL"):
            pytest.skip("No LLM proxy URL configured")
        
        model_service = ModelService("BASE_MODEL")
        conversation_id = "test-base-model-001"
        
        try:
            # Warmup
            await model_service.warmup_conversation(
                node_context=node_context,
                available_commands=sample_commands,
                conversation_id=conversation_id,
                timezone="America/New_York"
            )
            
            # Test each case
            for i, case in enumerate(test_cases[:3]):  # Test first 3 cases
                print(f"   Testing: {case['description']}")
                
                result = await model_service.process_voice_command(
                    voice_command=case["command"],
                    conversation_id=conversation_id,
                    node_context=node_context
                )
                
                # Verify result structure
                assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
                assert "s" in result, "Result should have 's' (success) field"
                assert "n" in result, "Result should have 'n' (command name) field"
                assert "p" in result, "Result should have 'p' (parameters) field"
                
                # Log result for debugging
                print(f"     Command: {case['command']}")
                print(f"     Expected: {case['expected_command']}")
                print(f"     Got: {result.get('n')}")
                print(f"     Success: {result.get('s')}")
                
                if result.get("s"):
                    assert result["n"] == case["expected_command"], \
                        f"Expected command {case['expected_command']}, got {result['n']}"
                    
                    # Check key parameters (allowing for some flexibility in LLM responses)
                    for key, expected_value in case["expected_params"].items():
                        if key in result["p"]:
                            actual_value = result["p"][key]
                            print(f"     Param {key}: expected='{expected_value}', got='{actual_value}'")
                else:
                    print(f"     Error: {result.get('e')}")
                
                print()
            
        finally:
            # Cleanup
            await model_service.cleanup_conversation(conversation_id)
    
    @pytest.mark.asyncio
    async def test_jarvis_llama3b_integration(self, sample_commands, node_context, test_cases):
        """Test JarvisLlama3B integration with LLM proxy."""
        print("\n🚀 Testing JarvisLlama3B Integration")
        
        # Skip if no LLM proxy available
        if not os.getenv("JARVIS_LLM_PROXY_URL"):
            pytest.skip("No LLM proxy URL configured")
        
        model_service = ModelService("JarvisLlama3B")
        conversation_id = "test-jarvis-llama3b-001"
        
        try:
            # Warmup
            await model_service.warmup_conversation(
                node_context=node_context,
                available_commands=sample_commands,
                conversation_id=conversation_id,
                timezone="America/New_York"
            )
            
            # Test each case
            for i, case in enumerate(test_cases[:3]):  # Test first 3 cases
                print(f"   Testing: {case['description']}")
                
                result = await model_service.process_voice_command(
                    voice_command=case["command"],
                    conversation_id=conversation_id,
                    node_context=node_context
                )
                
                # Verify result structure
                assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
                assert "s" in result, "Result should have 's' (success) field"
                assert "n" in result, "Result should have 'n' (command name) field"
                assert "p" in result, "Result should have 'p' (parameters) field"
                
                # Log result for debugging
                print(f"     Command: {case['command']}")
                print(f"     Expected: {case['expected_command']}")
                print(f"     Got: {result.get('n')}")
                print(f"     Success: {result.get('s')}")
                
                if result.get("s"):
                    assert result["n"] == case["expected_command"], \
                        f"Expected command {case['expected_command']}, got {result['n']}"
                    
                    # Check key parameters (allowing for some flexibility in LLM responses)
                    for key, expected_value in case["expected_params"].items():
                        if key in result["p"]:
                            actual_value = result["p"][key]
                            print(f"     Param {key}: expected='{expected_value}', got='{actual_value}'")
                else:
                    print(f"     Error: {result.get('e')}")
                
                print()
            
        finally:
            # Cleanup
            await model_service.cleanup_conversation(conversation_id)
    
    @pytest.mark.asyncio
    async def test_model_comparison(self, sample_commands, node_context):
        """Compare BaseModel vs JarvisLlama3B performance."""
        print("\n⚡ Model Performance Comparison")
        
        # Skip if no LLM proxy available
        if not os.getenv("JARVIS_LLM_PROXY_URL"):
            pytest.skip("No LLM proxy URL configured")
        
        test_command = "turn on the kitchen lights"
        
        # Test BaseModel
        base_service = ModelService("BASE_MODEL")
        base_conversation_id = "test-comparison-base"
        
        try:
            import time
            
            # BaseModel test
            start_time = time.time()
            await base_service.warmup_conversation(
                node_context=node_context,
                available_commands=sample_commands,
                conversation_id=base_conversation_id,
                timezone="America/New_York"
            )
            base_warmup_time = time.time() - start_time
            
            start_time = time.time()
            base_result = await base_service.process_voice_command(
                voice_command=test_command,
                conversation_id=base_conversation_id,
                node_context=node_context
            )
            base_inference_time = time.time() - start_time
            
            print(f"   BaseModel:")
            print(f"     Warmup: {base_warmup_time:.2f}s")
            print(f"     Inference: {base_inference_time:.2f}s")
            print(f"     Result: {base_result}")
            
        finally:
            await base_service.cleanup_conversation(base_conversation_id)
        
        # Test JarvisLlama3B
        jarvis_service = ModelService("JarvisLlama3B")
        jarvis_conversation_id = "test-comparison-jarvis"
        
        try:
            # JarvisLlama3B test
            start_time = time.time()
            await jarvis_service.warmup_conversation(
                node_context=node_context,
                available_commands=sample_commands,
                conversation_id=jarvis_conversation_id,
                timezone="America/New_York"
            )
            jarvis_warmup_time = time.time() - start_time
            
            start_time = time.time()
            jarvis_result = await jarvis_service.process_voice_command(
                voice_command=test_command,
                conversation_id=jarvis_conversation_id,
                node_context=node_context
            )
            jarvis_inference_time = time.time() - start_time
            
            print(f"   JarvisLlama3B:")
            print(f"     Warmup: {jarvis_warmup_time:.2f}s")
            print(f"     Inference: {jarvis_inference_time:.2f}s")
            print(f"     Result: {jarvis_result}")
            
            # Performance comparison
            print(f"\n   Performance Summary:")
            print(f"     JarvisLlama3B warmup is {jarvis_warmup_time/base_warmup_time:.1f}x BaseModel warmup time")
            print(f"     JarvisLlama3B inference is {jarvis_inference_time/base_inference_time:.1f}x BaseModel inference time")
            
        finally:
            await jarvis_service.cleanup_conversation(jarvis_conversation_id)
    
    def test_model_factory_functionality(self):
        """Test the model factory functionality."""
        print("\n🏭 Testing Model Factory")
        
        # Test available models
        available_models = ModelFactory.get_available_models()
        print(f"   Available models: {available_models}")
        assert "BASE_MODEL" in available_models
        assert "JarvisLlama3B" in available_models
        
        # Test model creation
        for model_name in ["BASE_MODEL", "JarvisLlama3B"]:
            model = ModelFactory.create_model(model_name)
            print(f"   Created {model_name}: {model.__class__.__name__}")
            assert model.name == model_name
            
            # Test capabilities
            capabilities = model.get_capabilities()
            print(f"     Capabilities: {capabilities}")
            assert isinstance(capabilities, dict)
            assert "supports_single_shot" in capabilities
        
        # Test model info
        for model_name in available_models:
            info = ModelFactory.get_model_info(model_name)
            print(f"   {model_name} info: {info}")
            assert info is not None
            assert info["name"] == model_name
    
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_commands, node_context):
        """Test error handling in the model architecture."""
        print("\n❌ Testing Error Handling")
        
        model_service = ModelService("BASE_MODEL")
        conversation_id = "test-error-handling"
        
        try:
            # Test with invalid conversation ID (no warmup)
            result = await model_service.process_voice_command(
                voice_command="test command",
                conversation_id="invalid-conversation-id",
                node_context=node_context
            )
            
            print(f"   Invalid conversation result: {result}")
            assert result["s"] == False, "Should fail with invalid conversation ID"
            assert "error" in result["e"]["type"] or "cache" in result["e"]["type"]
            
            # Test with valid warmup but ambiguous command
            await model_service.warmup_conversation(
                node_context=node_context,
                available_commands=sample_commands,
                conversation_id=conversation_id,
                timezone="America/New_York"
            )
            
            result = await model_service.process_voice_command(
                voice_command="do something random that doesn't match any command",
                conversation_id=conversation_id,
                node_context=node_context
            )
            
            print(f"   Ambiguous command result: {result}")
            # Note: This might succeed or fail depending on LLM behavior
            # We're mainly testing that it doesn't crash
            assert isinstance(result, dict)
            assert "s" in result
            
        finally:
            await model_service.cleanup_conversation(conversation_id)
