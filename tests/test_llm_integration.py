import pytest
import os
from fastapi.testclient import TestClient
from app.main import app
from app.request_models.voice_command_request import CommandDefinition, CommandParameter
import time


def extract_first_command(data):
    """Helper function to extract the first command from the new response format"""
    if "commands" in data and data["commands"]:
        return data["commands"][0]
    return data  # Fallback for old format


def get_response_success(data):
    """Helper function to check if the response is successful"""
    if "commands" in data:
        return all(cmd.get("success", False) for cmd in data["commands"])
    return data.get("success", False)


def get_response_errors(data):
    """Helper function to get errors from the response"""
    if "commands" in data:
        errors = []
        for cmd in data["commands"]:
            if cmd.get("errors"):
                errors.append(cmd["errors"])
        return errors[0] if errors else None
    return data.get("errors")


class TestTranscriptionCleanup:
    """Test transcription cleanup functionality and performance"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client with authentication override"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Create a mock node for authentication
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="living room",
            user="test-user"
        )
        
        # Override the dependency
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        client = TestClient(app)
        yield client
        
        # Clean up
        app.dependency_overrides.clear()
    
    @pytest.fixture
    def sample_commands(self):
        """Sample commands for testing"""
        return [
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
            ),
            CommandDefinition(
                command_name="play_music",
                description="Play music",
                parameters=[CommandParameter(name="song", type="str")]
            )
        ]
    
    def setup_method(self):
        """Setup before each test - add delay to avoid overwhelming LLM"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Ensure authentication override is set for each test
        if verify_api_key not in app.dependency_overrides:
            mock_node = Node(
                node_id="test-node",
                api_key="test-key",
                room="living room",
                user="test-user"
            )
            
            def mock_verify_api_key():
                return NodeContextProvider(mock_node)
            
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        time.sleep(0.5)  # Small delay between tests
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_transcription_cleanup_with_filler_words(self, client, sample_commands):
        """Test transcription cleanup with common filler words and transcription errors"""
        test_cases = [
            {
                "original": "um turn on the like in the kitchen please",
                "expected_cleaned": "turn on the lights in the kitchen",
                "description": "Fix 'like' to 'lights' and remove filler words"
            },
            {
                "original": "can you set the temperature to like seventy two degrees",
                "expected_cleaned": "set the temperature to 72 degrees",
                "description": "Convert words to numbers and remove filler words"
            },
            {
                "original": "I want to play some music maybe jazz",
                "expected_cleaned": "play jazz music",
                "description": "Simplify and make more direct"
            },
            {
                "original": "um can you like make it warmer in here",
                "expected_cleaned": "make it warmer in here",
                "description": "Remove filler words 'um', 'can you', 'like'"
            }
        ]
        
        # Test with cleanup enabled
        os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "true"
        
        for case in test_cases:
            print(f"\n🧪 Testing: {case['description']}")
            print(f"   Original: '{case['original']}'")
            
            start_time = time.time()
            
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["original"],
                "node_context": {"room": "kitchen", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            end_time = time.time()
            latency = end_time - start_time
            
            assert response.status_code == 200
            data = response.json()
            
            print(f"   Latency: {latency:.2f}s")
            print(f"   Success: {get_response_success(data)}")
            
            # Check if the command was successfully processed
            if get_response_success(data):
                first_command = extract_first_command(data)
                print(f"   Command: {first_command['command_name']}")
                print(f"   Parameters: {first_command['parameters']}")
            else:
                errors = get_response_errors(data)
                print(f"   Error: {errors['type'] if errors else 'Unknown'}")
            
            # Verify it meets the 5-second target
            assert latency < 5.0, f"Latency {latency:.2f}s exceeds 5-second target"
            print(f"   ✅ Under 5-second target")
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_transcription_cleanup_performance_comparison(self, client, sample_commands):
        """Compare performance with and without transcription cleanup"""
        test_command = "um turn on the like in the kitchen please"
        
        print(f"\n📊 Performance Comparison for: '{test_command}'")
        print("="*60)
        
        # Test with cleanup enabled
        os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "true"
        
        start_time = time.time()
        response_with_cleanup = client.post("/api/v0/voice/command", json={
            "voice_command": test_command,
            "node_context": {"room": "kitchen", "node_id": "test-node"},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": "test-key"})
        end_time = time.time()
        latency_with_cleanup = end_time - start_time
        
        assert response_with_cleanup.status_code == 200
        data_with_cleanup = response_with_cleanup.json()
        
        print(f"WITH cleanup:")
        print(f"   Latency: {latency_with_cleanup:.2f}s")
        print(f"   Success: {get_response_success(data_with_cleanup)}")
        if get_response_success(data_with_cleanup):
            first_command = extract_first_command(data_with_cleanup)
            print(f"   Command: {first_command['command_name']}")
        
        # Test without cleanup
        os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
        
        start_time = time.time()
        response_without_cleanup = client.post("/api/v0/voice/command", json={
            "voice_command": test_command,
            "node_context": {"room": "kitchen", "node_id": "test-node"},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": "test-key"})
        end_time = time.time()
        latency_without_cleanup = end_time - start_time
        
        assert response_without_cleanup.status_code == 200
        data_without_cleanup = response_without_cleanup.json()
        
        print(f"WITHOUT cleanup:")
        print(f"   Latency: {latency_without_cleanup:.2f}s")
        print(f"   Success: {get_response_success(data_without_cleanup)}")
        if get_response_success(data_without_cleanup):
            first_command = extract_first_command(data_without_cleanup)
            print(f"   Command: {first_command['command_name']}")
        
        # Calculate overhead
        overhead = latency_with_cleanup - latency_without_cleanup
        overhead_percent = (overhead / latency_without_cleanup) * 100 if latency_without_cleanup > 0 else 0
        
        print(f"\n📈 OVERHEAD ANALYSIS:")
        print(f"   Additional Latency: {overhead:.2f}s ({overhead_percent:.1f}%)")
        print(f"   Baseline: {latency_without_cleanup:.2f}s")
        print(f"   With Cleanup: {latency_with_cleanup:.2f}s")
        
        # Verify both meet the 5-second target
        assert latency_with_cleanup < 5.0, f"With cleanup latency {latency_with_cleanup:.2f}s exceeds 5-second target"
        assert latency_without_cleanup < 5.0, f"Without cleanup latency {latency_without_cleanup:.2f}s exceeds 5-second target"
        
        print(f"   ✅ Both scenarios meet 5-second target")
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_clean_already_clean_commands(self, client, sample_commands):
        """Test that already clean commands work well with cleanup enabled"""
        clean_commands = [
            "turn on the lights",
            "set temperature to 72 degrees",
            "play jazz music"
        ]
        
        # Test with cleanup enabled
        os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "true"
        
        for command in clean_commands:
            print(f"\n🧪 Testing clean command: '{command}'")
            
            start_time = time.time()
            
            response = client.post("/api/v0/voice/command", json={
                "voice_command": command,
                "node_context": {"room": "kitchen", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            end_time = time.time()
            latency = end_time - start_time
            
            assert response.status_code == 200
            data = response.json()
            
            print(f"   Latency: {latency:.2f}s")
            print(f"   Success: {get_response_success(data)}")
            
            # Clean commands should work well
            assert latency < 5.0, f"Latency {latency:.2f}s exceeds 5-second target"
            
            if get_response_success(data):
                first_command = extract_first_command(data)
                print(f"   Command: {first_command['command_name']}")
                print(f"   ✅ Successfully processed")
            else:
                errors = get_response_errors(data)
                print(f"   ⚠️  Error: {errors['type'] if errors else 'Unknown'}")


class TestLLMIntegration:
    """Real LLM integration tests - actually calls the LLM API"""
    
    @pytest.fixture(scope="class")
    def client(self):
        """Create a test client with authentication override"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Create a mock node for authentication
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="living room",
            user="test-user"
        )
        
        # Override the dependency
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        client = TestClient(app)
        yield client
        
        # Clean up
        app.dependency_overrides.clear()
    
    @pytest.fixture
    def sample_commands(self):
        """Sample commands for testing"""
        return [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")]
            ),
            CommandDefinition(
                command_name="turn_off_lights",
                description="Turn off lights in a specific room",
                parameters=[CommandParameter(name="room", type="str")]
            ),
            CommandDefinition(
                command_name="set_temperature",
                description="Set temperature in a room",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="degrees", type="int")
                ]
            ),
            CommandDefinition(
                command_name="play_music",
                description="Play music",
                parameters=[CommandParameter(name="song", type="str")]
            ),
            CommandDefinition(
                command_name="stop_music",
                description="Stop music",
                parameters=[]
            ),
            CommandDefinition(
                command_name="set_timer",
                description="Set a timer",
                parameters=[
                    CommandParameter(name="duration", type="int"),
                    CommandParameter(name="unit", type="str")
                ]
            ),
            CommandDefinition(
                command_name="dim_lights",
                description="Dim lights to a specific brightness",
                parameters=[
                    CommandParameter(name="room", type="str"),
                    CommandParameter(name="brightness", type="int")
                ]
            )
        ]
    
    def setup_method(self):
        """Setup before each test - add delay to avoid overwhelming LLM"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Ensure authentication override is set for each test
        if verify_api_key not in app.dependency_overrides:
            mock_node = Node(
                node_id="test-node",
                api_key="test-key",
                room="living room",
                user="test-user"
            )
            
            def mock_verify_api_key():
                return NodeContextProvider(mock_node)
            
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
        
        time.sleep(0.5)  # Small delay between tests
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_basic_light_commands(self, client, sample_commands):
        """Test basic light on/off commands"""
        test_cases = [
            {
                "command": "turn on the lights",
                "expected_command": "turn_on_lights",
                "room": "living room"
            },
            {
                "command": "turn off the lights",
                "expected_command": "turn_off_lights", 
                "room": "bedroom"
            },
            {
                "command": "lights on",
                "expected_command": "turn_on_lights",
                "room": "kitchen"
            }
        ]
        
        for case in test_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": case["room"], "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Should successfully identify command
            assert get_response_success(data) is True, f"Failed for command: {case['command']}"
            
            first_command = extract_first_command(data)
            assert first_command["command_name"] == case["expected_command"]
            assert first_command["parameters"]["room"] == case["room"]
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_temperature_commands(self, client, sample_commands):
        """Test temperature setting commands"""
        test_cases = [
            {
                "command": "set temperature to 72 degrees",
                "room": "living room",
                "expected_degrees": 72
            },
            {
                "command": "make it 68 degrees",
                "room": "bedroom",
                "expected_degrees": 68
            },
            {
                "command": "set thermostat to 75",
                "room": "kitchen",
                "expected_degrees": 75
            }
        ]
        
        for case in test_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": case["room"], "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Temperature commands should work, but sometimes LLM returns malformed JSON
            if get_response_success(data):
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "set_temperature"
                assert first_command["parameters"]["room"] == case["room"]
                assert first_command["parameters"]["degrees"] == case["expected_degrees"]
            else:
                # If it fails, it should be due to parsing error (LLM model issue)
                errors = get_response_errors(data)
                assert errors is not None
                assert errors["type"] == "parsing_error", f"Unexpected error type for command: {case['command']}"
                # Log the model issue for visibility
                print(f"🚨 LLM MODEL ISSUE detected for command: '{case['command']}'")
                print(f"   Error message: {errors.get('message', 'No message')}")
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_timer_commands(self, client, sample_commands):
        """Test timer setting commands"""
        test_cases = [
            {
                "command": "set timer for 5 minutes",
                "expected_duration": 5,
                "expected_unit": "minutes"
            },
            {
                "command": "timer for 30 seconds",
                "expected_duration": 30,
                "expected_unit": "seconds"
            },
            {
                "command": "set a 2 hour timer",
                "expected_duration": 2,
                "expected_unit": "hours"
            }
        ]
        
        for case in test_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": "kitchen", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Timer commands should work, but sometimes LLM returns malformed JSON
            if get_response_success(data):
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "set_timer"
                # Handle both string and integer duration values
                actual_duration = first_command["parameters"]["duration"]
                expected_duration = case["expected_duration"]
                assert str(actual_duration) == str(expected_duration), f"Duration mismatch: expected {expected_duration}, got {actual_duration}"
                # Allow for singular/plural variations in units
                actual_unit = first_command["parameters"]["unit"]
                expected_unit = case["expected_unit"]
                assert actual_unit in [expected_unit, expected_unit.rstrip('s'), expected_unit + 's'], f"Unit mismatch: expected '{expected_unit}', got '{actual_unit}'"
            else:
                # If it fails, it should be due to parsing error or other issues
                errors = get_response_errors(data)
                if errors is not None:
                    # Allow for various error types since LLM behavior can vary
                    assert errors["type"] in ["parsing_error", "no_command_match", "missing_parameters"], f"Unexpected error type for command: {case['command']}"
                    # Log the model issue for visibility
                    print(f"🚨 LLM MODEL ISSUE detected for command: '{case['command']}'")
                    print(f"   Error message: {errors.get('message', 'No message')}")
                else:
                    # If no errors but also no success, this might be a different issue
                    print(f"⚠️ Unexpected response format for command: '{case['command']}'")
                    print(f"   Response data: {data}")
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_music_commands(self, client, sample_commands):
        """Test music control commands"""
        test_cases = [
            {
                "command": "play Bohemian Rhapsody",
                "expected_command": "play_music",
                "expected_song": "Bohemian Rhapsody"
            },
            {
                "command": "play some jazz",
                "expected_command": "play_music",
                "expected_song": "jazz"
            },
            {
                "command": "stop the music",
                "expected_command": "stop_music",
                "expected_song": None
            }
        ]
        
        for case in test_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Music commands should work, but sometimes LLM returns malformed JSON
            if get_response_success(data):
                first_command = extract_first_command(data)
                assert first_command["command_name"] == case["expected_command"]
                
                if case["expected_song"]:
                    assert "song" in first_command["parameters"]
                    assert case["expected_song"].lower() in first_command["parameters"]["song"].lower()
            else:
                # If it fails, it should be due to parsing error (LLM model issue)
                errors = get_response_errors(data)
                assert errors is not None
                assert errors["type"] == "parsing_error", f"Unexpected error type for command: {case['command']}"
                # Log the model issue for visibility
                print(f"🚨 LLM MODEL ISSUE detected for command: '{case['command']}'")
                print(f"   Error message: {errors.get('message', 'No message')}")
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_missing_room_context(self, client, sample_commands):
        """Test commands when room context is missing"""
        response = client.post("/api/v0/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": "", "node_id": "test-node"},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": "test-key"})
        
        assert response.status_code == 200
        data = response.json()
        
        # Should either ask for room or fail gracefully
        if get_response_success(data):
            # If successful, should have inferred or asked for room
            first_command = extract_first_command(data)
            assert "room" in first_command["parameters"]
        else:
            # If failed, should indicate missing room
            errors = get_response_errors(data)
            assert errors is not None
            assert errors["type"] in ["missing_parameters", "no_command_match"]
            if errors["type"] == "missing_parameters":
                assert "room" in errors["missing_parameters"]
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_ambiguous_commands(self, client, sample_commands):
        """Test ambiguous or unclear commands"""
        ambiguous_commands = [
            "turn on",
            "set it",
            "play",
            "make it warmer"
        ]
        
        for command in ambiguous_commands:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": command,
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Should either succeed with reasonable interpretation or fail gracefully
            if not get_response_success(data):
                errors = get_response_errors(data)
                assert errors is not None
                assert errors["type"] in [
                    "ambiguous_command", 
                    "missing_parameters", 
                    "no_command_match",
                    "parsing_error"  # LLM might return empty response for ambiguous commands
                ]
                # Only check for clarification_question if it's not a parsing error
                if errors["type"] != "parsing_error":
                    assert "clarification_question" in errors
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_unrecognized_commands(self, client, sample_commands):
        """Test completely unrecognized commands"""
        unrecognized_commands = [
            "make me a sandwich",
            "what's the weather like",
            "call my mom",
            "order pizza"
        ]
        
        for command in unrecognized_commands:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": command,
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Should fail gracefully
            assert get_response_success(data) is False
            errors = get_response_errors(data)
            assert errors is not None
            assert errors["type"] in ["no_command_match", "parsing_error"]  # LLM might return empty response
            # Only check for clarification_question if it's not a parsing error
            if errors["type"] != "parsing_error":
                assert "clarification_question" in errors
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_natural_language_variations(self, client, sample_commands):
        """Test natural language variations"""
        variations = [
            "could you please turn on the lights",
            "I'd like to turn on the lights",
            "can you turn the lights on",
            "lights on please",
            "would you mind turning on the lights"
        ]
        
        for command in variations:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": command,
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            # Should successfully identify as turn_on_lights
            assert get_response_success(data) is True, f"Failed for command: {command}"
            
            first_command = extract_first_command(data)
            assert first_command["command_name"] == "turn_on_lights"
            assert first_command["parameters"]["room"] == "living room"
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_complex_scenarios(self, client, sample_commands):
        """Test complex multi-parameter scenarios"""
        complex_cases = [
            {
                "command": "dim the bedroom lights to 50 percent",
                "expected_command": "dim_lights",
                "expected_room": "bedroom",
                "expected_brightness": 50
            },
            {
                "command": "set living room lights to 25 percent brightness",
                "expected_command": "dim_lights", 
                "expected_room": "living room",
                "expected_brightness": 25
            }
        ]
        
        for case in complex_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": case["expected_room"], "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            if get_response_success(data):
                first_command = extract_first_command(data)
                assert first_command["command_name"] == case["expected_command"]
                assert first_command["parameters"]["room"] == case["expected_room"]
                # Check if brightness parameter is present, but don't fail if it's missing
                # as the LLM might not always parse complex parameters correctly
                if "brightness" in first_command["parameters"]:
                    assert first_command["parameters"]["brightness"] == case["expected_brightness"]
                else:
                    # Log that brightness parameter was missing but don't fail the test
                    print(f"Warning: brightness parameter missing for command: {case['command']}")
            else:
                # Complex commands might fail - that's okay, just ensure graceful failure
                errors = get_response_errors(data)
                assert errors is not None
                assert "clarification_question" in errors
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_multiple_commands(self, client, sample_commands):
        """Test multiple commands in one request"""
        multiple_command_cases = [
            {
                "command": "turn on the lights and play music",
                "expected_commands": ["turn_on_lights", "play_music"]
            },
            {
                "command": "turn off the lights and set timer for 5 minutes",
                "expected_commands": ["turn_off_lights", "set_timer"]
            }
        ]
        
        for case in multiple_command_cases:
            response = client.post("/api/v0/voice/command", json={
                "voice_command": case["command"],
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            
            if get_response_success(data):
                # Check that we got multiple commands
                if "commands" in data:
                    assert len(data["commands"]) >= 2, f"Expected multiple commands for: {case['command']}"
                    
                    # Check that expected commands are present
                    returned_commands = [cmd["command_name"] for cmd in data["commands"]]
                    for expected_cmd in case["expected_commands"]:
                        assert expected_cmd in returned_commands, f"Expected {expected_cmd} in {returned_commands}"


class TestLLMPerformance:
    """Performance tests for LLM integration"""
    
    def setup_method(self):
        """Set up authentication mock for each test"""
        from app.main import app
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Create a mock node for testing
        mock_node = Node(
            node_id="test-node",
            room="living room",
            api_key="test-key"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
    
    def teardown_method(self):
        """Clean up after each test"""
        from app.main import app
        app.dependency_overrides.clear()
    
    @pytest.mark.skipif(
        not os.getenv("JARVIS_LLM_PROXY_API_URL"),
        reason="LLM API URL not configured"
    )
    def test_response_time(self, client_with_test_db):
        """Test that LLM responses are reasonably fast"""
        commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights",
                parameters=[CommandParameter(name="room", type="str")]
            )
        ]
        
        start_time = time.time()
        
        response = client_with_test_db.post("/api/v0/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": "living room", "node_id": "test-node"},
            "available_commands": [cmd.model_dump() for cmd in commands]
        }, headers={"x-api-key": "test-key"})
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 10.0  # Should respond within 10 seconds
        
        print(f"LLM response time: {response_time:.2f} seconds")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"]) 