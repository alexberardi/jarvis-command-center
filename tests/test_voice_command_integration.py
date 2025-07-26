import pytest
import json
from fastapi.testclient import TestClient
from app.main import app


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


class TestVoiceCommandIntegration:
    """Test voice command integration with mocked LLM responses"""
    
    def test_successful_command_with_room_context(self):
        """Test successful command processing with room context"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        
        # Create a mock node
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="living room",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        # Mock the LLM response
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(mock_llm_response)
                            }
                        }
                    ]
                }
                
                client = TestClient(app)
                response = client.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "turn_on_lights"
                assert first_command["parameters"]["room"] == "living room"
                assert first_command["errors"] is None
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_successful_command_multiple_parameters(self):
        """Test successful command processing with multiple parameters"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        
        # Create a mock node
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="bedroom",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        # Mock the LLM response
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "set_temperature",
                    "parameters": {"room": "bedroom", "degrees": 72},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(mock_llm_response)
                            }
                        }
                    ]
                }
                
                client = TestClient(app)
                response = client.post("/api/v0/voice/command", json={
                    "voice_command": "set temperature to 72 degrees",
                    "node_context": {"room": "bedroom", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "set_temperature"
                assert first_command["parameters"]["room"] == "bedroom"
                assert first_command["parameters"]["degrees"] == 72
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_missing_room_parameter(self):
        """Test handling of missing room parameter"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        
        # Create a mock node with empty room
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        # Mock the LLM response for missing room parameter
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "missing_parameters",
                        "message": "Room parameter is required",
                        "missing_parameters": ["room"],
                        "clarification_question": "Which room would you like to control?"
                    }
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(mock_llm_response)
                            }
                        }
                    ]
                }
                
                client = TestClient(app)
                response = client.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is False
                
                errors = get_response_errors(data)
                assert errors["type"] == "missing_parameters"
                assert "room" in errors["missing_parameters"]
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_malformed_llm_response(self):
        """Test handling of malformed LLM response"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        
        # Create a mock node
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="living room",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                # Return malformed JSON
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": "This is not valid JSON"
                            }
                        }
                    ]
                }
                
                client = TestClient(app)
                response = client.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is False
                
                errors = get_response_errors(data)
                assert errors["type"] == "parsing_error"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_different_room_contexts(self):
        """Test that different room contexts are handled correctly"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        
        # Create a mock node
        mock_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="kitchen",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        # Mock the LLM response
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "kitchen"},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(mock_llm_response)
                            }
                        }
                    ]
                }
                
                client = TestClient(app)
                response = client.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "kitchen", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "turn_on_lights"
                assert first_command["parameters"]["room"] == "kitchen"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_system_prompt_generation(self):
        """Test that system prompts are generated correctly"""
        from app.context_providers.standard_system_prompt_provider import StandardCommandInferenceSystemPromptProvider
        from app.request_models.voice_command_request import CommandDefinition, CommandParameter
        
        # Test with room context
        provider = StandardCommandInferenceSystemPromptProvider()
        commands = [
            CommandDefinition(
                command_name="turn_on_lights",
                description="Turn on lights",
                parameters=[CommandParameter(name="room", type="str")]
            )
        ]
        
        node_context = {"room": "living room", "node_id": "test-node"}
        prompt = provider.build_system_prompt(node_context, commands)
        
        # Check that the prompt contains expected elements
        assert "living room" in prompt
        assert "turn_on_lights" in prompt
        assert "room" in prompt 