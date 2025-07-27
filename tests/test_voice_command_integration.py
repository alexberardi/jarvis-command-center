import pytest
import json
import os
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch, AsyncMock


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
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
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
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_successful_command_multiple_parameters(self):
        """Test successful command processing with multiple parameters"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
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
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
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
                assert first_command["errors"] is None
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_missing_room_parameter(self):
        """Test command processing when room parameter is missing"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
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
                    "success": False,
                    "command_name": "turn_on_lights",
                    "parameters": {},
                    "errors": {
                        "type": "missing_parameters",
                        "message": "Missing required parameter: room"
                    }
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
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
                assert get_response_success(data) is False
                
                first_command = extract_first_command(data)
                assert first_command["command_name"] == "turn_on_lights"
                assert first_command["errors"]["type"] == "missing_parameters"
                assert "Missing required parameter: room" in first_command["errors"]["message"]
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_malformed_llm_response(self):
        """Test handling of malformed LLM responses"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
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
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
                # Return malformed response
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": "invalid json response"
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
                # Should handle malformed response gracefully
                assert get_response_success(data) is False
                first_command = extract_first_command(data)
                assert first_command["errors"] is not None
                assert "Failed to parse LLM response" in first_command["errors"]["message"]
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_different_room_contexts(self):
        """Test command processing with different room contexts"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
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
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
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
                assert first_command["errors"] is None
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_system_prompt_generation(self):
        """Test that system prompts are generated correctly"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
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
                    "parameters": {"room": "bedroom", "degrees": 70},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            # Disable transcription cleanup for this test
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            with patch('app.main.post', new_callable=AsyncMock) as mock_post:
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
                    "voice_command": "set temperature to 70 degrees",
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
                assert first_command["parameters"]["degrees"] == 70
                assert first_command["errors"] is None
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Restore transcription cleanup setting
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None) 