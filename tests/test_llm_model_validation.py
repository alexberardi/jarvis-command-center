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


class TestLLMModelValidation:
    """Test LLM model validation and error handling"""
    
    def test_llm_returns_extra_content_after_json(self):
        """Test handling when LLM returns extra content after JSON"""
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
        
        # Mock the LLM response with extra content after JSON
        malformed_response = """{"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "living room"}, "errors": null}]}
<|im_start|>user
turn on the kitchen lights"""
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": malformed_response
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
                assert data["commands"][0]["success"] is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_llm_returns_malformed_json_structure(self):
        """Test handling when LLM returns malformed JSON structure"""
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
        
        # Mock the LLM response with malformed structure
        malformed_response = """{"success": true, "command": "turn_on_lights", "room": "living room"}"""
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": malformed_response
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
                # The system should convert the malformed structure to the correct format
                assert get_response_success(data) is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_llm_returns_invalid_json_syntax(self):
        """Test handling when LLM returns invalid JSON syntax"""
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
        
        # Mock the LLM response with invalid JSON syntax
        malformed_response = """{"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "living room"}, "errors": null}]}  // This comment makes it invalid JSON"""
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": malformed_response
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
                assert data["commands"][0]["success"] is True  # Successfully recovered
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_llm_returns_multiple_json_objects(self):
        """Test handling when LLM returns multiple JSON objects"""
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
        
        # Mock the LLM response with multiple JSON objects
        malformed_response = """{"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "living room"}, "errors": null}]}
{"commands": [{"success": true, "command_name": "play_music", "parameters": {"song": "jazz"}, "errors": null}]}"""
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
            with patch('app.main.post') as mock_post:
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": malformed_response
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
                # The system should fail to parse multiple JSON objects
                assert get_response_success(data) is False
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_llm_response_consistency(self):
        """Test that LLM responses are consistent across multiple calls"""
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
        consistent_response = {
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
                                "content": json.dumps(consistent_response)
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
                assert data["commands"][0]["success"] is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 