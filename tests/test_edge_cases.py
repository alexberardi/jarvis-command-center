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


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_voice_command(self):
        """Test handling of empty voice command"""
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
        
        # Mock the LLM response for empty command
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "No command found for empty input",
                        "clarification_question": "Please provide a command to execute."
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
                    "voice_command": "",
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
                assert errors["type"] == "no_command_match"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_extremely_long_voice_command(self):
        """Test handling of extremely long voice command"""
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
        
        # Mock the LLM response for long command
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
        
        # Create an extremely long command
        long_command = "turn on the lights " * 1000
        
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
                    "voice_command": long_command,
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
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_no_available_commands(self):
        """Test handling when no commands are available"""
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
        
        # Mock the LLM response for no available commands
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "No commands available",
                        "clarification_question": "No commands are currently available."
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
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": []
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is False
                
                errors = get_response_errors(data)
                assert errors["type"] == "no_command_match"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_command_with_null_parameters(self):
        """Test handling of commands with null parameters"""
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
        
        # Mock the LLM response with null parameters
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": None,
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
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_command_with_extra_parameters(self):
        """Test handling of commands with extra parameters"""
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
        
        # Mock the LLM response with extra parameters
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {
                        "room": "living room",
                        "extra_param": "extra_value",
                        "another_extra": 123
                    },
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
                assert first_command["parameters"]["room"] == "living room"
                assert first_command["parameters"]["extra_param"] == "extra_value"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_malformed_json_response(self):
        """Test handling of malformed JSON response from LLM"""
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
                                "content": "This is not valid JSON {"
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
    
    def test_empty_json_response(self):
        """Test handling of empty JSON response from LLM"""
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
                # Return empty response
                mock_post.return_value = {
                    "choices": [
                        {
                            "message": {
                                "content": ""
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
    
    def test_llm_timeout(self):
        """Test handling of LLM timeout"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        from unittest.mock import patch
        import httpx
        
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
                # Simulate timeout
                mock_post.side_effect = httpx.TimeoutException("Request timed out")
                
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
                assert errors["type"] == "system_error"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear() 