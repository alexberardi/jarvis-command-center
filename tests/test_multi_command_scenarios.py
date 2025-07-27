import pytest
import json
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


class TestMultiCommandScenarios:
    """Test scenarios involving multiple commands"""
    
    def test_sequential_commands_with_and(self):
        """Test processing multiple commands connected with 'and'"""
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
        
        # Mock the LLM response for multiple commands
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                },
                {
                    "success": True,
                    "command_name": "set_temperature",
                    "parameters": {"room": "living room", "degrees": 72},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
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
                    "voice_command": "turn on the lights and set temperature to 72",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        },
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
                assert len(data["commands"]) == 2
                
                # Check first command
                assert data["commands"][0]["command_name"] == "turn_on_lights"
                assert data["commands"][0]["parameters"]["room"] == "living room"
                
                # Check second command
                assert data["commands"][1]["command_name"] == "set_temperature"
                assert data["commands"][1]["parameters"]["room"] == "living room"
                assert data["commands"][1]["parameters"]["degrees"] == 72
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_sequential_commands_with_then(self):
        """Test processing multiple commands connected with 'then'"""
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
        
        # Mock the LLM response for multiple commands
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_off_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                },
                {
                    "success": True,
                    "command_name": "play_music",
                    "parameters": {"song": "relaxing music"},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
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
                    "voice_command": "turn off the lights then play relaxing music",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_off_lights",
                            "description": "Turn off lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                assert len(data["commands"]) == 2
                
                # Check first command
                assert data["commands"][0]["command_name"] == "turn_off_lights"
                assert data["commands"][0]["parameters"]["room"] == "living room"
                
                # Check second command
                assert data["commands"][1]["command_name"] == "play_music"
                assert data["commands"][1]["parameters"]["song"] == "relaxing music"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_mixed_success_failure_commands(self):
        """Test handling of mixed success and failure commands"""
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
        
        # Mock the LLM response with mixed success/failure
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                },
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "Cannot find command to make coffee",
                        "clarification_question": "I don't have a coffee maker command. Would you like me to do something else?"
                    }
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
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
                    "voice_command": "turn on the lights and make coffee",
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
                assert len(data["commands"]) == 2
                
                # First command should succeed
                assert data["commands"][0]["success"] is True
                assert data["commands"][0]["command_name"] == "turn_on_lights"
                
                # Second command should fail
                assert data["commands"][1]["success"] is False
                assert data["commands"][1]["errors"]["type"] == "no_command_match"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_three_or_more_commands(self):
        """Test processing three or more commands"""
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
        
        # Mock the LLM response for three commands
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                },
                {
                    "success": True,
                    "command_name": "set_temperature",
                    "parameters": {"room": "living room", "degrees": 72},
                    "errors": None
                },
                {
                    "success": True,
                    "command_name": "play_music",
                    "parameters": {"song": "jazz"},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
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
                    "voice_command": "turn on lights, set temperature to 72, and play jazz music",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ]
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                assert len(data["commands"]) == 3
                
                # Check all commands
                assert data["commands"][0]["command_name"] == "turn_on_lights"
                assert data["commands"][1]["command_name"] == "set_temperature"
                assert data["commands"][2]["command_name"] == "play_music"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
    
    def test_commands_with_different_rooms(self):
        """Test processing commands for different rooms"""
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
        
        # Mock the LLM response for different rooms
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": "living room"},
                    "errors": None
                },
                {
                    "success": True,
                    "command_name": "turn_off_lights",
                    "parameters": {"room": "bedroom"},
                    "errors": None
                }
            ]
        }
        
        try:
            # Override the dependency
            app.dependency_overrides[verify_api_key] = mock_verify_api_key
            
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
                    "voice_command": "turn on living room lights and turn off bedroom lights",
                    "node_context": {"room": "living room", "node_id": "test-node"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        },
                        {
                            "command_name": "turn_off_lights",
                            "description": "Turn off lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
                assert len(data["commands"]) == 2
                
                # Check room assignments
                assert data["commands"][0]["parameters"]["room"] == "living room"
                assert data["commands"][1]["parameters"]["room"] == "bedroom"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 