import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.request_models.voice_command_request import CommandDefinition, CommandParameter
from unittest.mock import patch
import json


def extract_commands(data):
    """Helper function to extract all commands from the response"""
    if "commands" in data and data["commands"]:
        return data["commands"]
    return [data]  # Fallback for old format


def get_response_success(data):
    """Helper function to check if the response is successful"""
    if "commands" in data:
        return all(cmd.get("success", False) for cmd in data["commands"])
    return data.get("success", False)


class TestMultiCommandScenarios:
    """Test cases for multi-command functionality"""
    
    def setup_method(self):
        """Setup authentication mock for each test"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from app.models import Node
        
        # Create a mock node for testing
        mock_node = Node(
            node_id="test-node",
            room="living room",
            api_key="test-key",
            user="test-user"
        )
        
        def mock_verify_api_key():
            return NodeContextProvider(mock_node)
        
        app.dependency_overrides[verify_api_key] = mock_verify_api_key
    
    def teardown_method(self):
        """Clean up after each test"""
        app.dependency_overrides.clear()
    
    def test_sequential_commands_with_and(self):
        """Test commands connected with 'and'"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            
            commands = extract_commands(data)
            assert len(commands) == 2
            assert commands[0]["command_name"] == "turn_on_lights"
            assert commands[1]["command_name"] == "set_temperature"
    
    def test_sequential_commands_with_then(self):
        """Test commands connected with 'then'"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            
            commands = extract_commands(data)
            assert len(commands) == 2
            assert commands[0]["command_name"] == "turn_off_lights"
            assert commands[1]["command_name"] == "play_music"
    
    def test_mixed_success_failure_commands(self):
        """Test when some commands succeed and others fail"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            assert get_response_success(data) is False  # Overall should fail
            
            commands = extract_commands(data)
            assert len(commands) == 2
            assert commands[0]["success"] is True
            assert commands[1]["success"] is False
    
    def test_three_or_more_commands(self):
        """Test handling of 3+ commands in one request"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            
            commands = extract_commands(data)
            assert len(commands) == 3
            assert commands[0]["command_name"] == "turn_on_lights"
            assert commands[1]["command_name"] == "set_temperature"
            assert commands[2]["command_name"] == "play_music"
    
    def test_commands_with_different_rooms(self):
        """Test commands that affect different rooms"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            
            commands = extract_commands(data)
            assert len(commands) == 2
            assert commands[0]["parameters"]["room"] == "living room"
            assert commands[1]["parameters"]["room"] == "bedroom"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 