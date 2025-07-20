import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
import json


class TestEdgeCases:
    """Test edge cases and error handling scenarios"""
    
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
    
    def test_empty_voice_command(self):
        """Test handling of empty voice command"""
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "No command provided",
                        "clarification_question": "What would you like me to do?"
                    }
                }
            ]
        }
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is False
            assert data["commands"][0]["errors"]["type"] == "no_command_match"
    
    def test_extremely_long_voice_command(self):
        """Test handling of extremely long voice commands"""
        # Create a very long command
        long_command = "turn on the lights " * 1000
        
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True
    
    def test_voice_command_with_special_characters(self):
        """Test handling of voice commands with special characters"""
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
        
        special_commands = [
            "turn on the lights! @#$%^&*()",
            "turn on the lights\n\n\ntabs and newlines",
            "turn on the lights ñáéíóú",
            "turn on the lights 你好世界",
            "turn on the lights 🏠💡✨"
        ]
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            
            for command in special_commands:
                response = client.post("/voice/command", json={
                    "voice_command": command,
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
                assert len(data["commands"]) == 1
    
    def test_no_available_commands(self):
        """Test handling when no commands are available"""
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "No commands available",
                        "clarification_question": "No commands are available for this device."
                    }
                }
            ]
        }
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
                "voice_command": "turn on the lights",
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": []
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is False
            assert data["commands"][0]["errors"]["type"] == "no_command_match"
    
    def test_command_with_null_parameters(self):
        """Test handling of commands with null parameters"""
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "stop_music",
                    "parameters": None,
                    "errors": None
                }
            ]
        }
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
                "voice_command": "stop the music",
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [
                    {
                        "command_name": "stop_music",
                        "description": "Stop music",
                        "parameters": []
                    }
                ]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True
            assert data["commands"][0]["command_name"] == "stop_music"
    
    def test_command_with_extra_parameters(self):
        """Test handling when LLM returns extra parameters"""
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {
                        "room": "living room",
                        "brightness": 100,  # Extra parameter not in schema
                        "color": "warm white"  # Another extra parameter
                    },
                    "errors": None
                }
            ]
        }
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
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
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True
            # Should still contain the extra parameters
            assert "brightness" in data["commands"][0]["parameters"]
            assert "color" in data["commands"][0]["parameters"]
    
    def test_preprocessing_with_no_matching_keywords(self):
        """Test command preprocessing when no keywords match"""
        from app.request_models.voice_command_request import CommandDefinition, CommandParameter
        
        # Create commands with specific keywords that won't match
        commands = [
            CommandDefinition(
                command_name="start_dishwasher",
                description="Start the dishwasher",
                parameters=[],
                keywords=["dishwasher", "wash", "dishes"]
            )
        ]
        
        mock_llm_response = {
            "commands": [
                {
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "no_command_match",
                        "message": "No matching command found",
                        "clarification_question": "I don't understand that command."
                    }
                }
            ]
        }
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            response = client.post("/voice/command", json={
                "voice_command": "turn on the lights",  # No keywords match
                "node_context": {"room": "living room", "node_id": "test-node"},
                "available_commands": [cmd.model_dump() for cmd in commands]
            }, headers={"x-api-key": "test-key"})
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is False
    
    def test_llm_timeout_simulation(self):
        """Test handling of LLM timeout scenarios"""
        with patch('app.main.post') as mock_post:
            # Simulate a timeout or connection error
            mock_post.side_effect = Exception("Request timeout")
            
            client = TestClient(app)
            
            # The exception should be raised and not handled gracefully
            # This test verifies that timeouts are properly propagated
            with pytest.raises(Exception) as exc_info:
                response = client.post("/voice/command", json={
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
            
            # Verify the exception message
            assert "Request timeout" in str(exc_info.value)
    
    def test_node_context_edge_cases(self):
        """Test edge cases in node context handling"""
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
        
        edge_cases = [
            {"room": None, "node_id": "test-node"},
            {"room": "   ", "node_id": "test-node"},  # Whitespace only
            {"room": "living room", "node_id": ""},
            {"room": "living room", "node_id": None},
        ]
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(mock_llm_response)}
            
            client = TestClient(app)
            
            for node_context in edge_cases:
                response = client.post("/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": node_context,
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": "test-key"})
                
                # Should handle gracefully
                assert response.status_code in [200, 422]  # Either success or validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 