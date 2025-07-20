import pytest
from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch
import json


class TestLLMModelValidation:
    """Test cases for validating LLM model behavior and detecting issues"""
    
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
    
    def test_llm_returns_extra_content_after_json(self):
        """Test handling when LLM returns valid JSON followed by extra content"""
        # Simulate the actual problematic response we've seen
        malformed_response = """{
    "commands": [
        {
            "success": true,
            "command_name": "turn_on_lights",
            "parameters": {"room": "living room"},
            "errors": null
        }
    ]
}
<|im_start|>user
turn on the kitchen lights"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
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
            
            # Should successfully recover the JSON
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True
            assert data["commands"][0]["command_name"] == "turn_on_lights"
    
    def test_llm_returns_explanation_before_json(self):
        """Test handling when LLM returns explanation text before JSON"""
        malformed_response = """Based on your request, I'll turn on the lights. Here's the response:

{
    "commands": [
        {
            "success": true,
            "command_name": "turn_on_lights",
            "parameters": {"room": "living room"},
            "errors": null
        }
    ]
}"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
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
            
            # Should fail gracefully with parsing error
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is False
            assert data["commands"][0]["errors"]["type"] == "parsing_error"
    
    def test_llm_returns_completely_empty_response(self):
        """Test handling when LLM returns empty response"""
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": ""}
            
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
            assert data["commands"][0]["success"] is False
            assert data["commands"][0]["errors"]["type"] == "parsing_error"
    
    def test_llm_returns_malformed_json_structure(self):
        """Test handling when LLM returns JSON with wrong structure"""
        malformed_response = """{
    "success": true,
    "command": "turn_on_lights",
    "room": "living room"
}"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
            client = TestClient(app)
            
            # Should raise a validation error due to missing required fields
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
            
            # Verify it's a validation error
            assert "ValidationError" in str(exc_info.value) or "Field required" in str(exc_info.value)
    
    def test_llm_returns_invalid_json_syntax(self):
        """Test handling when LLM returns syntactically invalid JSON"""
        malformed_response = """{
    "commands": [
        {
            "success": true,
            "command_name": "turn_on_lights",
            "parameters": {"room": "living room"},
            "errors": null
        }
    ]
}  // This comment makes it invalid JSON"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
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
            
            # Our system successfully recovers the JSON part and warns about extra content
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True  # Successfully recovered
            assert data["commands"][0]["command_name"] == "turn_on_lights"
    
    def test_llm_returns_nested_json_in_text(self):
        """Test handling when LLM embeds JSON within other text"""
        malformed_response = """I understand you want to turn on the lights. 
        
The command would be: {"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "living room"}, "errors": null}]}

Let me know if you need anything else!"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
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
            
            # Our current system doesn't extract JSON from within text - it fails gracefully
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is False
            assert data["commands"][0]["errors"]["type"] == "parsing_error"
    
    def test_llm_returns_multiple_json_objects(self):
        """Test handling when LLM returns multiple JSON objects"""
        malformed_response = """{"commands": [{"success": true, "command_name": "turn_on_lights", "parameters": {"room": "living room"}, "errors": null}]}
{"commands": [{"success": true, "command_name": "play_music", "parameters": {"song": "jazz"}, "errors": null}]}"""
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": malformed_response}
            
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
            
            # Should extract only the first JSON object
            assert response.status_code == 200
            data = response.json()
            assert len(data["commands"]) == 1
            assert data["commands"][0]["success"] is True
            assert data["commands"][0]["command_name"] == "turn_on_lights"
    
    def test_llm_response_consistency(self):
        """Test that the same command produces consistent results"""
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
        
        with patch('app.main.post') as mock_post:
            mock_post.return_value = {"response": json.dumps(consistent_response)}
            
            client = TestClient(app)
            
            # Make the same request multiple times
            for i in range(3):
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
                assert data["commands"][0]["command_name"] == "turn_on_lights"
                assert data["commands"][0]["parameters"]["room"] == "living room"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 