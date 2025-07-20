import pytest
from unittest.mock import patch
from app.request_models.voice_command_request import CommandDefinition, CommandParameter
import json


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


class TestVoiceCommandDatabase:
    """Test voice command processing with database operations"""
    
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
    
    def mock_llm_response(self, command_data):
        """Helper to create mock LLM responses in the new format"""
        if isinstance(command_data, dict) and "success" in command_data:
            # Convert old format to new format
            return {"response": json.dumps({
                "commands": [command_data]
            })}
        return {"response": json.dumps(command_data)}
    
    @patch('app.main.post')
    def test_real_database_authentication(self, mock_post, client_with_test_db, test_node, sample_commands):
        """Test that authentication works with real database"""
        mock_post.return_value = self.mock_llm_response({
            "success": True,
            "command_name": "turn_on_lights",
            "parameters": {"room": test_node.room},
            "errors": None
        })
        
        response = client_with_test_db.post("/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": test_node.room, "node_id": test_node.node_id},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": test_node.api_key})
        
        assert response.status_code == 200
        data = response.json()
        assert get_response_success(data) is True
        
        first_command = extract_first_command(data)
        assert first_command["command_name"] == "turn_on_lights"
        assert first_command["parameters"]["room"] == test_node.room
    
    @patch('app.main.post')
    def test_database_node_with_empty_room(self, mock_post, client_with_test_db, test_node_empty_room, sample_commands):
        """Test database node with empty room context"""
        mock_post.return_value = self.mock_llm_response({
            "success": False,
            "command_name": None,
            "parameters": None,
            "errors": {
                "type": "missing_parameters",
                "message": "Room parameter is required",
                "missing_parameters": ["room"],
                "clarification_question": "Which room would you like to control?"
            }
        })
        
        response = client_with_test_db.post("/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": test_node_empty_room.room, "node_id": test_node_empty_room.node_id},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": test_node_empty_room.api_key})
        
        assert response.status_code == 200
        data = response.json()
        assert get_response_success(data) is False
        
        errors = get_response_errors(data)
        assert errors is not None
        assert errors["type"] == "missing_parameters"
        assert "room" in errors["missing_parameters"]
    
    @patch('app.main.post')
    def test_database_command_with_parameters(self, mock_post, client_with_test_db, test_node, sample_commands):
        """Test database command with multiple parameters"""
        mock_post.return_value = self.mock_llm_response({
            "success": True,
            "command_name": "set_temperature",
            "parameters": {"room": test_node.room, "degrees": 72},
            "errors": None
        })
        
        response = client_with_test_db.post("/voice/command", json={
            "voice_command": "set temperature to 72 degrees",
            "node_context": {"room": test_node.room, "node_id": test_node.node_id},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": test_node.api_key})
        
        assert response.status_code == 200
        data = response.json()
        assert get_response_success(data) is True
        
        first_command = extract_first_command(data)
        assert first_command["command_name"] == "set_temperature"
        assert first_command["parameters"]["room"] == test_node.room
        assert first_command["parameters"]["degrees"] == 72
    
    @patch('app.main.post')
    def test_database_multiple_nodes(self, mock_post, client_with_test_db, multiple_test_nodes, sample_commands):
        """Test multiple database nodes"""
        for node in multiple_test_nodes:
            if node.room == "":
                # Mock response for empty room
                mock_post.return_value = self.mock_llm_response({
                    "success": False,
                    "command_name": None,
                    "parameters": None,
                    "errors": {
                        "type": "missing_parameters",
                        "message": "Room parameter is required",
                        "missing_parameters": ["room"]
                    }
                })
                expected_success = False
            else:
                # Mock response for valid room
                mock_post.return_value = self.mock_llm_response({
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": node.room},
                    "errors": None
                })
                expected_success = True
            
            response = client_with_test_db.post("/voice/command", json={
                "voice_command": "turn on the lights",
                "node_context": {"room": node.room, "node_id": node.node_id},
                "available_commands": [cmd.model_dump() for cmd in sample_commands]
            }, headers={"x-api-key": node.api_key})
            
            assert response.status_code == 200
            data = response.json()
            assert get_response_success(data) is expected_success
            
            if expected_success:
                first_command = extract_first_command(data)
                assert first_command["parameters"]["room"] == node.room
    
    @patch('app.main.post')
    def test_database_command_history(self, mock_post, client_with_test_db, test_node, sample_commands):
        """Test that command history is properly stored"""
        # First command
        mock_post.return_value = self.mock_llm_response({
            "success": True,
            "command_name": "turn_on_lights",
            "parameters": {"room": test_node.room},
            "errors": None
        })
        
        response1 = client_with_test_db.post("/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": test_node.room, "node_id": test_node.node_id},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": test_node.api_key})
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert get_response_success(data1) is True
        
        # Second command
        mock_post.return_value = self.mock_llm_response({
            "success": True,
            "command_name": "set_temperature",
            "parameters": {"room": test_node.room, "degrees": 68},
            "errors": None
        })
        
        response2 = client_with_test_db.post("/voice/command", json={
            "voice_command": "set temperature to 68",
            "node_context": {"room": test_node.room, "node_id": test_node.node_id},
            "available_commands": [cmd.model_dump() for cmd in sample_commands]
        }, headers={"x-api-key": test_node.api_key})
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert get_response_success(data2) is True
        
        # Verify both commands were processed correctly
        first_command1 = extract_first_command(data1)
        assert first_command1["command_name"] == "turn_on_lights"
        
        first_command2 = extract_first_command(data2)
        assert first_command2["command_name"] == "set_temperature"
        assert first_command2["parameters"]["degrees"] == 68


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 