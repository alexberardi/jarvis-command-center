import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db import Base
from app.deps import get_db
from app.models import Node
from app.request_models.voice_command_request import CommandDefinition, CommandParameter
from unittest.mock import patch, AsyncMock
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


class TestVoiceCommandWithRealDB:
    """Integration tests using real database with mocked LLM responses"""
    
    @pytest.fixture
    def client_with_test_db(self):
        """Create a test client with a real test database"""
        # Create test database
        SQLALCHEMY_DATABASE_URL = "sqlite:///./test_voice_command.db"
        engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        
        def override_get_db():
            try:
                db = TestingSessionLocal()
                yield db
            finally:
                db.close()
        
        app.dependency_overrides[get_db] = override_get_db
        
        # Add test node to database
        db = TestingSessionLocal()
        test_node = Node(
            node_id="test-node",
            api_key="test-key",
            room="living room",
            user="test-user"
        )
        db.add(test_node)
        db.commit()
        db.close()
        
        client = TestClient(app)
        yield client
        
        # Clean up
        app.dependency_overrides.clear()
        Base.metadata.drop_all(bind=engine)
    
    @patch('app.core.utils.rest_client.post', new_callable=AsyncMock)
    def test_with_real_database_node(self, mock_post, client_with_test_db):
        """Test with real database node authentication"""
        # Mock LLM response
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
        
        mock_post.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(mock_llm_response)
                    }
                }
            ]
        }
        
        response = client_with_test_db.post("/api/v0/voice/command", json={
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
    
    @patch('app.core.utils.rest_client.post', new_callable=AsyncMock)
    def test_with_empty_room_node(self, mock_post, client_with_test_db):
        """Test with node that has empty room context"""
        # Add another test node with empty room using the same approach as the fixture
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        SQLALCHEMY_DATABASE_URL = "sqlite:///./test_voice_command.db"
        engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestingSessionLocal()
        empty_room_node = Node(
            node_id="empty-room-node",
            api_key="empty-room-key",
            room="",
            user="test-user"
        )
        db.add(empty_room_node)
        db.commit()
        db.close()
        
        # Mock LLM response - the LLM should succeed even with empty room
        # as it can use the room from the node context
        mock_llm_response = {
            "commands": [
                {
                    "success": True,
                    "command_name": "turn_on_lights",
                    "parameters": {"room": ""},
                    "errors": None
                }
            ]
        }
        
        # Mock main LLM call
        mock_post.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(mock_llm_response)
                    }
                }
            ]
        }
        
        # Set environment variable to disable transcription cleanup
        import os
        original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
        os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
        
        try:
            response = client_with_test_db.post("/api/v0/voice/command", json={
                "voice_command": "turn on the lights",
                "node_context": {"room": "", "node_id": "empty-room-node"},
                "available_commands": [
                    {
                        "command_name": "turn_on_lights",
                        "description": "Turn on lights",
                        "parameters": [{"name": "room", "type": "str"}]
                    }
                ]
            }, headers={"x-api-key": "empty-room-key"})
        finally:
            # Restore original environment variable
            if original_value is not None:
                os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
            else:
                os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
        
        assert response.status_code == 200
        data = response.json()
        assert get_response_success(data) is True
        
        first_command = extract_first_command(data)
        assert first_command["command_name"] == "turn_on_lights"
        assert first_command["parameters"]["room"] == ""
    
    @patch('app.core.utils.rest_client.post', new_callable=AsyncMock)
    def test_with_multiple_nodes(self, mock_post, client_with_test_db):
        """Test with multiple nodes in database"""
        # Add multiple test nodes using the same approach as the fixture
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        
        SQLALCHEMY_DATABASE_URL = "sqlite:///./test_voice_command.db"
        engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
        TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        db = TestingSessionLocal()
        
        nodes = [
            Node(node_id="node-1", api_key="key-1", room="bedroom", user="user1"),
            Node(node_id="node-2", api_key="key-2", room="kitchen", user="user2"),
            Node(node_id="node-3", api_key="key-3", room="", user="user3")
        ]
        
        for node in nodes:
            db.add(node)
        db.commit()
        db.close()
        
        test_cases = [
            ("key-1", "bedroom", True),
            ("key-2", "kitchen", True),
            ("key-3", "", True)  # Empty room should succeed
        ]
        
        for api_key, room, expected_success in test_cases:
            # Mock appropriate LLM response
            if expected_success:
                mock_llm_response = {
                    "commands": [
                        {
                            "success": True,
                            "command_name": "turn_on_lights",
                            "parameters": {"room": room},
                            "errors": None
                        }
                    ]
                }
            else:
                mock_llm_response = {
                    "commands": [
                        {
                            "success": False,
                            "command_name": None,
                            "parameters": None,
                            "errors": {
                                "type": "missing_parameters",
                                "message": "Room parameter is required",
                                "missing_parameters": ["room"]
                            }
                        }
                    ]
                }
            
            # Mock main LLM call
            mock_post.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(mock_llm_response)
                        }
                    }
                ]
            }
            
            # Set environment variable to disable transcription cleanup
            import os
            original_value = os.environ.get("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED")
            os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = "false"
            
            try:
                response = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": room, "node_id": f"node-{api_key.split('-')[1]}"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights",
                            "parameters": [{"name": "room", "type": "str"}]
                        }
                    ]
                }, headers={"x-api-key": api_key})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is expected_success
                
                if expected_success:
                    first_command = extract_first_command(data)
                    assert first_command["parameters"]["room"] == room
            finally:
                # Restore original environment variable
                if original_value is not None:
                    os.environ["JARVIS_TRANSCRIPTION_CLEANUP_ENABLED"] = original_value
                else:
                    os.environ.pop("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", None)
    
    def test_invalid_api_key(self, client_with_test_db):
        """Test with invalid API key"""
        response = client_with_test_db.post("/api/v0/voice/command", json={
            "voice_command": "turn on the lights",
            "node_context": {"room": "living room", "node_id": "test-node"},
                         "available_commands": [
                 {
                     "command_name": "turn_on_lights",
                     "description": "Turn on lights",
                     "parameters": [{"name": "room", "type": "str"}]
                 }
             ]
        }, headers={"x-api-key": "invalid-key"})
        
        assert response.status_code == 401
        assert response.json()["detail"] == "Invalid API Key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 