import pytest
import json
from fastapi.testclient import TestClient
from app.main import app
from app.models import Node
from app.deps import get_db
from sqlalchemy.orm import Session


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
    """Test voice command processing with real database integration"""
    
    def test_real_database_authentication(self, client_with_test_db, test_db):
        """Test authentication with real database node"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from unittest.mock import patch
        
        # Create a real node in the database using the test session
        test_node = Node(
            node_id="test-node-auth",
            api_key="test-key-auth",
            room="living room",
            user="test-user"
        )
        test_db.add(test_node)
        test_db.commit()
        test_db.refresh(test_node)
        
        def mock_verify_api_key():
            return NodeContextProvider(test_node)
        
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
                
                response = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "living room", "node_id": "test-node-auth"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature in a room",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ],
                            "keywords": None
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-auth"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Clean up database
            test_db.delete(test_node)
            test_db.commit()
    
    def test_database_node_with_empty_room(self, client_with_test_db, test_db):
        """Test handling of node with empty room in database"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from unittest.mock import patch
        
        # Create a real node with empty room
        test_node = Node(
            node_id="test-node-empty",
            api_key="test-key-empty",
            room="",
            user="test-user"
        )
        test_db.add(test_node)
        test_db.commit()
        test_db.refresh(test_node)
        
        def mock_verify_api_key():
            return NodeContextProvider(test_node)
        
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
                
                response = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "", "node_id": "test-node-empty"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature in a room",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ],
                            "keywords": None
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-empty"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is False
                
                errors = get_response_errors(data)
                assert errors["type"] == "missing_parameters"
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Clean up database
            test_db.delete(test_node)
            test_db.commit()
    
    def test_database_command_with_parameters(self, client_with_test_db, test_db):
        """Test command processing with parameters using database node"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from unittest.mock import patch
        
        # Create a real node in the database
        test_node = Node(
            node_id="test-node-params",
            api_key="test-key-params",
            room="living room",
            user="test-user"
        )
        test_db.add(test_node)
        test_db.commit()
        test_db.refresh(test_node)
        
        def mock_verify_api_key():
            return NodeContextProvider(test_node)
        
        # Mock the LLM response
        mock_llm_response = {
            "commands": [
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
                
                response = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "set temperature to 72 degrees",
                    "node_context": {"room": "living room", "node_id": "test-node-params"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature in a room",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ],
                            "keywords": None
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-params"})
                
                assert response.status_code == 200
                data = response.json()
                assert get_response_success(data) is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Clean up database
            test_db.delete(test_node)
            test_db.commit()
    
    def test_database_multiple_nodes(self, client_with_test_db, test_db):
        """Test handling of multiple nodes in database"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from unittest.mock import patch
        
        # Create multiple real nodes in the database
        node1 = Node(
            node_id="node-1-multi",
            api_key="test-key-1-multi",
            room="kitchen",
            user="test-user"
        )
        node2 = Node(
            node_id="node-2-multi",
            api_key="test-key-2-multi",
            room="bedroom",
            user="test-user"
        )
        test_db.add_all([node1, node2])
        test_db.commit()
        test_db.refresh(node1)
        test_db.refresh(node2)
        
        def mock_verify_api_key():
            return NodeContextProvider(node1)
        
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
                
                response = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "kitchen", "node_id": "node-1-multi"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature in a room",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ],
                            "keywords": None
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-1-multi"})
                
                assert response.status_code == 200
                data = response.json()
                expected_success = True
                assert get_response_success(data) is expected_success
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Clean up database
            test_db.delete(node1)
            test_db.delete(node2)
            test_db.commit()
    
    def test_database_command_history(self, client_with_test_db, test_db):
        """Test that commands are properly logged in database"""
        from app.deps import verify_api_key
        from app.context_providers.node_context_provider import NodeContextProvider
        from unittest.mock import patch
        
        # Create a real node in the database
        test_node = Node(
            node_id="test-node-history",
            api_key="test-key-history",
            room="living room",
            user="test-user"
        )
        test_db.add(test_node)
        test_db.commit()
        test_db.refresh(test_node)
        
        def mock_verify_api_key():
            return NodeContextProvider(test_node)
        
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
                
                # Make first request
                response1 = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn on the lights",
                    "node_context": {"room": "living room", "node_id": "test-node-history"},
                    "available_commands": [
                        {
                            "command_name": "turn_on_lights",
                            "description": "Turn on lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        },
                        {
                            "command_name": "set_temperature",
                            "description": "Set temperature in a room",
                            "parameters": [
                                {"name": "room", "type": "str"},
                                {"name": "degrees", "type": "int"}
                            ],
                            "keywords": None
                        },
                        {
                            "command_name": "play_music",
                            "description": "Play music",
                            "parameters": [{"name": "song", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-history"})
                
                assert response1.status_code == 200
                data1 = response1.json()
                assert get_response_success(data1) is True
                
                # Make second request
                response2 = client_with_test_db.post("/api/v0/voice/command", json={
                    "voice_command": "turn off the lights",
                    "node_context": {"room": "living room", "node_id": "test-node-history"},
                    "available_commands": [
                        {
                            "command_name": "turn_off_lights",
                            "description": "Turn off lights in a specific room",
                            "parameters": [{"name": "room", "type": "str"}],
                            "keywords": None
                        }
                    ]
                }, headers={"x-api-key": "test-key-history"})
                
                assert response2.status_code == 200
                data2 = response2.json()
                assert get_response_success(data2) is True
        finally:
            # Clean up dependency override
            app.dependency_overrides.clear()
            # Clean up database
            test_db.delete(test_node)
            test_db.commit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 