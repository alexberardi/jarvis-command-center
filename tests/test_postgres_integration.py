import pytest
import os
import subprocess
import time
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db import get_database_url, create_database_engine, get_session_local
from app.models import Base, Node
from app.main import app


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES"),
    reason="PostgreSQL integration tests require TEST_POSTGRES environment variable"
)
class TestPostgreSQLIntegration:
    """Integration tests with real PostgreSQL database"""
    
    def setup_method(self):
        """Setup PostgreSQL test environment"""
        # Set PostgreSQL environment variables
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://test:test@localhost:5432/test"
        
        # Create test engine
        self.engine = create_database_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        
        # Create tables
        Base.metadata.create_all(bind=self.engine)
    
    def teardown_method(self):
        """Clean up after each test"""
        # Drop all tables
        Base.metadata.drop_all(bind=self.engine)
    
    def test_postgres_connection(self):
        """Test basic PostgreSQL connection"""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            assert "PostgreSQL" in version
    
    def test_postgres_node_operations(self):
        """Test Node model operations with PostgreSQL"""
        db = self.SessionLocal()
        
        try:
            # Create node
            node = Node(
                node_id="postgres-test",
                api_key="postgres-key",
                room="postgres-room",
                user="postgres-user"
            )
            db.add(node)
            db.commit()
            db.refresh(node)
            
            # Query node
            retrieved_node = db.query(Node).filter(Node.node_id == "postgres-test").first()
            assert retrieved_node is not None
            assert retrieved_node.api_key == "postgres-key"
            assert retrieved_node.room == "postgres-room"
            
            # Update node
            retrieved_node.room = "updated-room"
            db.commit()
            db.refresh(retrieved_node)
            
            updated_node = db.query(Node).filter(Node.node_id == "postgres-test").first()
            assert updated_node.room == "updated-room"
            
            # Delete node
            db.delete(updated_node)
            db.commit()
            
            deleted_node = db.query(Node).filter(Node.node_id == "postgres-test").first()
            assert deleted_node is None
            
        finally:
            db.close()
    
    def test_postgres_concurrent_operations(self):
        """Test concurrent operations with PostgreSQL"""
        db1 = self.SessionLocal()
        db2 = self.SessionLocal()
        
        try:
            # Create node in first session
            node1 = Node(
                node_id="concurrent-test-1",
                api_key="key1",
                room="room1"
            )
            db1.add(node1)
            db1.commit()
            
            # Create node in second session
            node2 = Node(
                node_id="concurrent-test-2",
                api_key="key2",
                room="room2"
            )
            db2.add(node2)
            db2.commit()
            
            # Query from both sessions
            nodes1 = db1.query(Node).all()
            nodes2 = db2.query(Node).all()
            
            assert len(nodes1) == 2  # Should see both nodes
            assert len(nodes2) == 2  # Should see both nodes
            
        finally:
            db1.close()
            db2.close()
    
    def test_postgres_transaction_rollback(self):
        """Test PostgreSQL transaction rollback"""
        db = self.SessionLocal()
        
        try:
            # Start transaction
            node = Node(
                node_id="rollback-test",
                api_key="rollback-key",
                room="rollback-room"
            )
            db.add(node)
            db.commit()
            
            # Verify node exists
            retrieved_node = db.query(Node).filter(Node.node_id == "rollback-test").first()
            assert retrieved_node is not None
            
            # Rollback transaction
            db.rollback()
            
            # Node should still exist (commit was already done)
            retrieved_node = db.query(Node).filter(Node.node_id == "rollback-test").first()
            assert retrieved_node is not None
            
        finally:
            db.close()
    
    def test_postgres_fastapi_integration(self):
        """Test FastAPI app with PostgreSQL database"""
        client = TestClient(app)
        
        # Test ping endpoint
        response = client.get("/api/v0/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}
        
        # Test health endpoint
        response = client.get("/api/v0/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "jarvis-command-center"
    
    def test_postgres_admin_api(self):
        """Test admin API with PostgreSQL database"""
        client = TestClient(app)
        
        # Set admin API key
        os.environ["ADMIN_API_KEY"] = "test-admin-key"
        
        # Create node via admin API
        node_data = {
            "node_id": "admin-test",
            "api_key": "admin-test-key",
            "room": "admin-room",
            "user": "admin-user"
        }
        
        response = client.post(
            "/api/v0/admin/nodes",
            json=node_data,
            headers={"x-api-key": "test-admin-key"}
        )
        
        assert response.status_code == 200
        created_node = response.json()
        assert created_node["node_id"] == "admin-test"
        assert created_node["room"] == "admin-room"
        
        # List nodes
        response = client.get("/api/v0/admin/nodes")
        assert response.status_code == 200
        nodes = response.json()
        assert len(nodes) >= 1
        
        # Find our created node
        admin_node = next((n for n in nodes if n["node_id"] == "admin-test"), None)
        assert admin_node is not None
    
    def test_postgres_voice_command_api(self):
        """Test voice command API with PostgreSQL database"""
        client = TestClient(app)
        
        # First create a node via admin API
        os.environ["ADMIN_API_KEY"] = "test-admin-key"
        
        node_data = {
            "node_id": "voice-test",
            "api_key": "voice-test-key",
            "room": "voice-room",
            "user": "voice-user"
        }
        
        client.post(
            "/api/v0/admin/nodes",
            json=node_data,
            headers={"x-api-key": "test-admin-key"}
        )
        
        # Test voice command with the created node
        voice_data = {
            "voice_command": "turn on the lights",
            "node_context": {"room": "voice-room", "node_id": "voice-test"},
            "available_commands": [
                {
                    "command_name": "turn_on_lights",
                    "description": "Turn on lights",
                    "parameters": [{"name": "room", "type": "str"}]
                }
            ]
        }
        
        response = client.post(
            "/api/v0/voice/command",
            json=voice_data,
            headers={"x-api-key": "voice-test-key"}
        )
        
        # Should get a response (may be success or error depending on LLM availability)
        assert response.status_code == 200
        data = response.json()
        assert "commands" in data


class TestPostgreSQLDockerIntegration:
    """Test PostgreSQL integration with Docker Compose"""
    
    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES_DOCKER"),
        reason="Docker PostgreSQL tests require TEST_POSTGRES_DOCKER environment variable"
    )
    def test_docker_postgres_setup(self):
        """Test PostgreSQL setup with Docker Compose"""
        # This test would require Docker Compose to be running
        # In a real CI environment, you'd start the services here
        
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"
        
        # Test connection
        engine = create_database_engine()
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
    
    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES_DOCKER"),
        reason="Docker PostgreSQL tests require TEST_POSTGRES_DOCKER environment variable"
    )
    def test_docker_postgres_migrations(self):
        """Test migrations with Docker PostgreSQL"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"
        
        # Create tables
        engine = create_database_engine()
        Base.metadata.create_all(bind=engine)
        
        # Verify tables exist
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            assert "nodes" in tables


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 