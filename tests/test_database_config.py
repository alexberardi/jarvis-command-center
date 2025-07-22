import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db import get_database_url, create_database_engine, SessionLocal, get_session_local
from app.models import Base, Node
from app.main import app


class TestDatabaseConfiguration:
    """Test database configuration and connection"""
    
    def setup_method(self):
        """Clear environment variables before each test"""
        # Clear database environment variables
        if "DB_TYPE" in os.environ:
            del os.environ["DB_TYPE"]
        if "DB_URL" in os.environ:
            del os.environ["DB_URL"]
    
    def test_default_sqlite_configuration(self):
        """Test default SQLite configuration when no env vars are set"""
        url = get_database_url()
        assert url == "sqlite:///./data/voice_api.db"
    
    def test_sqlite_with_custom_path(self):
        """Test SQLite configuration with custom file path"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "/tmp/custom.db"
        
        url = get_database_url()
        assert url == "sqlite:////tmp/custom.db"
    
    def test_sqlite_with_sqlite_url(self):
        """Test SQLite configuration with SQLite URL"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///./custom.db"
        
        url = get_database_url()
        assert url == "sqlite:///./custom.db"
    
    def test_postgres_configuration(self):
        """Test PostgreSQL configuration"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        
        url = get_database_url()
        assert url == "postgresql://user:pass@localhost:5432/db"
    
    def test_invalid_db_type(self):
        """Test that invalid database type raises error"""
        os.environ["DB_TYPE"] = "mysql"
        os.environ["DB_URL"] = "mysql://localhost/db"
        
        with pytest.raises(ValueError, match="Unsupported database type"):
            get_database_url()
    
    def test_sqlite_engine_creation(self):
        """Test SQLite engine creation"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        engine = create_database_engine()
        assert engine.url.drivername == "sqlite"
    
    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES"),
        reason="PostgreSQL tests require TEST_POSTGRES environment variable"
    )
    def test_postgres_engine_creation(self):
        """Test PostgreSQL engine creation"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        
        engine = create_database_engine()
        assert engine.url.drivername == "postgresql"
    
    def test_sqlite_connection_and_operations(self):
        """Test SQLite connection and basic operations"""
        # Use in-memory SQLite for testing
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        # Recreate engine with new configuration
        test_engine = create_database_engine()
        TestSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
        
        # Create tables
        Base.metadata.create_all(bind=test_engine)
        
        # Test connection
        with test_engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        # Test ORM operations
        db = TestSessionLocal()
        try:
            # Create a test node
            test_node = Node(
                node_id="test-node",
                api_key="test-key",
                room="test-room",
                user="test-user"
            )
            db.add(test_node)
            db.commit()
            db.refresh(test_node)
            
            # Query the node
            retrieved_node = db.query(Node).filter(Node.node_id == "test-node").first()
            assert retrieved_node is not None
            assert retrieved_node.api_key == "test-key"
            assert retrieved_node.room == "test-room"
            
        finally:
            db.close()
    
    def test_environment_variable_override(self):
        """Test that environment variables properly override defaults"""
        # Set custom SQLite path
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            os.environ["DB_TYPE"] = "sqlite"
            os.environ["DB_URL"] = tmp_path
            
            url = get_database_url()
            expected_url = f"sqlite:///{tmp_path}"
            assert url == expected_url
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_case_insensitive_db_type(self):
        """Test that DB_TYPE is case insensitive"""
        os.environ["DB_TYPE"] = "POSTGRES"
        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        
        url = get_database_url()
        assert url == "postgresql://user:pass@localhost:5432/db"
        
        os.environ["DB_TYPE"] = "SQLITE"
        os.environ["DB_URL"] = "/tmp/test.db"
        
        url = get_database_url()
        assert url == "sqlite:////tmp/test.db"
    
    def test_get_session_local_dynamic(self):
        """Test that get_session_local returns different sessions for different configs"""
        # Test SQLite session
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        sqlite_session = get_session_local()
        
        # Test PostgreSQL session (mock)
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://user:pass@localhost:5432/db"
        
        postgres_session = get_session_local()
        
        # They should be different session factories
        assert sqlite_session != postgres_session


class TestDatabaseIntegration:
    """Test database integration with the application"""
    
    def setup_method(self):
        """Setup test database"""
        # Use in-memory SQLite for integration tests
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        # Recreate engine with test configuration
        self.test_engine = create_database_engine()
        self.TestSessionLocal = sessionmaker(bind=self.test_engine, autocommit=False, autoflush=False)
        
        # Create tables
        Base.metadata.create_all(bind=self.test_engine)
    
    def test_node_crud_operations(self):
        """Test complete CRUD operations on Node model"""
        db = self.TestSessionLocal()
        
        try:
            # Create
            node = Node(
                node_id="integration-test",
                api_key="integration-key",
                room="living-room",
                user="test-user"
            )
            db.add(node)
            db.commit()
            db.refresh(node)
            
            # Read
            retrieved_node = db.query(Node).filter(Node.node_id == "integration-test").first()
            assert retrieved_node is not None
            assert retrieved_node.api_key == "integration-key"
            
            # Update
            retrieved_node.room = "bedroom"
            db.commit()
            db.refresh(retrieved_node)
            
            updated_node = db.query(Node).filter(Node.node_id == "integration-test").first()
            assert updated_node.room == "bedroom"
            
            # Delete
            db.delete(updated_node)
            db.commit()
            
            deleted_node = db.query(Node).filter(Node.node_id == "integration-test").first()
            assert deleted_node is None
            
        finally:
            db.close()
    
    def test_multiple_nodes(self):
        """Test handling multiple nodes"""
        db = self.TestSessionLocal()
        
        try:
            # Create multiple nodes
            nodes = [
                Node(node_id="node1", api_key="key1", room="room1"),
                Node(node_id="node2", api_key="key2", room="room2"),
                Node(node_id="node3", api_key="key3", room="room3"),
            ]
            
            for node in nodes:
                db.add(node)
            db.commit()
            
            # Query all nodes
            all_nodes = db.query(Node).all()
            assert len(all_nodes) == 3
            
            # Query by API key
            node_by_key = db.query(Node).filter(Node.api_key == "key2").first()
            assert node_by_key is not None
            assert node_by_key.node_id == "node2"
            
        finally:
            db.close()


class TestFastAPIDatabaseIntegration:
    """Test FastAPI application with different database configurations"""
    
    def setup_method(self):
        """Setup test environment"""
        # Clear environment variables
        if "DB_TYPE" in os.environ:
            del os.environ["DB_TYPE"]
        if "DB_URL" in os.environ:
            del os.environ["DB_URL"]
    
    def test_fastapi_with_sqlite(self):
        """Test FastAPI app with SQLite database"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        # Create test client
        client = TestClient(app)
        
        # Test that the app starts without errors
        response = client.get("/api/v0/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}
    
    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES"),
        reason="PostgreSQL tests require TEST_POSTGRES environment variable"
    )
    def test_fastapi_with_postgres(self):
        """Test FastAPI app with PostgreSQL database"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://test:test@localhost:5432/test"
        
        # Create test client
        client = TestClient(app)
        
        # Test that the app starts without errors
        response = client.get("/api/v0/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}


class TestDatabaseMigration:
    """Test database migration functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        # Clear environment variables
        if "DB_TYPE" in os.environ:
            del os.environ["DB_TYPE"]
        if "DB_URL" in os.environ:
            del os.environ["DB_URL"]
    
    def test_migration_with_sqlite(self):
        """Test that migrations work with SQLite"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "sqlite:///:memory:"
        
        # Test that we can create tables
        engine = create_database_engine()
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result.fetchall()]
            assert "nodes" in tables
    
    @pytest.mark.skipif(
        not os.getenv("TEST_POSTGRES"),
        reason="PostgreSQL tests require TEST_POSTGRES environment variable"
    )
    def test_migration_with_postgres(self):
        """Test that migrations work with PostgreSQL"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "postgresql://test:test@localhost:5432/test"
        
        # Test that we can create tables
        engine = create_database_engine()
        Base.metadata.create_all(bind=engine)
        
        # Verify tables were created
        with engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
            tables = [row[0] for row in result.fetchall()]
            assert "nodes" in tables


class TestDatabaseErrorHandling:
    """Test database error handling"""
    
    def setup_method(self):
        """Setup test environment"""
        # Clear environment variables
        if "DB_TYPE" in os.environ:
            del os.environ["DB_TYPE"]
        if "DB_URL" in os.environ:
            del os.environ["DB_URL"]
    
    def test_invalid_postgres_url(self):
        """Test handling of invalid PostgreSQL URL"""
        os.environ["DB_TYPE"] = "postgres"
        os.environ["DB_URL"] = "invalid-url"
        
        # Should raise an error when trying to create engine
        with pytest.raises(Exception):
            create_database_engine()
    
    def test_sqlite_file_permissions(self):
        """Test handling of SQLite file permission issues"""
        os.environ["DB_TYPE"] = "sqlite"
        os.environ["DB_URL"] = "/root/no-access.db"
        
        # Should handle permission errors gracefully
        try:
            engine = create_database_engine()
            # If we get here, the error was handled gracefully
        except Exception as e:
            # Expected to fail due to permissions
            assert "permission" in str(e).lower() or "access" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 