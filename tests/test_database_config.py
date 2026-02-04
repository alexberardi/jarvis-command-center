"""
Tests for PostgreSQL database configuration.

This test file covers database configuration and connection functionality
for PostgreSQL. SQLite is not supported.
"""
import pytest
import os
from unittest.mock import patch
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from app.db import get_database_url, create_database_engine, get_session_local
from app.models import Base, Node
from app.main import app


class TestDatabaseConfiguration:
    """Test PostgreSQL database configuration."""

    def test_get_database_url_from_database_url_env(self):
        """Test that DATABASE_URL env var is used."""
        test_url = "postgresql://user:pass@localhost:5432/testdb"
        with patch.dict(os.environ, {"DATABASE_URL": test_url}, clear=False):
            # Clear DB_URL if set to ensure DATABASE_URL is used
            env = dict(os.environ)
            env.pop("DB_URL", None)
            with patch.dict(os.environ, env, clear=True):
                with patch.dict(os.environ, {"DATABASE_URL": test_url}):
                    url = get_database_url()
                    assert url == test_url

    def test_get_database_url_from_db_url_env(self):
        """Test that DB_URL env var is used as fallback."""
        test_url = "postgresql://user:pass@localhost:5432/testdb"
        with patch.dict(os.environ, {"DB_URL": test_url}, clear=True):
            url = get_database_url()
            assert url == test_url

    def test_get_database_url_normalizes_postgres_to_postgresql(self):
        """Test that postgres:// URLs are normalized to postgresql://."""
        with patch.dict(os.environ, {"DATABASE_URL": "postgres://user:pass@localhost:5432/db"}, clear=True):
            url = get_database_url()
            assert url.startswith("postgresql://")
            assert url == "postgresql://user:pass@localhost:5432/db"

    def test_get_database_url_raises_when_not_set(self):
        """Test that missing DATABASE_URL raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="DATABASE_URL environment variable is required"):
                get_database_url()

    def test_get_database_url_raises_for_non_postgresql(self):
        """Test that non-PostgreSQL URLs raise ValueError."""
        with patch.dict(os.environ, {"DATABASE_URL": "mysql://user:pass@localhost/db"}, clear=True):
            with pytest.raises(ValueError, match="Invalid database URL"):
                get_database_url()

    def test_create_database_engine(self):
        """Test that create_database_engine returns a valid engine."""
        # Use the test database URL from environment
        engine = create_database_engine()
        assert engine is not None
        assert engine.url.drivername == "postgresql"

    def test_get_session_local_returns_sessionmaker(self):
        """Test that get_session_local returns a sessionmaker."""
        session_factory = get_session_local()
        assert session_factory is not None


class TestDatabaseIntegration:
    """Test database integration with PostgreSQL."""

    def setup_method(self):
        """Setup test database connection."""
        self.engine = create_database_engine()
        self.TestSessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)

    def test_database_connection(self):
        """Test that we can connect to the database."""
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            assert result.scalar() == 1

    def test_node_crud_operations(self):
        """Test complete CRUD operations on Node model."""
        db = self.TestSessionLocal()
        unique_id = f"crud-test-{os.getpid()}-{id(self)}"

        try:
            # Create
            node = Node(
                node_id=unique_id,
                api_key=f"key-{unique_id}",
                room="living-room",
                user="test-user"
            )
            db.add(node)
            db.commit()
            db.refresh(node)

            # Read
            retrieved_node = db.query(Node).filter(Node.node_id == unique_id).first()
            assert retrieved_node is not None
            assert retrieved_node.api_key == f"key-{unique_id}"

            # Update
            retrieved_node.room = "bedroom"
            db.commit()
            db.refresh(retrieved_node)

            updated_node = db.query(Node).filter(Node.node_id == unique_id).first()
            assert updated_node.room == "bedroom"

            # Delete
            db.delete(updated_node)
            db.commit()

            deleted_node = db.query(Node).filter(Node.node_id == unique_id).first()
            assert deleted_node is None

        finally:
            # Cleanup in case test fails partway through
            leftover = db.query(Node).filter(Node.node_id == unique_id).first()
            if leftover:
                db.delete(leftover)
                db.commit()
            db.close()

    def test_multiple_nodes(self):
        """Test handling multiple nodes."""
        db = self.TestSessionLocal()
        base_id = f"multi-{os.getpid()}-{id(self)}"
        node_ids = [f"{base_id}-1", f"{base_id}-2", f"{base_id}-3"]

        try:
            # Create multiple nodes
            nodes = [
                Node(node_id=node_ids[0], api_key=f"key-{node_ids[0]}", room="room1"),
                Node(node_id=node_ids[1], api_key=f"key-{node_ids[1]}", room="room2"),
                Node(node_id=node_ids[2], api_key=f"key-{node_ids[2]}", room="room3"),
            ]

            for node in nodes:
                db.add(node)
            db.commit()

            # Query all created nodes
            created_nodes = db.query(Node).filter(Node.node_id.in_(node_ids)).all()
            assert len(created_nodes) == 3

            # Query by API key
            node_by_key = db.query(Node).filter(Node.api_key == f"key-{node_ids[1]}").first()
            assert node_by_key is not None
            assert node_by_key.node_id == node_ids[1]

        finally:
            # Cleanup
            for nid in node_ids:
                node = db.query(Node).filter(Node.node_id == nid).first()
                if node:
                    db.delete(node)
            db.commit()
            db.close()


class TestFastAPIDatabaseIntegration:
    """Test FastAPI application with PostgreSQL database."""

    def test_fastapi_ping_endpoint(self):
        """Test FastAPI app ping endpoint works."""
        client = TestClient(app)
        response = client.get("/api/v0/ping")
        assert response.status_code == 200
        assert response.json() == {"message": "pong"}


class TestDatabaseMigration:
    """Test database migration functionality."""

    def test_tables_exist(self):
        """Test that expected tables exist in database."""
        engine = create_database_engine()

        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            ))
            tables = [row[0] for row in result.fetchall()]
            assert "nodes" in tables


class TestDatabaseErrorHandling:
    """Test database error handling."""

    def test_invalid_postgres_url_format(self):
        """Test handling of malformed PostgreSQL URL."""
        with patch.dict(os.environ, {"DATABASE_URL": "not-a-valid-url"}, clear=True):
            with pytest.raises(ValueError, match="Invalid database URL"):
                get_database_url()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
