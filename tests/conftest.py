import pytest
import os
import tempfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base
from app.models import Node
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client without database overrides"""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_db():
    """Create a fresh test database for each test"""
    # Create a temporary file for the test database
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)  # Close the file descriptor, we just need the path
    
    # Create engine for this specific test
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=False,
        connect_args={"check_same_thread": False}
    )
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
        # Cleanup: remove the temporary database file
        try:
            os.unlink(db_path)
        except OSError:
            pass


@pytest.fixture
def test_node(test_db):
    """Create a test node in the database"""
    node = Node(
        node_id="test-node",
        api_key="test-key",
        room="living room",
        user="test-user"
    )
    test_db.add(node)
    test_db.commit()
    test_db.refresh(node)
    
    yield node
    
    # Cleanup after test
    test_db.delete(node)
    test_db.commit()


@pytest.fixture
def test_node_empty_room(test_db):
    """Create a test node with empty room"""
    node = Node(
        node_id="test-node-empty",
        api_key="test-key-empty",
        room="",
        user="test-user"
    )
    test_db.add(node)
    test_db.commit()
    test_db.refresh(node)
    
    yield node
    
    # Cleanup after test
    test_db.delete(node)
    test_db.commit()


@pytest.fixture
def client_with_test_db(test_db):
    """Create a test client with dependency override for database"""
    from app.deps import get_db
    
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Clean up dependency override
        app.dependency_overrides.clear()


@pytest.fixture
def multiple_test_nodes(test_db):
    """Create multiple test nodes for testing"""
    nodes = [
        Node(node_id="node-1", api_key="key-1", room="kitchen", user="user1"),
        Node(node_id="node-2", api_key="key-2", room="bedroom", user="user2"),
        Node(node_id="node-3", api_key="key-3", room="", user="user3")  # Empty room
    ]
    
    for node in nodes:
        test_db.add(node)
    test_db.commit()
    
    for node in nodes:
        test_db.refresh(node)
    
    yield nodes
    
    # Cleanup
    for node in nodes:
        test_db.delete(node)
    test_db.commit() 