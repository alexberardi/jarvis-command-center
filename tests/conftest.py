import pytest
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import Base, Node, SettingsRequest, SettingsSnapshot
from fastapi.testclient import TestClient
from app.main import app


def get_test_database_url():
    """Get the test database URL from environment or use default Docker setup."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://jarvis_user:jarvis_password@localhost:5433/jarvis_command_center"
    )


@pytest.fixture(scope="session")
def test_engine():
    """Create a database engine for the test session."""
    database_url = get_test_database_url()
    engine = create_engine(database_url)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    engine.dispose()


@pytest.fixture
def test_db(test_engine):
    """Create a fresh database session for each test with transaction rollback."""
    connection = test_engine.connect()
    transaction = connection.begin()

    # Create session bound to this connection
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = SessionLocal()

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client():
    """Create a test client without database overrides"""
    with TestClient(app) as client:
        yield client


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
