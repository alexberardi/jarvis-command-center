import pytest
import os
import uuid
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.models import Base, Node, Setting, SettingsRequest, SettingsSnapshot


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

    # Clean up any stale test data from previous runs
    with engine.connect() as conn:
        # Settings table may not exist yet on first run, so we use try/except
        try:
            conn.execute(text("DELETE FROM settings WHERE household_id LIKE 'h%' OR household_id IS NULL"))
        except Exception:
            pass  # Table doesn't exist yet
        conn.execute(text("DELETE FROM settings_snapshots WHERE node_id LIKE 'test-%' OR node_id LIKE 'node-%'"))
        conn.execute(text("DELETE FROM settings_requests WHERE node_id LIKE 'test-%' OR node_id LIKE 'node-%'"))
        conn.execute(text("DELETE FROM nodes WHERE node_id LIKE 'test-%' OR node_id LIKE 'node-%'"))
        conn.commit()

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
    """Create a test client without database overrides.

    This fixture uses lazy import to avoid importing app.main at conftest load time,
    which would fail in CI where jarvis_auth_client is not available.
    """
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_node(test_db):
    """Create a test node in the database with unique ID."""
    unique_id = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-node-{unique_id}",
        api_key=f"test-key-{unique_id}",
        room="living room",
        user="test-user"
    )
    test_db.add(node)
    test_db.commit()
    test_db.refresh(node)

    yield node


@pytest.fixture
def test_node_with_household(test_db):
    """Create a test node with household_id for media proxy tests."""
    unique_id = str(uuid.uuid4())[:8]
    household_id = f"household-{unique_id}"
    node = Node(
        node_id=f"test-node-{unique_id}",
        api_key=f"test-key-{unique_id}",
        room="living room",
        user="test-user"
    )
    test_db.add(node)
    test_db.commit()
    test_db.refresh(node)

    # Return both node and household_id since Node model doesn't have household_id yet
    yield node, household_id


@pytest.fixture
def test_node_empty_room(test_db):
    """Create a test node with empty room and unique ID."""
    unique_id = str(uuid.uuid4())[:8]
    node = Node(
        node_id=f"test-node-empty-{unique_id}",
        api_key=f"test-key-empty-{unique_id}",
        room="",
        user="test-user"
    )
    test_db.add(node)
    test_db.commit()
    test_db.refresh(node)

    yield node


@pytest.fixture
def client_with_test_db(test_db):
    """Create a test client with dependency override for database.

    This fixture uses lazy import to avoid importing app.main at conftest load time.
    """
    from app.main import app
    from app.deps import get_db
    from fastapi.testclient import TestClient

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
    """Create multiple test nodes for testing with unique IDs."""
    unique_id = str(uuid.uuid4())[:8]
    nodes = [
        Node(node_id=f"node-1-{unique_id}", api_key=f"key-1-{unique_id}", room="kitchen", user="user1"),
        Node(node_id=f"node-2-{unique_id}", api_key=f"key-2-{unique_id}", room="bedroom", user="user2"),
        Node(node_id=f"node-3-{unique_id}", api_key=f"key-3-{unique_id}", room="", user="user3"),
    ]

    for node in nodes:
        test_db.add(node)
    test_db.commit()

    for node in nodes:
        test_db.refresh(node)

    yield nodes
