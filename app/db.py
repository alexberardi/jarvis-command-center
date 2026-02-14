import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_database_url() -> str:
    """Get the PostgreSQL database URL from environment.

    Raises:
        ValueError: If DATABASE_URL is not set.
    """
    database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL environment variable is required. "
            "Set it to a PostgreSQL connection string, e.g.: "
            "postgresql://user:password@localhost:5432/dbname"
        )

    # Normalize postgres:// to postgresql:// (for compatibility)
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)

    if not database_url.startswith(("postgresql://", "postgresql+")):
        raise ValueError(
            f"Invalid database URL. Expected PostgreSQL URL starting with 'postgresql://', "
            f"got: {database_url[:20]}..."
        )

    return database_url


def create_database_engine():
    """Create the PostgreSQL database engine."""
    database_url = get_database_url()
    return create_engine(database_url)


def get_session_local():
    """Get a session factory based on current configuration."""
    engine = create_database_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Create default engine and session
# Note: This will fail if DATABASE_URL is not set
try:
    default_engine = create_database_engine()
    SessionLocal = sessionmaker(bind=default_engine, autocommit=False, autoflush=False)
except ValueError:
    # Allow import even if DATABASE_URL is not set (for testing setup)
    default_engine = None
    SessionLocal = None
