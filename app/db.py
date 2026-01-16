import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base


def _infer_db_type(database_url: str) -> str:
    if database_url.startswith("postgresql://") or database_url.startswith("postgres://"):
        return "postgres"
    if database_url.startswith("sqlite://"):
        return "sqlite"
    return "sqlite"


def _resolve_database_config() -> tuple[str, str]:
    database_url = os.getenv("DATABASE_URL") or os.getenv("DB_URL", "sqlite:///./data/voice_api.db")
    db_type_env = os.getenv("DB_TYPE")
    db_type = db_type_env.lower() if db_type_env else _infer_db_type(database_url)
    return db_type, database_url

def get_database_url():
    """Get the appropriate database URL based on configuration"""
    db_type, db_url = _resolve_database_config()
    
    if db_type == "postgres":
        # Use the provided DB_URL for PostgreSQL
        return db_url
    elif db_type == "sqlite":
        # For SQLite, use the DB_URL if provided, otherwise default
        if db_url.startswith("sqlite://"):
            return db_url
        else:
            # If DB_URL is a file path, convert to SQLite URL
            return f"sqlite:///{db_url}"
    else:
        raise ValueError(f"Unsupported database type: {db_type}. Use 'postgres' or 'sqlite'")

def create_database_engine():
    """Create the appropriate database engine based on configuration"""
    database_url = get_database_url()
    db_type, _ = _resolve_database_config()
    
    if db_type == "postgres":
        # PostgreSQL configuration
        return create_engine(database_url)
    elif db_type == "sqlite":
        # SQLite configuration with threading support
        return create_engine(database_url, connect_args={"check_same_thread": False})
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

def get_session_local():
    """Get a session factory based on current configuration"""
    engine = create_database_engine()
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Create default engine and session (for backward compatibility)
# These will be used if no environment variables are set
default_engine = create_engine("sqlite:///./data/voice_api.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=default_engine, autocommit=False, autoflush=False)

# Create tables on the default engine
Base.metadata.create_all(bind=default_engine)


