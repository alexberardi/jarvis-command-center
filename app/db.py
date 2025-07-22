import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

def get_database_url():
    """Get the appropriate database URL based on configuration"""
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    db_url = os.getenv("DB_URL", "sqlite:///./data/voice_api.db")
    
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
    db_type = os.getenv("DB_TYPE", "sqlite").lower()
    
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


