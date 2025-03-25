from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base

SQLITE_URL = "sqlite:///./voice_api.db"
engine = create_engine(SQLITE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


Base.metadata.create_all(bind=engine)


