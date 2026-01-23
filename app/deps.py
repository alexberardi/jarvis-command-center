import os
import sys
import logging

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from app.context_providers.node_context_provider import NodeContextProvider
from app.db import get_session_local
from app.models import Node
from app.core.model_service import ModelService
from dotenv import load_dotenv

# Add the root project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

logger = logging.getLogger("uvicorn")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

def get_db():
    """Get database session with dynamic configuration"""
    SessionLocal = get_session_local()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_api_key(x_api_key: str = Header(...), db: Session = Depends(get_db)):
    node = db.query(Node).filter(Node.api_key == x_api_key).first()
    if not node:
        logger.warning("Unauthorized attempt with API key: %s", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return NodeContextProvider(node)

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_API_KEY:
        logger.warning("Unauthorized admin access attempt with API key: %s", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid Admin API Key")


def get_model_service() -> ModelService:
    """
    Get the configured model service instance.
    
    Uses JARVIS_MODEL_INTERFACE environment variable to determine which model to use.
    Defaults to BASE_MODEL if not specified.
    
    Available models:
    - JarvisToolModel: Tool-based model using JSON tool calls
    - JarvisAdapterModel: Slim prompt model for adapter-tuned usage
    - Custom models can be added to app/core/models/custom/
    """
    return ModelService()

