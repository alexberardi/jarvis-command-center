import os
import sys

from app.context_providers.node_context_provider import NodeContextProvider
# Add the root project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect
import pkgutil
import logging
from typing import Optional

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from .db import get_session_local
from .models import Node
from app.core.interfaces.ijarvis_context_provider import ICommandInferenceSystemPromptProvider, ITranscriptionCleanupSystemPromptProvider
from app.context_providers.standard_system_prompt_provider import StandardCommandInferenceSystemPromptProvider
from app.context_providers.no_op_transcription_cleanup_provider import NoOpTranscriptionCleanupProvider


from dotenv import load_dotenv

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


def get_command_inference_system_prompt_provider() -> ICommandInferenceSystemPromptProvider:
    import os
    import importlib
    provider_name = os.getenv("JARVIS_SYSTEM_PROMPT_PROVIDER", "STANDARD")

    if provider_name == "STANDARD":
        return StandardCommandInferenceSystemPromptProvider()
    
    base_path = [
        os.path.join(os.path.dirname(__file__), "context_providers"),
        os.path.join(os.path.dirname(__file__), "context_providers", "custom")
    ]
    prefix = "app.context_providers."
    for path in base_path:
        if not os.path.exists(path):
            continue
        for finder, module_name, ispkg in pkgutil.walk_packages(path=[path], prefix=prefix):
            try:
                imported_module = importlib.import_module(module_name)
            except Exception as e:
                print(e)
                continue
            for _, cls in inspect.getmembers(imported_module, inspect.isclass):
                if issubclass(cls, ICommandInferenceSystemPromptProvider) and cls is not ICommandInferenceSystemPromptProvider:
                    instance = cls()
                    if instance.name.upper() == provider_name.upper():
                        return instance
    return StandardCommandInferenceSystemPromptProvider()


def get_transcription_cleanup_system_prompt_provider() -> ITranscriptionCleanupSystemPromptProvider:
    import os
    import importlib
    
    # Check if transcription cleanup is enabled
    if os.getenv("JARVIS_TRANSCRIPTION_CLEANUP_ENABLED", "false").lower() != "true":
        return NoOpTranscriptionCleanupProvider()
    
    provider_name = os.getenv("JARVIS_TRANSCRIPTION_CLEANUP_PROMPT_PROVIDER", "STANDARD")
    
    base_path = [
        os.path.join(os.path.dirname(__file__), "context_providers"),
        os.path.join(os.path.dirname(__file__), "context_providers", "custom")
    ]
    prefix = "app.context_providers."
    for path in base_path:
        if not os.path.exists(path):
            continue
        for finder, module_name, ispkg in pkgutil.walk_packages(path=[path], prefix=prefix):
            try:
                imported_module = importlib.import_module(module_name)
            except Exception as e:
                print(e)
                continue
            for _, cls in inspect.getmembers(imported_module, inspect.isclass):
                if issubclass(cls, ITranscriptionCleanupSystemPromptProvider) and cls is not ITranscriptionCleanupSystemPromptProvider:
                    instance = cls()
                    if instance.name.upper() == provider_name.upper():
                        return instance
    return NoOpTranscriptionCleanupProvider()


