import os
import sys

from app.context_providers.node_context_provider import NodeContextProvider
# Add the root project path to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import inspect
import pkgutil
import logging

from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session

from .db import SessionLocal
from .models import Node
from app.core.interfaces.ijarvis_context_provider import ISystemPromptProvider
from app.context_providers.standard_system_prompt_provider import StandardSystemPromptProvider


from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("uvicorn")

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

def get_db():
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


def get_system_prompt_provider() -> ISystemPromptProvider:
    import os
    import importlib
    provider_name = os.getenv("JARVIS_SYSTEM_PROMPT_PROVIDER", "STANDARD")

    if provider_name == "STANDARD":
        return StandardSystemPromptProvider()
    
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
                if issubclass(cls, ISystemPromptProvider) and cls is not ISystemPromptProvider:
                    instance = cls()
                    if instance.name.upper() == provider_name.upper():
                        return instance
    return StandardSystemPromptProvider()


