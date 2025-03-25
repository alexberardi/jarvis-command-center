from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import Node
import os
import logging
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
    return node

def verify_admin_key(x_api_key: str = Header(...)):
    if x_api_key != ADMIN_API_KEY:
        logger.warning("Unauthorized admin access attempt with API key: %s", x_api_key)
        raise HTTPException(status_code=401, detail="Invalid Admin API Key")
