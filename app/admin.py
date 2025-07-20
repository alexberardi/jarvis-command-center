from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict
from .models import Node
from .deps import get_db, verify_admin_key
from typing import List
import logging


router = APIRouter()
logger = logging.getLogger("uvicorn")


class NodeResponse(BaseModel):
    node_id: str
    room: str
    user: str
    voice_mode: str

    model_config = ConfigDict(from_attributes=True)


class NodeCreate(BaseModel):
    node_id: str
    api_key: str
    room: str
    user: str = "default"
    voice_mode: str = "brief"

class NodeUpdate(BaseModel):
    room: str | None = None
    user: str | None = None
    voice_mode: str | None = None



@router.get("/nodes", response_model=List[NodeResponse])
def list_nodes(db: Session = Depends(get_db)):
    return db.query(Node).all()


@router.post("/nodes", response_model=NodeResponse, dependencies=[Depends(verify_admin_key)])
def create_node(node: NodeCreate, db: Session = Depends(get_db)):
    existing = db.query(Node).filter(Node.node_id == node.node_id).first()

    if existing:
        raise HTTPException(status_code=400, detail="Node already exists")
    db_node = Node(**node.model_dump())
    db.add(db_node)
    db.commit()
    db.refresh(db_node)
    logger.info(f"Node created: {db_node.node_id}")
    return db_node


@router.delete("/nodes/{node_id}", dependencies=[Depends(verify_admin_key)])
def delete_node(node_id: str, db: Session = Depends(get_db)):
    node = db.query(Node).filter(Node.node_id == node_id).first()

    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    db.delete(node)
    db.commit()
    logger.info(f"Node deleted: {node.node_id}")
    return {"message": "Deleted"}

@router.patch("/nodes/{node_id}", response_model=NodeResponse, dependencies=[Depends(verify_admin_key)])
def update_node(node_id: str, payload: NodeUpdate, db: Session = Depends(get_db)):
    node = db.query(Node).filter(Node.node_id == node_id).first()
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    data = payload.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(node, key, value)

    db.commit()
    db.refresh(node)
    logger.info(f"Node updated: {node.node_id}")
    return node
