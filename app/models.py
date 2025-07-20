from sqlalchemy import Column, String, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Node(Base):
    __tablename__ = 'nodes'
    node_id = Column(String, primary_key=True)
    api_key = Column(String, nullable=False)
    room = Column(String, nullable=False)
    user = Column(String, default="default")
    voice_mode = Column(String, default="brief")
    last_seen = Column(DateTime, default=datetime.utcnow)
