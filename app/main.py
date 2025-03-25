from fastapi import FastAPI, Depends, Request
from .deps import verify_api_key
from .models import Node
from sqlalchemy.orm import Session
from . import admin
import logging
from .mistral_client import query_mistral

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn")

app = FastAPI()

app.include_router(admin.router, prefix="/admin", tags=["admin"])

@app.middleware("http")
async def log_requests(request: Request, call_next):
    body = await request.body()
    try:
        body_str = body.decode("utf-8")
    except Exception:
        body_str = str(body)

    logger.info(f"Incoming request: {request.method} {request.url}")
    if request.method in ("POST", "PUT", "PATCH"):
        logger.info(f"Payload: {body_str}")

    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")

    return response


@app.get("/")
def hello_world():
    return {"message": "Hello World!"}


@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/voice")
async def handle_voice(node: Node = Depends(verify_api_key)):
    # Stub handler
    logger.info(f"Voice command from node: {node.node_id} in room {node.room}")
    result = await query_mistral("Tell me a joke")
    return { "response": result }
