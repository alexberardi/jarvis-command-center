"""
Chat endpoints for LLM interactions.
"""

import logging
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from app.request_models.chat_completion_request import ChatCompletionRequest
from app.core.llm_proxy_client import LLMProxyClient

logger = logging.getLogger("uvicorn")

# Create router for chat endpoints
chat_router = APIRouter()

@chat_router.post("/chat")
async def passthrough_chat(request: ChatCompletionRequest):
    """Full model passthrough endpoint."""
    llm_client = LLMProxyClient()
    
    try:
        result = await llm_client.passthrough_chat(request.dict(), "/chat")
        return JSONResponse(status_code=result["status_code"], content=result["content"])
    except Exception as e:
        logger.error(f"[passthrough /chat] error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@chat_router.post("/lightweight/chat")
async def passthrough_lightweight_chat(request: ChatCompletionRequest):
    """Lightweight chat passthrough endpoint."""
    llm_client = LLMProxyClient()
    
    try:
        result = await llm_client.passthrough_chat(request.dict(), "/lightweight/chat")
        return JSONResponse(status_code=result["status_code"], content=result["content"])
    except Exception as e:
        logger.error(f"[passthrough /lightweight/chat] error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
