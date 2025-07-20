from typing import List 
from fastapi import FastAPI, Depends, Request

from app.context_providers.node_context_provider import NodeContextProvider
from app.core.interfaces.ijarvis_context_provider import IJarvisContextProvider, ISystemPromptProvider

from .deps import verify_api_key, get_system_prompt_provider
from . import admin
import logging
import os
from app.core.utils.rest_client import post
from app.request_models.voice_command_request import VoiceCommandRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn")

app = FastAPI()

app.include_router(admin.router, prefix="/admin", tags=["admin"])

if os.getenv("DEBUG", "false").lower() == "true":
    try:
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
    except ImportError:
        print("debugpy is not installed, but DEBUG is set.")


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


@app.get("/ping")
def ping():
    return {"message": "pong"}


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers"""
    from datetime import datetime
    
    # Check LLM API availability
    llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
    llm_status = "unknown"
    
    try:
        # You could add a quick ping to the LLM API here if needed
        # For now, just check if the URL is configured
        if llm_api_url:
            llm_status = "configured"
    except Exception:
        llm_status = "error"
    
    return {
        "status": "healthy",
        "service": "jarvis-command-center",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": {
            "llm_api_url": llm_api_url,
            "llm_status": llm_status,
            "command_preprocessing_enabled": os.getenv("COMMAND_PREPROCESSING_ENABLED", "false").lower() == "true",
            "debug_mode": os.getenv("DEBUG", "false").lower() == "true"
        }
    }


@app.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    system_prompt_provider: ISystemPromptProvider = Depends(get_system_prompt_provider)
):
    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")

    # Build system prompt with optional voice command for preprocessing
    system_prompt = system_prompt_provider.build_system_prompt(
        request.node_context, 
        request.available_commands, 
        request.voice_command
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.voice_command}
    ]

    llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
    chat_url = f"{llm_api_url}/chat"

    llm_response = await post(url=chat_url, json_data={"messages": messages})

    # Parse the LLM response and cast to VoiceCommandResponse
    try:
        # Assuming the LLM response has a 'content' field with the JSON string
        response_content = llm_response.get('content', llm_response.get('response', ''))
        if isinstance(response_content, str):
            original_content = response_content
            response_content = response_content.strip()
            
            # Check if we need to extract JSON from malformed response
            if response_content.startswith('{'):
                brace_count = 0
                json_end = -1
                for i, char in enumerate(response_content):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i + 1
                            break
                
                if json_end > 0 and json_end < len(response_content):
                    # There's extra content after the JSON - this is a model issue!
                    extra_content = response_content[json_end:].strip()
                    if extra_content:
                        logger.warning(f"🚨 LLM MODEL ISSUE: Response contains extra content after JSON")
                        logger.warning(f"Extra content: {repr(extra_content[:100])}{'...' if len(extra_content) > 100 else ''}")
                        logger.warning(f"Full response length: {len(original_content)} chars")
                        logger.warning(f"JSON ends at position: {json_end}")
                    
                    # Extract only the JSON part
                    json_part = response_content[:json_end]
                    response_data = json.loads(json_part)
                    
                    # Log successful recovery
                    logger.info(f"✅ Successfully recovered JSON from malformed response")
                else:
                    # Normal case - try to parse the whole thing
                    response_data = json.loads(response_content)
            else:
                response_data = json.loads(response_content)
        else:
            response_data = response_content
        
        return VoiceCommandResponse(**response_data)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        from app.response_models.voice_command_response import SingleCommandResponse
        return VoiceCommandResponse(
            commands=[SingleCommandResponse(
                success=False,
                errors=VoiceCommandError(
                    type="parsing_error",
                    message=f"Failed to parse LLM response: {str(e)}",
                    clarification_question="Could you please rephrase your command?"
                )
            )]
        )



