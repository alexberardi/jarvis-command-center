import json
import time
import os
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse, RequestInformation
from app.debug_setup import setup_debugger
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from app.core.conversation_cache import conversation_cache
from app.core.llm_manager import LLMManager
from app.core.command_validation_service import CommandValidationService
from app.core.parameter_extraction_service import ParameterExtractionService
from . import admin, chat, date_context
from app.deps import verify_api_key, get_model_service
from app.core.model_service import ModelService
from app.core.utils.rest_client import post  # For test mocking compatibility

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn")

# Set up debugger if DEBUG environment variable is set
setup_debugger()

# Initialize malformed JSON extractor service
malformed_json_extractor = MalformedJsonExtractorService()



app = FastAPI(title="Jarvis Command Center", version="1.0.0")

# Add shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on app shutdown."""
    logger.info("🛑 Shutting down Jarvis Command Center...")
    logger.info("✅ Shutdown complete")

# Create versioned routers
v0_router = APIRouter()

# Include admin router with versioning
app.include_router(admin.router, prefix="/api/v0/admin", tags=["admin"])

# Include chat router
app.include_router(chat.chat_router, prefix="/api/v0", tags=["chat"])

# Include date context router
app.include_router(date_context.date_context_router, prefix="/api/v0", tags=["date-context"])


# Basic routes
@v0_router.get("/ping")
def ping():
    return {"message": "pong"}


@v0_router.get("/health")
def health_check():
    """Health check endpoint for monitoring and load balancers"""
    from datetime import datetime

    # Check LLM API availability
    llm_proxy_available = True
    try:
        # We could add an actual health check to the LLM proxy here
        pass
    except Exception as e:
        llm_proxy_available = False

    return {
        "status": "healthy" if llm_proxy_available else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "llm_proxy": "available" if llm_proxy_available else "unavailable",
            "database": "available",  # We could add database health check here
            "conversation_cache": "available"  # Could add cache health check
        }
    }




@v0_router.post("/conversation/start")
async def start_conversation(
    request: ConversationStartRequest,
    background_tasks: BackgroundTasks,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    """Start a new conversation and warm up the model with context."""
    try:
        # Build node context from node properties (ignore client-provided context for security)
        node_context = {
            "room": node_context_provider.node.room,
            "node_id": node_context_provider.node.node_id,
            "user": node_context_provider.node.user,
            "voice_mode": node_context_provider.node.voice_mode
        }
        
        # Extract timezone from client context for date calculations
        if request.node_context:
            client_timezone = request.node_context.get("timezone")
        else:
            client_timezone = None
        
        # Use the new model service for warmup
        await model_service.warmup_conversation(
            node_context=node_context,
            available_commands=request.available_commands,
            conversation_id=request.conversation_id,
            timezone=client_timezone
        )

        
        # Return success immediately - LLM warm-up and cache population will happen in background
        return {"status": "success", "conversation_id": request.conversation_id}
            
    except Exception as e:
        logger.error(f"❌ Error starting conversation {request.conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")




@v0_router.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    start_time = time.time()

    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")
    logger.info(f"Command: '{request.voice_command}'")

    try:
        # Build node context for the model
        node_context = {
            "room": node_context_provider.node.room,
            "node_id": node_context_provider.node.node_id,
            "user": node_context_provider.node.user,
            "voice_mode": node_context_provider.node.voice_mode
        }
        
        # Use the new model service for complete inference
        result = await model_service.process_voice_command(
            voice_command=request.voice_command,
            conversation_id=request.conversation_id,
            node_context=node_context
        )
        
        # Convert model result to VoiceCommandResponse format
        if result["s"]:  # Success
            single_command = SingleCommandResponse(
                success=True,
                command_name=result["n"],
                parameters=result["p"],
                errors=None
            )
        else:  # Error
            error_info = result.get("e", {})
            single_command = SingleCommandResponse(
                success=False,
                command_name=result["n"],
                parameters=result["p"],
                errors=VoiceCommandError(
                    type=error_info.get("type", "unknown_error"),
                    message=error_info.get("message", "An unknown error occurred")
                )
            )
        
        # Create response with request information
        response = VoiceCommandResponse(
            commands=[single_command],
            request_information=RequestInformation(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id
            )
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Voice command processed in {duration:.2f}s using {model_service.model.name}")
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Voice command processing failed after {duration:.2f}s: {e}")
        
        # Return error response
        error_command = SingleCommandResponse(
            success=False,
            command_name=None,
            parameters=None,
            errors=VoiceCommandError(
                type="processing_error",
                message=f"Failed to process command: {str(e)}"
            )
        )
        
        return VoiceCommandResponse(
            commands=[error_command],
            request_information=RequestInformation(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id
            )
        )


# Include versioned routes at the end after all routes are defined
app.include_router(v0_router, prefix="/api/v0", tags=["v0"])
