import json
import time
import os
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.request_models.tool_result_request import ToolResultRequest
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

# Return clearer validation errors than the default 422
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Flatten error messages for readability
    errors = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "Invalid value")
        errors.append(f"{loc}: {msg}" if loc else msg)

    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": "Request validation failed. Please correct the highlighted fields.",
            "details": errors,
        },
    )

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
        
        # Use tool-based warmup if client provides tools, otherwise use legacy approach
        if request.client_tools is not None:
            logger.info(f"🔧 Starting tool-based conversation with {len(request.client_tools)} client tools")
            
            # Debug: Print structure of first client tool to see what we're receiving
            if request.client_tools:
                import json
                first_tool = request.client_tools[0]
                logger.info(f"🔍 DEBUG: First client tool structure:")
                logger.info(f"   Type: {type(first_tool)}")
                logger.info(f"   Keys: {list(first_tool.keys()) if isinstance(first_tool, dict) else 'N/A'}")
                
                # Check for example properties
            await model_service.warmup_conversation_with_tools(
                node_context=node_context,
                conversation_id=request.conversation_id,
                timezone=client_timezone,
                client_tools=request.client_tools,
                available_commands=request.available_commands
            )
        else:
            # Legacy warmup (for backward compatibility during transition)
            logger.info(f"🔄 Starting legacy conversation")
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
        # Check if this is a tool-based conversation
        from app.core.conversation_cache import conversation_cache
        tools = conversation_cache.get_tools(request.conversation_id)
        
        if tools is not None:
            # Tool-based conversation
            logger.info(f"🔧 Processing as tool-based conversation")
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id
            )
            
            # Build response based on stop_reason
            from app.response_models.voice_command_response import StopReason, ToolCall, ValidationRequest
            
            response = VoiceCommandResponse(
                commands=[],  # Empty for tool-based responses
                request_information=RequestInformation(
                    voice_command=request.voice_command,
                    conversation_id=request.conversation_id
                ),
                stop_reason=StopReason(result.get("stop_reason", "complete")),
                assistant_message=result.get("assistant_message"),
                tool_calls=[ToolCall(**tc) for tc in result.get("tool_calls", [])],
                validation_request=(
                    ValidationRequest(**result["validation_request"])
                    if result.get("validation_request") else None
                )
            )
            
            duration = time.time() - start_time
            logger.info(f"✅ Tool-based command processed in {duration:.2f}s, stop_reason={response.stop_reason}")
            return response
        
        else:
            # Legacy command inference
            logger.info(f"🔄 Processing as legacy conversation")
            node_context = {
                "room": node_context_provider.node.room,
                "node_id": node_context_provider.node.node_id,
                "user": node_context_provider.node.user,
                "voice_mode": node_context_provider.node.voice_mode
            }
            
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
                stop_reason = StopReason.COMPLETE
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
                stop_reason = StopReason.COMPLETE
            
            response = VoiceCommandResponse(
                commands=[single_command],
                request_information=RequestInformation(
                    voice_command=request.voice_command,
                    conversation_id=request.conversation_id
                ),
                stop_reason=stop_reason,
                tool_calls=[],
                validation_request=None,
                assistant_message=None
            )
            
            duration = time.time() - start_time
            logger.info(f"✅ Legacy command processed in {duration:.2f}s using {model_service.model.name}")
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
            ),
            stop_reason=StopReason.COMPLETE,
            tool_calls=[],
            validation_request=None,
            assistant_message=None
        )


@v0_router.post("/voice/command/continue", response_model=VoiceCommandResponse)
async def continue_voice_command(
    request: ToolResultRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    """Continue a conversation by providing tool execution results."""
    from app.response_models.voice_command_response import StopReason, ToolCall, ValidationRequest
    
    start_time = time.time()
    
    logger.info(f"Continue conversation from node: {node_context_provider.node.node_id}")
    logger.info(f"Conversation ID: {request.conversation_id}, tool results: {len(request.tool_results)}")
    
    try:
        # Convert tool results to the format expected by model service
        tool_results = [
            {
                "tool_call_id": tr.tool_call_id,
                "output": tr.output
            }
            for tr in request.tool_results
        ]
        
        # Continue conversation with tool results
        result = await model_service.continue_conversation_with_tool_results(
            conversation_id=request.conversation_id,
            tool_results=tool_results
        )
        
        # Build response
        response = VoiceCommandResponse(
            commands=[],  # Empty for tool-based responses
            request_information=RequestInformation(
                voice_command="[continuation with tool results]",
                conversation_id=request.conversation_id
            ),
            stop_reason=StopReason(result.get("stop_reason", "complete")),
            assistant_message=result.get("assistant_message"),
            tool_calls=[ToolCall(**tc) for tc in result.get("tool_calls", [])],
            validation_request=(
                ValidationRequest(**result["validation_request"])
                if result.get("validation_request") else None
            )
        )
        
        duration = time.time() - start_time
        logger.info(f"✅ Continuation processed in {duration:.2f}s, stop_reason={response.stop_reason}")
        return response
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Continuation failed after {duration:.2f}s: {e}")
        
        error_command = SingleCommandResponse(
            success=False,
            command_name=None,
            parameters=None,
            errors=VoiceCommandError(
                type="processing_error",
                message=f"Failed to continue conversation: {str(e)}"
            )
        )
        
        return VoiceCommandResponse(
            commands=[error_command],
            request_information=RequestInformation(
                voice_command="[continuation with tool results]",
                conversation_id=request.conversation_id
            )
        )


# Include versioned routes at the end after all routes are defined
app.include_router(v0_router, prefix="/api/v0", tags=["v0"])
