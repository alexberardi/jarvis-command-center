import time
import logging
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.request_models.tool_result_request import ToolResultRequest
from app.request_models.tool_router_training_request import ToolRouterTrainingRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse, RequestInformation
from app.debug_setup import setup_debugger
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from app.core.conversation_cache import conversation_cache
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
        
        client_tools = request.client_tools or []
        logger.info(f"🔧 Starting tool-based conversation with {len(client_tools)} client tools")
        await model_service.warmup_conversation_with_tools(
            node_context=node_context,
            conversation_id=request.conversation_id,
            timezone=client_timezone,
            client_tools=client_tools,
            available_commands=request.available_commands
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
        tools = conversation_cache.get_tools(request.conversation_id)
        if tools is None:
            raise HTTPException(status_code=400, detail="Conversation not initialized for tool-based flow")
        
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


@v0_router.post("/tool-router/train")
async def train_tool_router(
    request: ToolRouterTrainingRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key)
):
    """Train the tool router classifier using provided commands/examples."""
    from pathlib import Path
    from app.core.tool_router import training

    repo_root = Path(__file__).resolve().parents[1]
    output_path = Path(request.output_model_path) if request.output_model_path else repo_root / "temp" / "tool_classifier.bin"
    training_jsonl_path = repo_root / "temp" / "tool_router_training.jsonl"

    command_payloads = [cmd.model_dump() for cmd in request.available_commands]
    extra_examples = []
    if request.extra_training:
        extra_examples = [
            training.TrainingExample(utterance=ex.utterance, tool_name=ex.tool_name)
            for ex in request.extra_training
        ]

    examples = training.build_training_examples(
        repo_root=repo_root,
        extra_examples=extra_examples,
        extra_jsonl=request.extra_training_jsonl,
        command_payloads=command_payloads
    )
    if request.save_training_jsonl:
        training.write_training_jsonl(examples, training_jsonl_path)

    epoch = request.epoch if request.epoch is not None else 25
    lr = request.lr if request.lr is not None else 0.5
    word_ngrams = request.word_ngrams if request.word_ngrams is not None else 2

    training.train_fasttext_classifier(
        examples=examples,
        output_path=output_path,
        epoch=epoch,
        lr=lr,
        word_ngrams=word_ngrams
    )

    return {
        "status": "success",
        "examples": len(examples),
        "model_path": str(output_path),
        "training_jsonl_path": str(training_jsonl_path) if request.save_training_jsonl else None
    }


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
