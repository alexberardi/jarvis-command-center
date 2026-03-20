import time
import logging
import os
import json
import urllib.parse
from pathlib import Path
import hashlib
from uuid import uuid4
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.request_models.tool_result_request import ToolResultRequest
from app.request_models.tool_router_training_request import ToolRouterTrainingRequest
from app.request_models.adapter_training_request import AdapterTrainingRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse, RequestInformation, StopReason, ToolCall, ValidationRequest
from app.debug_setup import setup_debugger
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from app.core.conversation_cache import conversation_cache
from app.core.utils.latency_logger import latency_logger
from . import admin, chat, date_context, node_settings, provisioning
from app.api import media, node_commands, test_commands
from app.deps import verify_api_key, get_model_service
from app.core.model_service import ModelService
from app.core.utils.rest_client import post  # For test mocking compatibility

# Set up logging - console at WARNING level for quieter output
console_level = os.getenv("JARVIS_LOG_CONSOLE_LEVEL", "WARNING")
logging.basicConfig(
    level=getattr(logging, console_level.upper(), logging.WARNING),
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("uvicorn")

# Remote logging handler (initialized in startup event)
_jarvis_handler = None


def _setup_service_config() -> None:
    """Set up service discovery via jarvis-config-service."""
    from app.core import service_config

    config_url = os.getenv("JARVIS_CONFIG_URL")
    if not config_url:
        logger.warning(
            "⚠️  JARVIS_CONFIG_URL not set - using legacy env vars for service URLs"
        )
        return

    try:
        # Initialize with database engine for persistent caching
        from app.db import default_engine
        success = service_config.init(db_engine=default_engine)
        if success:
            logger.info("✅ Service discovery initialized from jarvis-config-service")
        else:
            logger.warning("⚠️  Service discovery using cached/fallback data")
    except Exception as e:
        logger.error(f"❌ Failed to initialize service discovery: {e}")
        logger.warning("⚠️  Falling back to legacy env vars for service URLs")


def _setup_remote_logging() -> None:
    """Set up remote logging to jarvis-logs server. Called after uvicorn initializes."""
    global _jarvis_handler
    try:
        from jarvis_log_client import init as init_log_client, JarvisLogHandler

        app_id = os.getenv("JARVIS_APP_ID", "command-center")
        app_key = os.getenv("JARVIS_APP_KEY")
        if not app_key:
            return

        init_log_client(app_id=app_id, app_key=app_key)

        remote_level = os.getenv("JARVIS_LOG_REMOTE_LEVEL", "DEBUG")
        _jarvis_handler = JarvisLogHandler(
            service="command-center",
            level=getattr(logging, remote_level.upper(), logging.DEBUG),
        )

        # Add to uvicorn loggers (they propagate, but adding directly is more reliable)
        for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
            lg = logging.getLogger(logger_name)
            lg.addHandler(_jarvis_handler)

        logging.getLogger("uvicorn").info("📡 Remote logging enabled to jarvis-logs")
    except ImportError:
        pass  # jarvis-log-client not installed

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

# Add startup event to initialize services after uvicorn is ready
@app.on_event("startup")
async def startup_event():
    """Initialize services on app startup."""
    import asyncio
    from app.provisioning import cleanup_expired_tokens

    # Initialize service discovery first
    _setup_service_config()
    # Then set up remote logging
    _setup_remote_logging()

    # Connect MCP client (non-blocking, service works without it)
    try:
        from jarvis_mcp_client import init as init_mcp
        mcp_url = os.getenv("JARVIS_MCP_URL", "http://localhost:7709")
        await init_mcp(mcp_url=mcp_url)
        logger.info("MCP client connected to %s", mcp_url)
    except ImportError:
        logger.info("jarvis-mcp-client not installed, MCP tools unavailable")
    except Exception as e:
        logger.warning("MCP client connection failed (non-fatal): %s", e)

    # Clean up expired provisioning tokens from previous runs
    try:
        from app.db import get_session_local
        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            removed = cleanup_expired_tokens(db)
            if removed:
                logger.info("Cleaned up %d expired provisioning tokens on startup", removed)
        finally:
            db.close()
    except Exception as e:
        logger.warning("Could not clean up provisioning tokens on startup: %s", e)

    # Schedule hourly cleanup
    async def _periodic_cleanup() -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                db = SessionLocal()
                try:
                    removed = cleanup_expired_tokens(db)
                    if removed:
                        logger.info("Hourly cleanup removed %d provisioning tokens", removed)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Hourly provisioning token cleanup failed: %s", e)

    asyncio.create_task(_periodic_cleanup())

    # Include settings router (after service_config is initialized so auth URL resolves)
    from jarvis_settings_client import create_settings_router, create_combined_auth, create_superuser_auth
    from app.services.settings_service import get_settings_service
    from app.core import service_config

    auth_url = service_config.get_auth_url()
    _settings_router = create_settings_router(
        service=get_settings_service(),
        auth_dependency=create_combined_auth(auth_url),
        write_auth_dependency=create_superuser_auth(auth_url),
    )
    app.include_router(_settings_router, prefix="/settings", tags=["settings"])


# Add shutdown event for cleanup
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on app shutdown."""
    logger.info("🛑 Shutting down Jarvis Command Center...")
    # Shutdown service discovery
    try:
        from app.core import service_config
        service_config.shutdown()
    except Exception as e:
        pass
    # Disconnect MQTT client
    try:
        if node_settings.mqtt_client is not None:
            node_settings.mqtt_client.disconnect()
    except Exception as e:
        pass
    # Disconnect MCP client
    try:
        from jarvis_mcp_client import get_client
        client = get_client()
        if client and client.is_connected():
            await client.disconnect()
    except Exception as e:
        logger.debug("MCP client disconnect error (non-fatal): %s", e)
    # Flush and close remote log handler
    for handler in logger.handlers:
        if hasattr(handler, 'close'):
            handler.close()
    logger.info("✅ Shutdown complete")


# Root-level health endpoint (standardized across all services)
@app.get("/health")
def root_health_check():
    """Health check endpoint at root level for service discovery."""
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "jarvis-command-center",
        "timestamp": datetime.utcnow().isoformat(),
    }


# Create versioned routers
v0_router = APIRouter()

# Include admin router with versioning
app.include_router(admin.router, prefix="/api/v0/admin", tags=["admin"])

# Include provisioning router
app.include_router(provisioning.router, prefix="/api/v0", tags=["provisioning"])

# Include chat router
app.include_router(chat.chat_router, prefix="/api/v0", tags=["chat"])

# Include date context router
app.include_router(date_context.date_context_router, prefix="/api/v0", tags=["date-context"])

# Include node settings router
app.include_router(node_settings.router, prefix="/api/v0", tags=["node-settings"])

# Include smart home router (config push, rooms, devices)
from app.api import smart_home
app.include_router(smart_home.router, prefix="/api/v0", tags=["smart-home"])

# Include media proxy router
app.include_router(media.router, prefix="/api/v0", tags=["media"])

# Include node commands router
app.include_router(node_commands.router, prefix="/api/v0", tags=["node-commands"])

# Include test commands router (app-to-app auth)
app.include_router(test_commands.router, prefix="/api/v0", tags=["testing"])

# Include package install router (Pantry store → node install via MQTT)
from app.api import package_install
app.include_router(package_install.router, prefix="/api/v0", tags=["package-install"])

# Include memory CRUD router
from app.api import memories
app.include_router(memories.router, prefix="/api/v0", tags=["memories"])

# Include OAuth session management router
from app.api import oauth
app.include_router(oauth.router, prefix="/api/v0", tags=["oauth"])

# Include agent utility endpoints (news, calendar for node-side agents)
from app.api import agents
app.include_router(agents.router, prefix="/api/v0", tags=["agents"])

# Include mobile chat, audio, and node tools endpoints (JWT auth)
from app.api import mobile_chat, mobile_audio, node_tools
app.include_router(mobile_chat.router, prefix="/api/v0/mobile", tags=["mobile-chat"])
app.include_router(mobile_audio.router, prefix="/api/v0/mobile", tags=["mobile-audio"])
app.include_router(node_tools.router, prefix="/api/v0/mobile", tags=["node-tools"])

# Settings router is included in startup_event after service_config is initialized


# Basic routes
@v0_router.get("/ping")
def ping():
    return {"message": "pong"}





@v0_router.post("/conversation/start")
async def start_conversation(
    request: ConversationStartRequest,
    background_tasks: BackgroundTasks,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    """Start a new conversation and warm up the model with context."""
    timing = latency_logger.start_request(request.conversation_id, "warmup")
    timing.checkpoint("auth_complete")

    try:
        # Build node context from node properties (ignore most client-provided context for security)
        node_context = {
            "room": node_context_provider.node.room,
            "node_id": node_context_provider.node.node_id,
            "user": node_context_provider.node.user,
            "voice_mode": node_context_provider.node.voice_mode,
            "adapter_hash": node_context_provider.node.adapter_hash,
            "household_id": node_context_provider.household_id,
        }

        # Inject room hierarchy from CC database for LLM prompt context
        if node_context.get("household_id"):
            try:
                from app.models import Room as RoomModel
                from app.db import get_session_local
                _session = get_session_local()()
                try:
                    _hh_id = node_context["household_id"]
                    _rooms = _session.query(RoomModel).filter(RoomModel.household_id == _hh_id).all()
                    _has_hierarchy = any(r.parent_room_id for r in _rooms)
                    if _has_hierarchy:
                        node_context["room_hierarchy"] = [
                            {"id": r.id, "name": r.name, "parent_room_id": r.parent_room_id}
                            for r in _rooms
                        ]
                finally:
                    _session.close()
            except Exception as e:
                logger.warning(f"Failed to load room hierarchy: {e}")

        # Extract timezone and speaker identity from client context
        client_timezone = None
        if request.node_context:
            client_timezone = request.node_context.get("timezone")
            # Include agent context (e.g., Home Assistant devices) - this is read-only data
            if "agents" in request.node_context:
                node_context["agents"] = request.node_context["agents"]
                logger.info(f"📦 Received agent context: {list(request.node_context['agents'].keys())}")

            # Extract speaker identity from voice recognition
            speaker_user_id = request.node_context.get("speaker_user_id")
            if speaker_user_id is not None:
                node_context["speaker_user_id"] = speaker_user_id
                # Resolve speaker_user_id to display name
                try:
                    from app.core import service_config
                    from app.core.utils.speaker_resolver import resolve_speaker_name
                    auth_url = service_config.get_auth_url()
                    speaker_name = await resolve_speaker_name(auth_url, speaker_user_id)
                    if speaker_name:
                        node_context["speaker_name"] = speaker_name
                        logger.info(f"🎤 Speaker identified: {speaker_name} (user_id={speaker_user_id})")
                    else:
                        logger.info(f"🎤 Speaker user_id={speaker_user_id} (name not resolved)")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to resolve speaker name: {e}")

        client_tools = request.client_tools or []
        available_commands = request.available_commands or []
        logger.info(f"🔧 Starting tool-based conversation with {len(client_tools)} client tools (skip_warmup={request.skip_warmup_inference})")

        with timing.measure("warmup_conversation_with_tools"):
            await model_service.warmup_conversation_with_tools(
                node_context=node_context,
                conversation_id=request.conversation_id,
                timezone=client_timezone,
                client_tools=client_tools,
                available_commands=request.available_commands,
                skip_warmup_inference=request.skip_warmup_inference
            )

        latency_logger.end_request(request.conversation_id)
        # Return success immediately - LLM warm-up and cache population will happen in background
        return {"status": "success", "conversation_id": request.conversation_id}

    except Exception as e:
        latency_logger.end_request(request.conversation_id)
        logger.error(f"❌ Error starting conversation {request.conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")




@v0_router.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    timing = latency_logger.start_request(request.conversation_id, "voice_command")
    timing.checkpoint("auth_complete")
    start_time = time.time()

    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")
    logger.info(f"Command: '{request.voice_command}'")

    try:
        with timing.measure("cache_get_tools"):
            tools = conversation_cache.get_tools(request.conversation_id)
        if tools is None:
            raise HTTPException(status_code=400, detail="Conversation not initialized for tool-based flow")

        logger.info(f"🔧 Processing as tool-based conversation")
        with timing.measure("process_voice_command_with_tools"):
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id
            )

        # Build response based on stop_reason
        with timing.measure("build_response"):
            stop_reason_raw = result.get("stop_reason", "complete")
            stop_reason = StopReason.COMPLETE
            if isinstance(stop_reason_raw, str):
                try:
                    stop_reason = StopReason(stop_reason_raw)
                except ValueError:
                    logger.warning("⚠️ Unknown stop_reason=%r; defaulting to 'complete'", stop_reason_raw)

            # If stop_reason is ERROR, build an error response
            if stop_reason == StopReason.ERROR:
                error_message = result.get("error", "An internal error occurred")
                logger.error(f"❌ Tool loop returned error: {error_message}")
                error_command = SingleCommandResponse(
                    success=False,
                    command_name=None,
                    parameters=None,
                    errors=VoiceCommandError(
                        type="llm_error",
                        message=error_message
                    )
                )
                response = VoiceCommandResponse(
                    commands=[error_command],
                    request_information=RequestInformation(
                        voice_command=request.voice_command,
                        conversation_id=request.conversation_id
                    ),
                    stop_reason=stop_reason,
                    tool_calls=[],
                    validation_request=None,
                    assistant_message=None
                )
            else:
                response = VoiceCommandResponse(
                    commands=[],  # Empty for tool-based responses
                    request_information=RequestInformation(
                        voice_command=request.voice_command,
                        conversation_id=request.conversation_id
                    ),
                    stop_reason=stop_reason,
                    assistant_message=result.get("assistant_message"),
                    tool_calls=[ToolCall(**tc) for tc in result.get("tool_calls", [])],
                    validation_request=(
                        ValidationRequest(**result["validation_request"])
                        if result.get("validation_request") else None
                    )
                )

        duration = time.time() - start_time
        logger.info(f"✅ Tool-based command processed in {duration:.2f}s, stop_reason={response.stop_reason}")
        latency_logger.end_request(request.conversation_id)
        return response

    except Exception as e:
        latency_logger.end_request(request.conversation_id)
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


@v0_router.post("/voice/command/stream")
async def handle_voice_stream(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service),
):
    """Unified streaming voice command endpoint.

    Runs the full tool-calling pipeline (same as /voice/command) and then
    routes based on the result:

    - **200 audio/raw**: Conversational response — streamed PCM audio with
      format metadata in X-Audio-* headers.
    - **202 application/json**: Tool calls, validation, error, or empty
      response — JSON body matching VoiceCommandResponse shape.
    """
    timing = latency_logger.start_request(request.conversation_id, "voice_command_stream")
    timing.checkpoint("auth_complete")
    start_time = time.time()

    logger.info(
        f"Unified stream command from node: {node_context_provider.node.node_id}"
    )
    logger.info(f"Command: '{request.voice_command}'")

    tools = conversation_cache.get_tools(request.conversation_id)
    if tools is None:
        raise HTTPException(
            status_code=400,
            detail="Conversation not initialized for tool-based flow",
        )

    try:
        # Run the full blocking pipeline (tool filtering, routing, server tools)
        with timing.measure("process_voice_command_with_tools"):
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id,
            )

        stop_reason = result.get("stop_reason", "complete")
        assistant_message = result.get("assistant_message")

        # Conversational response with text → stream as audio
        # Use strip() to avoid treating whitespace-only as "has text"
        if stop_reason == "complete" and assistant_message and assistant_message.strip():
            from app.core.streaming_handler import stream_text_as_audio
            from app.core.clients.tts_client import TTSClient

            tts_client = TTSClient(
                household_id=node_context_provider.household_id,
                node_id=node_context_provider.node.node_id,
            )

            # Get audio format for response headers
            try:
                audio_fmt = await tts_client.get_audio_format()
            except Exception:
                audio_fmt = {"sample_rate": "16000", "channels": "1", "sample_width": "2"}

            duration = time.time() - start_time
            logger.info(f"✅ Streaming audio response in {duration:.2f}s")
            latency_logger.end_request(request.conversation_id)

            # Include the text in a header so the node can display it
            # (e.g., keyboard_listener prints it instead of "(streamed audio)")
            encoded_text = urllib.parse.quote(assistant_message, safe="")

            return StreamingResponse(
                stream_text_as_audio(assistant_message, tts_client),
                status_code=200,
                media_type="audio/raw",
                headers={
                    "X-Audio-Sample-Rate": audio_fmt["sample_rate"],
                    "X-Audio-Channels": audio_fmt["channels"],
                    "X-Audio-Sample-Width": audio_fmt["sample_width"],
                    "X-Assistant-Message": encoded_text,
                },
            )

        # Tool calls, validation, error, or empty → return as JSON
        stop_reason_enum = StopReason.COMPLETE
        try:
            stop_reason_enum = StopReason(stop_reason)
        except ValueError:
            logger.warning("⚠️ Unknown stop_reason=%r; defaulting to 'complete'", stop_reason)

        body = VoiceCommandResponse(
            commands=[],
            request_information=RequestInformation(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id,
            ),
            stop_reason=stop_reason_enum,
            assistant_message=assistant_message,
            tool_calls=[ToolCall(**tc) for tc in result.get("tool_calls", [])],
            validation_request=(
                ValidationRequest(**result["validation_request"])
                if result.get("validation_request") else None
            ),
        )

        duration = time.time() - start_time
        logger.info(
            f"✅ Returning 202 JSON (stop_reason={stop_reason}) in {duration:.2f}s"
        )
        latency_logger.end_request(request.conversation_id)

        return JSONResponse(
            status_code=202,
            content=body.model_dump(mode="json"),
        )

    except Exception as e:
        latency_logger.end_request(request.conversation_id)
        duration = time.time() - start_time
        logger.error(f"❌ Unified stream failed after {duration:.2f}s: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process command: {str(e)}")


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


@v0_router.post("/adapters/train")
async def train_adapter(
    request: AdapterTrainingRequest,
    request_context: Request,
    node_context_provider: NodeContextProvider = Depends(verify_api_key)
):
    """Queue an adapter training job on llm-proxy for this node."""
    def _write_training_jsonl(dataset: dict, job_id: str) -> Optional[Path]:
        try:
            temp_dir = Path("/app/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_path = temp_dir / f"adapter_training_{job_id}.jsonl"

            commands = dataset.get("commands", [])
            with output_path.open("w", encoding="utf-8") as handle:
                for cmd in commands:
                    cmd_name = cmd.get("command_name")
                    for ex in cmd.get("examples", []) or []:
                        record = {
                            "command_name": cmd_name,
                            "voice_command": ex.get("voice_command"),
                            "expected_tool_call": ex.get("expected_tool_call"),
                            "formatted_completion": ex.get("formatted_completion"),
                            "formatted_system_prompt": ex.get("formatted_system_prompt"),
                        }
                        handle.write(json.dumps(record, ensure_ascii=True) + "\n")

            logger.info("📝 Wrote adapter training JSONL: %s", output_path)
            return output_path
        except Exception as exc:
            logger.warning("⚠️ Failed to write adapter training JSONL: %s", exc)
            return None

    def _build_dataset(commands, provider=None):
        dataset_commands = []
        for cmd in commands:
            examples = cmd.examples or []
            if not examples:
                continue
            formatted_examples = []
            for ex in examples:
                tool_call = {
                    "name": cmd.command_name,
                    "arguments": ex.expected_parameters,
                }
                example_dict = {
                    "voice_command": ex.voice_command,
                    "expected_tool_call": tool_call,
                }
                if provider is not None:
                    example_dict["formatted_completion"] = provider.build_training_completion(tool_call)
                    example_dict["formatted_prompt"] = provider.build_training_prompt(ex.voice_command)
                    example_dict["formatted_system_prompt"] = provider.build_training_system_prompt()
                formatted_examples.append(example_dict)
            dataset_commands.append(
                {
                    "command_name": cmd.command_name,
                    "examples": formatted_examples,
                }
            )
        return {"commands": dataset_commands}

    # Resolve the active prompt provider to format training data correctly
    from app.core.prompt_provider_factory import PromptProviderFactory
    provider = PromptProviderFactory.create_provider()

    dataset_payload = _build_dataset(request.available_commands, provider=provider)
    dataset_ref = {"format": "inline-json", "data": dataset_payload}
    if request.dataset_hash:
        dataset_hash = request.dataset_hash
    else:
        hash_input = {
            "base_model_id": request.base_model_id,
            "dataset": dataset_ref,
            "params": request.params.model_dump(exclude_none=True) if request.params else None,
        }
        payload_bytes = json.dumps(hash_input, sort_keys=True).encode("utf-8")
        dataset_hash = hashlib.sha256(payload_bytes).hexdigest()

    job_id = str(uuid4())
    training_jsonl_path = _write_training_jsonl(dataset_payload, job_id)
    callback_url = str(request_context.url_for("adapter_training_callback"))

    callback_token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    callback = {"url": callback_url}
    if callback_token:
        callback["auth_type"] = "bearer"
        callback["token"] = callback_token

    queue_payload = {
        "job_id": job_id,
        "job_type": "adapter_train",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "priority": request.priority or "normal",
        "trace_id": job_id,
        "idempotency_key": dataset_hash,
        "job_type_version": "v1",
        "ttl_seconds": 86400,
        "metadata": {
            "node_id": node_context_provider.node.node_id,
            "provider_name": provider.name if provider else "",
        },
        "request": {
            "node_id": node_context_provider.node.node_id,
            "base_model_id": request.base_model_id,
            "dataset_ref": dataset_ref,
            "dataset_hash": dataset_hash,
            "provider_name": provider.name if provider else "",
            "params": request.params.model_dump(exclude_none=True) if request.params else None,
        },
        "callback": callback,
    }

    # Get LLM proxy URL from service discovery or fallback to env var
    from app.core import service_config
    if service_config.is_initialized():
        llm_proxy_base_url = service_config.get_llm_proxy_url()
    else:
        llm_proxy_base_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")
    queue_url = f"{llm_proxy_base_url.rstrip('/')}/internal/queue/enqueue"
    headers = {}
    internal_token = os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal_token:
        headers["X-Internal-Token"] = internal_token

    try:
        result = await post(url=queue_url, json_data=queue_payload, headers=headers)
    except Exception as e:
        logger.error("❌ Failed to enqueue adapter training job: %s", e)
        raise HTTPException(status_code=502, detail="Failed to enqueue adapter training job")

    return {
        "status": "queued",
        "job_id": job_id,
        "dataset_hash": dataset_hash,
        "training_jsonl_path": str(training_jsonl_path) if training_jsonl_path else None,
        "llm_proxy_response": result,
    }


@v0_router.post("/adapters/jobs/callback", name="adapter_training_callback")
async def adapter_training_callback(request: Request):
    """Receive llm-proxy training job callbacks and update node adapter_hash on success."""
    from app.models import Node

    callback_token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    if callback_token:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {callback_token}":
            raise HTTPException(status_code=401, detail="Unauthorized callback")

    payload = await request.json()
    job_id = payload.get("job_id")
    status = payload.get("status")
    logger.info("📬 Adapter training callback received: job_id=%s status=%s", job_id, status)

    # On success, update the node's adapter_hash
    if status == "succeeded":
        result = payload.get("result", {})
        artifact_metadata = result.get("artifact_metadata", {})
        node_id = artifact_metadata.get("node_id")
        adapter_hash = artifact_metadata.get("dataset_hash")

        if node_id and adapter_hash:
            # Get a database session and update the node
            from app.db import get_session_local
            SessionLocal = get_session_local()
            db = SessionLocal()
            try:
                node = db.query(Node).filter(Node.node_id == node_id).first()
                if node:
                    old_hash = node.adapter_hash
                    node.adapter_hash = adapter_hash
                    db.commit()
                    logger.info(
                        "✅ Updated node %s adapter_hash: %s -> %s",
                        node_id,
                        old_hash[:8] if old_hash else "None",
                        adapter_hash[:8]
                    )
                else:
                    logger.warning("⚠️ Node %s not found for adapter_hash update", node_id)
            except Exception as e:
                logger.error("❌ Failed to update node adapter_hash: %s", e)
                db.rollback()
            finally:
                db.close()
        else:
            logger.warning(
                "⚠️ Missing node_id or adapter_hash in callback: node_id=%s adapter_hash=%s",
                node_id,
                adapter_hash[:8] if adapter_hash else None
            )
    elif status == "failed":
        error = payload.get("error", {})
        logger.error("❌ Adapter training failed for job %s: %s", job_id, error.get("message"))

    return {"status": "ok"}


@v0_router.post("/deep-research/callback", name="deep_research_callback")
async def deep_research_callback(request: Request):
    """Receive LLM queue callback for deep research summarization."""
    callback_token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    if callback_token:
        auth_header = request.headers.get("authorization", "")
        if auth_header != f"Bearer {callback_token}":
            raise HTTPException(status_code=401, detail="Unauthorized callback")

    payload = await request.json()
    job_id = payload.get("job_id")
    status = payload.get("status")
    query = payload.get("metadata", {}).get("query", "unknown")
    logger.info("📬 Deep research callback: job_id=%s status=%s query=%r", job_id, status, query)

    from app.services.deep_research_service import handle_summarization_callback
    try:
        await handle_summarization_callback(payload)
    except Exception as e:
        logger.error("❌ Deep research callback handling failed: %s", e, exc_info=True)

    return {"status": "ok"}


async def _maybe_push_actions_to_inbox(
    tool_results: list[dict],
    node_context_provider: NodeContextProvider,
) -> None:
    """If any tool result contains actions, push a confirmation to the inbox."""
    for tr in tool_results:
        output = tr.get("output")
        if not isinstance(output, dict):
            continue
        context = output.get("context")
        if not isinstance(context, dict):
            continue
        actions = context.get("actions")
        if not actions or not isinstance(actions, list):
            continue

        # Found actions — push to inbox
        draft = context.get("draft", {})
        preview = context.get("preview", "")
        message = context.get("message", "")
        command_name = context.get("command_name", "unknown")

        # Generic title/summary — commands provide inbox_title/inbox_summary
        title = context.get("inbox_title") or f"Confirm: {command_name}"
        summary = context.get("inbox_summary") or message or preview[:100]

        try:
            from app.services.inbox_notification_service import push_confirmation_to_inbox

            node = node_context_provider.node
            household_id = node.household_id if hasattr(node, "household_id") else ""
            if not household_id:
                logger.warning("Cannot push confirmation: no household_id on node")
                return

            await push_confirmation_to_inbox(
                household_id=household_id,
                user_id=None,
                node_id=node.node_id,
                title=title,
                summary=summary,
                body=preview,
                command_name=command_name,
                actions=actions,
                draft=draft,
            )
        except Exception as e:
            logger.warning("Failed to push actions to inbox: %s", e)


@v0_router.post("/voice/command/continue", response_model=VoiceCommandResponse)
async def continue_voice_command(
    request: ToolResultRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    """Continue a conversation by providing tool execution results."""
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
        
        # Check tool results for actionable responses (e.g. email send confirmation)
        # and push them to the inbox so the mobile app can render buttons.
        await _maybe_push_actions_to_inbox(
            tool_results=tool_results,
            node_context_provider=node_context_provider,
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
