import asyncio
import time
import logging
import os
import json
import urllib.parse
from pathlib import Path
import hashlib
import hmac
from uuid import uuid4
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.request_models.conversation_end_request import ConversationEndRequest
from app.request_models.tool_result_request import ToolResultRequest
from app.request_models.tool_router_training_request import ToolRouterTrainingRequest
from app.request_models.adapter_training_request import AdapterTrainingRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse, RequestInformation, StopReason, ToolCall, ValidationRequest
from app.debug_setup import setup_debugger
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from app.core.conversation_cache import conversation_cache
from app.core.errors import ConversationPreconditionError
from app.core.utils.latency_logger import latency_logger
from . import admin, chat, date_context, node_settings, provisioning
from app.api import ambient_noise, media, node_commands, test_commands
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

    # Async-job callback auth posture — surface a missing token loudly at boot (fail-closed).
    if not os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN"):
        if _insecure_callbacks_allowed():
            logger.warning(
                "⚠️ JARVIS_ADAPTER_CALLBACK_TOKEN is unset and JARVIS_ALLOW_INSECURE_CALLBACKS is "
                "enabled — adapter/deep-research/memory-extraction callbacks are UNAUTHENTICATED. "
                "Do NOT run this in production."
            )
        else:
            logger.error(
                "🔒 JARVIS_ADAPTER_CALLBACK_TOKEN is unset — async-job callbacks will be REJECTED "
                "(503). Set the token, or set JARVIS_ALLOW_INSECURE_CALLBACKS=1 for trusted local dev."
            )

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
    settings_service = get_settings_service()
    _settings_router = create_settings_router(
        service=settings_service,
        auth_dependency=create_combined_auth(auth_url),
        write_auth_dependency=create_superuser_auth(auth_url),
    )
    app.include_router(_settings_router, prefix="/settings", tags=["settings"])

    # Schedule passive memory extraction
    async def _periodic_memory_extraction() -> None:
        while True:
            try:
                interval = int(settings_service.get("memory.extraction_interval_seconds") or 300)
            except (TypeError, ValueError):
                interval = 300
            await asyncio.sleep(interval)
            try:
                enabled = settings_service.get("memory.extraction_enabled")
                if enabled is False or str(enabled).lower() in ("false", "0", "no"):
                    continue
                from app.services.memory_extraction_service import run_extraction_batch
                await run_extraction_batch()
            except Exception as e:
                logger.warning("Memory extraction batch failed: %s", e)

    asyncio.create_task(_periodic_memory_extraction())

    # Schedule transcript TTL cleanup (daily)
    async def _periodic_transcript_cleanup() -> None:
        while True:
            await asyncio.sleep(86400)
            try:
                ttl_days = int(settings_service.get("memory.transcript_ttl_days") or 7)
                from app.services.transcript_service import TranscriptService
                db = SessionLocal()
                try:
                    removed = TranscriptService(db).cleanup_expired(ttl_days)
                    if removed:
                        logger.info("Transcript cleanup removed %d expired entries", removed)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Transcript cleanup failed: %s", e)

    asyncio.create_task(_periodic_transcript_cleanup())

    # Schedule trace TTL cleanup (hourly)
    async def _periodic_trace_cleanup() -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                ttl_days = int(settings_service.get("tracing.retention_days") or 7)
                from app.models import RequestTrace as RT
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(days=ttl_days)
                db = SessionLocal()
                try:
                    removed = db.query(RT).filter(RT.created_at < cutoff).delete()
                    db.commit()
                    if removed:
                        logger.info("Trace cleanup removed %d expired entries", removed)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Trace cleanup failed: %s", e)

    asyncio.create_task(_periodic_trace_cleanup())

    # Schedule expired memory cleanup (agent-injected memories have TTL)
    async def _periodic_memory_cleanup() -> None:
        await asyncio.sleep(60)
        while True:
            await asyncio.sleep(1800)  # every 30 min
            try:
                from app.services.memory_service import MemoryService
                db = SessionLocal()
                try:
                    cleaned = MemoryService(db).cleanup_expired()
                    if cleaned:
                        logger.info("Cleaned up %d expired memories", cleaned)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Memory cleanup failed: %s", e)

    asyncio.create_task(_periodic_memory_cleanup())

    # Node-update task sweeper — fails any update task that's clearly dead.
    # Two thresholds:
    #   - ANY_STATE_CEILING (created_at): 15 min outer ceiling regardless
    #     of state. Catches pending tasks for offline nodes and dispatched
    #     tasks whose installer never started reporting.
    #   - IN_PROGRESS_NO_PROGRESS (updated_at, in_progress only): 10 min
    #     since the last state transition. reconcile_open_task deliberately
    #     does NOT bump updated_at on heartbeats that still report the OLD
    #     version, so a stuck in_progress task ages out fast even if the
    #     node keeps heartbeating.
    # Install.sh on a Pi Zero typically finishes in 3-6 min; 10 min in
    # in_progress and 15 min total are both well above that.
    from datetime import datetime, timedelta
    from sqlalchemy import and_, or_

    ANY_STATE_CEILING = timedelta(minutes=15)
    IN_PROGRESS_NO_PROGRESS = timedelta(minutes=10)

    async def _periodic_node_task_timeout():
        while True:
            await asyncio.sleep(120)
            try:
                from app.db import SessionLocal
                from app.models import NodeTask
                db = SessionLocal()
                try:
                    now = datetime.utcnow()
                    stale = (
                        db.query(NodeTask)
                        .filter(
                            NodeTask.state.in_(["pending", "dispatched", "in_progress"]),
                            or_(
                                NodeTask.created_at < now - ANY_STATE_CEILING,
                                and_(
                                    NodeTask.state == "in_progress",
                                    NodeTask.updated_at < now - IN_PROGRESS_NO_PROGRESS,
                                ),
                            ),
                        )
                        .all()
                    )
                    for task in stale:
                        task.state = "failed"
                        task.error_message = (
                            task.error_message
                            or f"Timeout: no heartbeat confirming {task.target_version}"
                        )
                        task.finished_at = now
                        task.updated_at = now
                    if stale:
                        db.commit()
                        logger.info("Marked %d stale node update tasks as failed", len(stale))
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Node task timeout sweep failed: %s", e)

    asyncio.create_task(_periodic_node_task_timeout())

    # Phase 5 — adapter auto-training loop. Off by default; flip
    # `adapter.auto_train_enabled` in settings to activate. Tick interval,
    # example threshold, and cooldown all settings-driven.
    async def _periodic_adapter_training() -> None:
        while True:
            try:
                interval = int(settings_service.get("adapter.auto_train_interval_seconds") or 3600)
            except (TypeError, ValueError):
                interval = 3600
            await asyncio.sleep(max(60, interval))
            try:
                enabled = settings_service.get("adapter.auto_train_enabled")
                if enabled is False or str(enabled).lower() in ("false", "0", "no", "none"):
                    continue
                from app.services.adapter_scheduler import run_scheduled_training
                result = await asyncio.to_thread(run_scheduled_training, settings_service)
                logger.info(
                    "adapter scheduler tick: considered=%d skipped=%s deployed=%s eval_failed=%s errored=%s",
                    len(result.considered), result.skipped, result.deployed,
                    result.eval_failed, list(result.errored.keys()),
                )
            except Exception as e:
                logger.warning("Adapter scheduler tick failed: %s", e)

    asyncio.create_task(_periodic_adapter_training())

    # Routine scheduler — fire cron/interval routines on their target node.
    async def _periodic_routine_scheduler() -> None:
        while True:
            try:
                interval = int(settings_service.get("routines.scheduler_interval_seconds") or 30)
            except (TypeError, ValueError):
                interval = 30
            await asyncio.sleep(max(10, interval))
            try:
                enabled = settings_service.get("routines.scheduler_enabled")
                if enabled is not True and str(enabled).lower() not in ("true", "1", "yes"):
                    continue
                from app.services.routine_scheduler import run_due_routines
                db = SessionLocal()
                try:
                    fired = await run_due_routines(db)
                    if fired:
                        logger.info("Routine scheduler fired %d routine(s)", fired)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Routine scheduler tick failed: %s", e)

    asyncio.create_task(_periodic_routine_scheduler())

    # Routine-execution audit TTL cleanup (daily).
    async def _periodic_routine_execution_cleanup() -> None:
        while True:
            await asyncio.sleep(86400)
            try:
                ttl_days = int(settings_service.get("routines.execution_ttl_days") or 7)
                from app.models import RoutineExecution as RE
                from datetime import datetime, timedelta
                cutoff = datetime.utcnow() - timedelta(days=ttl_days)
                db = SessionLocal()
                try:
                    removed = db.query(RE).filter(RE.started_at < cutoff).delete()
                    db.commit()
                    if removed:
                        logger.info("Routine-execution cleanup removed %d rows", removed)
                finally:
                    db.close()
            except Exception as e:
                logger.warning("Routine-execution cleanup failed: %s", e)

    asyncio.create_task(_periodic_routine_execution_cleanup())

    # Enrich llm.interface with available prompt providers dynamically
    if "llm.interface" in settings_service.definitions:
        from app.core.prompt_provider_factory import PromptProviderFactory
        original_list_all = settings_service.list_all

        def list_all_with_provider_options(**kwargs):  # type: ignore[no-untyped-def]
            results = original_list_all(**kwargs)
            for setting in results:
                if setting["key"] == "llm.interface":
                    try:
                        setting["options"] = PromptProviderFactory.get_available_providers()
                    except Exception:
                        pass
            return results

        settings_service.list_all = list_all_with_provider_options  # type: ignore[assignment]


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

# Unauthenticated node endpoints (e.g. factory-reset verification)
app.include_router(admin.node_public_router, prefix="/api/v0", tags=["node-public"])

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

# Include ambient-noise calibration router (mobile-triggered, node-fulfilled)
app.include_router(ambient_noise.router, prefix="/api/v0", tags=["ambient-noise"])

# Include test commands router (app-to-app auth)
app.include_router(test_commands.router, prefix="/api/v0", tags=["testing"])

# Include package install router (Pantry store → node install via MQTT)
from app.api import package_install
app.include_router(package_install.router, prefix="/api/v0", tags=["package-install"])

# Include test install router (Forge share code → node test install)
from app.api import test_install
app.include_router(test_install.router, prefix="/api/v0", tags=["test-install"])

# Include memory CRUD router
from app.api import memories
app.include_router(memories.router, prefix="/api/v0", tags=["memories"])

# Mobile-facing memory CRUD (user JWT, role-scoped)
from app.api import mobile_memories
app.include_router(mobile_memories.router, prefix="/api/v0", tags=["mobile-memories"])

# Include transcript feedback router (Phase 1 — mobile rating UI)
from app.api import transcripts
app.include_router(transcripts.router, prefix="/api/v0", tags=["transcripts"])

# Self-scoped account-data purge (user JWT) — account-deletion contract.
# jarvis-auth fans out to DELETE /api/v0/me/data during DELETE /auth/me.
from app.api import me as me_api
app.include_router(me_api.router, prefix="/api/v0", tags=["me"])

# Include adapter proposal router (Phase 7.1 — user-approved deploy flow)
from app.api import adapters as adapters_api
app.include_router(adapters_api.router, prefix="/api/v0", tags=["adapters"])

# Wake-response text generator — moved out of jarvis-tts so provider-aware
# sanitation (think-block stripping, etc.) happens at the only layer that
# owns the prompt provider.
from app.api import wake_response as wake_response_api
app.include_router(wake_response_api.router, prefix="/api/v0", tags=["wake-response"])

# Include OAuth session management router
from app.api import oauth
app.include_router(oauth.router, prefix="/api/v0", tags=["oauth"])

# Include camera streaming router (go2rtc proxy)
from app.api import cameras
app.include_router(cameras.router, prefix="/api/v0", tags=["cameras"])

# Include agent utility endpoints (news, calendar for node-side agents)
from app.api import agents
app.include_router(agents.router, prefix="/api/v0", tags=["agents"])

# Include node update router (mobile triggers update, CC dispatches via heartbeat)
from app.api import node_updates
app.include_router(node_updates.router, prefix="/api/v0", tags=["node-updates"])

# Include Bluetooth management router (scan, pair, disconnect via MQTT)
from app.api import bluetooth
app.include_router(bluetooth.router, prefix="/api/v0", tags=["bluetooth"])

# Include interactive-notification callback router (mobile tap -> node @callback)
from app.api import callbacks
app.include_router(callbacks.router, prefix="/api/v0", tags=["callbacks"])

# Include request trace endpoints (admin + mobile)
from app.api import traces as traces_api
app.include_router(traces_api.admin_router, prefix="/api/v0/admin", tags=["traces-admin"])
app.include_router(traces_api.mobile_router, prefix="/api/v0/mobile", tags=["traces-mobile"])

# Include mobile chat, audio, and node tools endpoints (JWT auth)
from app.api import mobile_chat, mobile_audio, mobile_voice_profiles, node_tools
app.include_router(mobile_chat.router, prefix="/api/v0/mobile", tags=["mobile-chat"])
app.include_router(mobile_audio.router, prefix="/api/v0/mobile", tags=["mobile-audio"])
app.include_router(mobile_voice_profiles.router, prefix="/api/v0/mobile", tags=["mobile-voice-profiles"])
app.include_router(node_tools.router, prefix="/api/v0/mobile", tags=["node-tools"])

# Routines: server-owned per-household routines (mobile CRUD + node pull).
# Router carries both /households/{id}/routines and /nodes/{id}/routines paths.
from app.api import routines
app.include_router(routines.router, prefix="/api/v0", tags=["routines"])

# Include mobile command-data browser router (JWT auth, MQTT round-trip)
from app.api import mobile_command_data
app.include_router(mobile_command_data.router, prefix="/api/v0/mobile", tags=["mobile-command-data"])

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
    timing.source = "node"
    timing.node_id = node_context_provider.node.node_id
    timing.household_id = node_context_provider.household_id
    timing.checkpoint("auth_complete")

    try:
        # Build node context from node properties (ignore most client-provided context for security)
        node_context = {
            "room": node_context_provider.node.room,
            "node_id": node_context_provider.node.node_id,
            "user": node_context_provider.node.user,
            "voice_mode": node_context_provider.node.voice_mode,
            # Adapters orphaned 2026-04-25: training delivered marginal gains
            # and the per-call resolution (incompatibility check + S3 head)
            # was costing ~80-150ms on every LLM call. Leave the field for
            # schema compatibility but never populate it.
            "adapter_hash": None,
            "household_id": node_context_provider.household_id,
        }

        # Phase 5: household-level adapter deployment takes precedence over the
        # legacy per-node adapter_hash column. A household-active adapter wins
        # because it's the one gated by the scheduler's eval run.
        # ORPHANED — see adapter_hash comment above.
        if False and node_context.get("household_id"):
            try:
                from app.services.adapter_registry import AdapterRegistry
                from app.db import get_session_local
                _s = get_session_local()()
                try:
                    _reg = AdapterRegistry(_s)
                    _active = _reg.get_active(node_context["household_id"])
                    if _active is not None:
                        node_context["adapter_hash"] = _active.adapter_hash
                        logger.info(
                            "🎯 Applied household adapter: hash=%s… pass_rate=%s trained_on=%s",
                            _active.adapter_hash[:12],
                            _active.pass_rate,
                            _active.trained_on_examples,
                        )
                finally:
                    _s.close()
            except Exception as e:
                logger.warning("active_adapter lookup failed: %s", e)

        # Phase 2 eval gate: test-mode adapter override. Only honored when the
        # server was started with JARVIS_TEST_MODE=1 — otherwise ignored entirely.
        # Overrides both the Phase 5 household value and the per-node fallback.
        # enabled=False forces a baseline run (no adapter at all) for A/B eval.
        if request.adapter_settings and os.getenv("JARVIS_TEST_MODE") == "1":
            override = request.adapter_settings
            if override.enabled and override.hash:
                node_context["adapter_hash"] = override.hash
                logger.info(
                    f"🧪 Test-mode adapter override applied: hash={override.hash[:12]}… "
                    f"scale={override.scale} enabled={override.enabled}"
                )
            else:
                node_context["adapter_hash"] = None
                logger.info("🧪 Test-mode adapter override: BASELINE (no adapter)")
        elif request.adapter_settings:
            logger.warning(
                "adapter_settings on /conversation/start ignored "
                "(JARVIS_TEST_MODE not enabled)"
            )

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

            # Carry over the node's most-recently-shown list so we can re-inject
            # the RECENTLY SHOWN block for a follow-up ("mark those as read") on a
            # fresh conversation (pre-route / re-wake), where there's no tool
            # result to stash from. Trusted — it comes from the authenticated node.
            if "recently_shown_items" in request.node_context:
                node_context["recently_shown_items"] = request.node_context["recently_shown_items"]

            # Extract speaker identity from voice recognition
            speaker_user_id = request.node_context.get("speaker_user_id")
            speaker_confidence = request.node_context.get("speaker_confidence")
            node_id_for_stickiness = node_context_provider.node.node_id

            from app.core.utils.speaker_stickiness import (
                inherit_speaker_for_node,
                record_speaker_for_node,
            )

            if speaker_user_id is None:
                # Short utterance — try inheriting a recent high-confidence
                # speaker from the same node. See speaker_stickiness.py.
                inherited = inherit_speaker_for_node(node_id_for_stickiness)
                if inherited is not None:
                    logger.info(
                        f"🎤 Speaker inherited from recent identification "
                        f"on node={node_id_for_stickiness}: user_id={inherited}"
                    )
                    speaker_user_id = inherited
            else:
                # Confident new ID — remember it so the next short follow-up
                # on this node can inherit it.
                record_speaker_for_node(
                    node_id_for_stickiness,
                    speaker_user_id,
                    float(speaker_confidence) if speaker_confidence is not None else None,
                )

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


@v0_router.post("/conversation/end")
async def end_conversation(
    request: ConversationEndRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
):
    """Mark a conversation as ended.

    The node calls this when its wake cycle finishes (follow-up window
    closed, no further turns expected without a new wake word). Right now
    the only thing this does is wipe the node's speaker stickiness so a
    later wake event by a different speaker doesn't inherit the previous
    speaker's identity. Future lifecycle cleanups (cache eviction,
    transcript finalization, etc.) can attach here.
    """
    from app.core.utils.speaker_stickiness import reset_node_history

    node_id = node_context_provider.node.node_id
    reset_node_history(node_id)
    logger.info(
        f"📤 Conversation ended: id={request.conversation_id} node={node_id}"
    )
    return {"status": "ok", "conversation_id": request.conversation_id}


async def _log_transcript(
    conversation_id: str,
    voice_command: str,
    assistant_message: str | None,
    tool_calls: list | None,
) -> None:
    """Fire-and-forget: log a conversation transcript for passive memory extraction."""
    try:
        node_context = conversation_cache.get_node_context(conversation_id)
        if not node_context:
            return
        speaker_user_id = node_context.get("speaker_user_id")
        household_id = node_context.get("household_id")
        if not speaker_user_id or not household_id:
            return

        from app.db import get_session_local
        from app.services.transcript_service import TranscriptService

        SessionLocal = get_session_local()
        db = SessionLocal()
        try:
            svc = TranscriptService(db)
            svc.log_transcript(
                user_id=speaker_user_id,
                household_id=household_id,
                conversation_id=conversation_id,
                user_message=voice_command,
                assistant_message=assistant_message,
                tool_calls=tool_calls,
            )
        finally:
            db.close()
    except Exception as e:
        logger.warning("Failed to log transcript (non-fatal): %s", e)


@v0_router.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service)
):
    timing = latency_logger.start_request(request.conversation_id, "voice_command")
    timing.source = "node"
    timing.node_id = node_context_provider.node.node_id
    timing.household_id = node_context_provider.household_id
    timing.user_command = request.voice_command
    timing.checkpoint("auth_complete")
    start_time = time.time()

    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")
    logger.info(f"Command: '{request.voice_command}'")

    try:
        with timing.measure("cache_get_tools"):
            tools = conversation_cache.get_tools(request.conversation_id)
        if tools is None:
            raise HTTPException(status_code=422, detail="Conversation not initialized for tool-based flow")

        logger.info(f"🔧 Processing as tool-based conversation")
        with timing.measure("process_voice_command_with_tools"):
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id,
                pre_wake_speech_seconds=request.pre_wake_speech_seconds,
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
                # Strip [Tool data: ...] prefix (history context, not for display)
                _msg = result.get("assistant_message") or ""
                if "[Tool data:" in _msg:
                    _idx = _msg.find("]\n\n")
                    if _idx != -1:
                        _msg = _msg[_idx + 3:].strip()

                response = VoiceCommandResponse(
                    commands=[],  # Empty for tool-based responses
                    request_information=RequestInformation(
                        voice_command=request.voice_command,
                        conversation_id=request.conversation_id
                    ),
                    stop_reason=stop_reason,
                    assistant_message=_msg or None,
                    reasoning=result.get("reasoning"),
                    tool_calls=[ToolCall(**tc) for tc in result.get("tool_calls", [])],
                    validation_request=(
                        ValidationRequest(**result["validation_request"])
                        if result.get("validation_request") else None
                    )
                )

        duration = time.time() - start_time
        logger.info(f"✅ Tool-based command processed in {duration:.2f}s, stop_reason={response.stop_reason}")
        latency_logger.end_request(request.conversation_id)

        # Fire-and-forget: log transcript for passive memory extraction
        import asyncio
        asyncio.create_task(_log_transcript(
            conversation_id=request.conversation_id,
            voice_command=request.voice_command,
            assistant_message=result.get("assistant_message"),
            tool_calls=result.get("tool_calls"),
        ))

        return response

    except HTTPException:
        # Whole-request rejection (e.g. precondition not met) — surface the
        # status to the client; do NOT swallow it into a 200 + errors body.
        latency_logger.end_request(request.conversation_id)
        raise
    except ConversationPreconditionError as e:
        # Un-started / expired conversation: the whole request was rejected,
        # no command executed. Surface as 422, not 200.
        latency_logger.end_request(request.conversation_id)
        raise HTTPException(status_code=422, detail=str(e))
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


class VoiceAcknowledgeRequest(BaseModel):
    voice_command: str


@v0_router.post("/voice/acknowledge")
async def voice_acknowledge(
    request: VoiceAcknowledgeRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
):
    """Instant keyword-matched acknowledgment for voice commands.

    Nodes call this in parallel with /voice/command/stream to get a quick
    acknowledgment to speak while the main pipeline processes. No LLM call —
    avoids competing for llama.cpp's inference slot or evicting the prefix cache.
    """
    from app.services.acknowledgment_service import generate_acknowledgment

    text = generate_acknowledgment(request.voice_command)
    return {"text": text}


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
    timing.source = "node"
    timing.node_id = node_context_provider.node.node_id
    timing.household_id = node_context_provider.household_id
    timing.user_command = request.voice_command
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
        from app.core.clients.tts_client import TTSClient

        tts_client = TTSClient(
            household_id=node_context_provider.household_id,
            node_id=node_context_provider.node.node_id,
        )

        # --- Fast path: router-gated streaming LLM → TTS ---
        # For conversational/search queries the router predicts with high
        # confidence, bypass the blocking tool loop entirely and stream
        # LLM tokens → sentences → TTS audio in real time.
        streaming_audio = await model_service.try_stream_voice_response(
            conversation_id=request.conversation_id,
            voice_command=request.voice_command,
            tts_client=tts_client,
            speaker_user_id=request.speaker_user_id,
            pre_wake_speech_seconds=request.pre_wake_speech_seconds,
        )
        if streaming_audio is not None:
            try:
                audio_fmt = await tts_client.get_audio_format()
            except Exception:
                audio_fmt = {"sample_rate": "16000", "channels": "1", "sample_width": "2"}

            duration = time.time() - start_time
            logger.info(f"🚀 Streaming LLM→TTS response in {duration:.2f}s")
            latency_logger.end_request(request.conversation_id)

            return StreamingResponse(
                streaming_audio,
                status_code=200,
                media_type="audio/raw",
                headers={
                    "X-Audio-Sample-Rate": audio_fmt["sample_rate"],
                    "X-Audio-Channels": audio_fmt["channels"],
                    "X-Audio-Sample-Width": audio_fmt["sample_width"],
                    "X-Assistant-Message": "",
                },
            )

        # --- Tool-streaming path: server-side tool then streamed final answer ---
        # For commands the router predicts will need server-side tool execution
        # (news, weather summaries, etc.), run iter 1 + tool exec blocking and
        # stream iter 2's prose response to TTS as sentences complete. Saves
        # 2-3s perceived latency on chatty answers vs the blocking path.
        # Gated by JARVIS_STREAM_TOOL_RESPONSES env flag (default off).
        streaming_audio = await model_service.try_stream_voice_response_with_tools(
            conversation_id=request.conversation_id,
            voice_command=request.voice_command,
            tts_client=tts_client,
            speaker_user_id=request.speaker_user_id,
            pre_wake_speech_seconds=request.pre_wake_speech_seconds,
        )
        if streaming_audio is not None:
            try:
                audio_fmt = await tts_client.get_audio_format()
            except Exception:
                audio_fmt = {"sample_rate": "16000", "channels": "1", "sample_width": "2"}

            duration = time.time() - start_time
            logger.info(f"🚀 Tool-stream LLM→TTS response in {duration:.2f}s")
            latency_logger.end_request(request.conversation_id)

            return StreamingResponse(
                streaming_audio,
                status_code=200,
                media_type="audio/raw",
                headers={
                    "X-Audio-Sample-Rate": audio_fmt["sample_rate"],
                    "X-Audio-Channels": audio_fmt["channels"],
                    "X-Audio-Sample-Width": audio_fmt["sample_width"],
                    "X-Assistant-Message": "",
                },
            )

        # --- Standard path: blocking tool execution + TTS ---
        with timing.measure("process_voice_command_with_tools"):
            result = await model_service.process_voice_command_with_tools(
                voice_command=request.voice_command,
                conversation_id=request.conversation_id,
                speaker_user_id=request.speaker_user_id,
                pre_wake_speech_seconds=request.pre_wake_speech_seconds,
            )

        stop_reason = result.get("stop_reason", "complete")
        assistant_message = result.get("assistant_message")

        # Conversational response with text → stream as audio
        # Use strip() to avoid treating whitespace-only as "has text"
        if stop_reason == "complete" and assistant_message and assistant_message.strip():
            from app.core.streaming_handler import stream_text_as_audio

            # Get audio format for response headers
            try:
                audio_fmt = await tts_client.get_audio_format()
            except Exception:
                audio_fmt = {"sample_rate": "16000", "channels": "1", "sample_width": "2"}

            duration = time.time() - start_time
            logger.info(f"✅ Streaming audio response in {duration:.2f}s")
            timing.assistant_message = assistant_message
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
        timing.assistant_message = assistant_message
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


@v0_router.post("/adapters/train", status_code=202)
async def train_adapter(
    request: AdapterTrainingRequest,
    request_context: Request,
    node_context_provider: NodeContextProvider = Depends(verify_api_key)
):
    """Accept an adapter training job and run expansion/enqueue in background.

    Returns 202 immediately with a job_id. The actual work (LLM-example
    expansion + provider formatting + llm-proxy enqueue) runs as an
    asyncio task so large expansions don't hold the HTTP connection.
    Caller polls llm-proxy's /v1/training/status/{job_id} for progress,
    or receives the completion callback configured below.
    """
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

    # Generate job_id up front so we can return it immediately — the rest
    # of the work (expansion, dataset build, llm-proxy enqueue) runs in
    # a background task because expansion can take minutes.
    job_id = str(uuid4())
    callback_url = str(request_context.url_for("adapter_training_callback"))
    node_id_snapshot = node_context_provider.node.node_id

    async def _run_training_pipeline() -> None:
        """Background: expand → format → enqueue.

        Errors are logged and surfaced via the llm-proxy job-status
        endpoint (node polls /v1/training/status/{job_id}). If enqueue
        itself fails, we write the failure to adapter_history so Phase 5's
        rollback logic remains self-consistent.
        """
        # 1) Optional LLM expansion (Phase 3.5) — gated by setting.
        try:
            from app.services.adapter_example_expansion import (
                ExpansionConfig,
                expand_examples,
            )
            from app.services.settings_service import get_settings_service
            _s = get_settings_service()
            expansion_cfg = ExpansionConfig(
                enabled=bool(_s.get("adapter.expansion_enabled") or False),
                expand_n=int(_s.get("adapter.expansion_count_per_canonical") or 2),
                max_canonicals=int(_s.get("adapter.expansion_max_canonicals") or 10),
            )
            if expansion_cfg.enabled:
                expansion_stats = await expand_examples(request.available_commands, expansion_cfg)
                total_baked = sum(s.baked_count for s in expansion_stats)
                total_expanded = sum(s.expanded_count for s in expansion_stats)
                logger.info(
                    "adapter training %s expansion: baked=%d expanded=%d total=%d across %d commands",
                    job_id[:8], total_baked, total_expanded,
                    total_baked + total_expanded, len(expansion_stats),
                )
        except Exception as e:
            logger.warning("adapter expansion failed (falling back to baked): %s", e)

        # 2) Resolve active provider + format dataset.
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

        _write_training_jsonl(dataset_payload, job_id)

        # 3) Build + send the queue payload.
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
                "node_id": node_id_snapshot,
                "provider_name": provider.name if provider else "",
            },
            "request": {
                "node_id": node_id_snapshot,
                "base_model_id": request.base_model_id,
                "dataset_ref": dataset_ref,
                "dataset_hash": dataset_hash,
                "provider_name": provider.name if provider else "",
                "params": request.params.model_dump(exclude_none=True) if request.params else None,
            },
            "callback": callback,
        }

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
            logger.info("✅ Adapter training enqueued job_id=%s dataset_hash=%s response=%s",
                        job_id, dataset_hash[:12], result)
        except Exception as e:
            logger.error("❌ Failed to enqueue adapter training job %s: %s", job_id, e)

    # Fire-and-forget — caller gets 202 immediately.
    asyncio.create_task(_run_training_pipeline())

    return {
        "status": "accepted",
        "job_id": job_id,
        "poll_url": f"/v1/training/status/{job_id}",
    }


def _insecure_callbacks_allowed() -> bool:
    """True only if an operator has explicitly opted into UNAUTHENTICATED async-job callbacks.

    Intended for trusted local/dev environments that don't set a token. Never enable in
    production — it disables the only auth on endpoints that mutate node state / inbox content.
    """
    return os.getenv("JARVIS_ALLOW_INSECURE_CALLBACKS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _verify_callback_auth(request: Request) -> None:
    """Authenticate an inbound async-LLM-job callback (FAIL-CLOSED).

    Shared by /adapters/jobs/callback, /deep-research/callback, /memory-extraction/callback
    (and the planned /daily-briefing/callback). These callbacks set node adapter_hash, write
    inbox content, and fan out household pushes, so an unauthenticated path is a real
    injection / spam vector.

    Posture:
      - JARVIS_ADAPTER_CALLBACK_TOKEN set  → require an exact `Bearer <token>` match (401 otherwise).
      - token unset                        → REJECT with 503 (misconfiguration is loud), UNLESS
                                             JARVIS_ALLOW_INSECURE_CALLBACKS is explicitly enabled.

    This replaces the previous fail-OPEN behaviour where an unset token silently accepted any
    caller (the outbound enqueue side only attaches the bearer when the token is set, so a
    token-less deployment is now loudly closed instead of silently open).
    """
    callback_token = os.getenv("JARVIS_ADAPTER_CALLBACK_TOKEN")
    if not callback_token:
        if _insecure_callbacks_allowed():
            return
        raise HTTPException(
            status_code=503,
            detail="Callback authentication is not configured (set JARVIS_ADAPTER_CALLBACK_TOKEN)",
        )
    auth_header = request.headers.get("authorization", "")
    # Constant-time compare to avoid leaking the token via response timing.
    if not hmac.compare_digest(auth_header, f"Bearer {callback_token}"):
        raise HTTPException(status_code=401, detail="Unauthorized callback")


@v0_router.post("/adapters/jobs/callback", name="adapter_training_callback")
async def adapter_training_callback(request: Request):
    """Receive llm-proxy training job callbacks and update node adapter_hash on success."""
    from app.models import Node

    _verify_callback_auth(request)

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
    _verify_callback_auth(request)

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


@v0_router.post("/memory-extraction/callback", name="memory_extraction_callback")
async def memory_extraction_callback(request: Request):
    """Receive LLM queue callback for passive memory extraction."""
    _verify_callback_auth(request)

    payload = await request.json()
    job_id = payload.get("job_id")
    status = payload.get("status")
    user_id = payload.get("metadata", {}).get("user_id", "?")
    logger.info("Memory extraction callback: job_id=%s status=%s user_id=%s", job_id, status, user_id)

    from app.services.memory_extraction_service import handle_extraction_callback
    try:
        await handle_extraction_callback(payload)
    except Exception as e:
        logger.error("Memory extraction callback failed: %s", e, exc_info=True)

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


@v0_router.post("/voice/command/continue/stream")
async def continue_voice_command_stream(
    request: ToolResultRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    model_service: ModelService = Depends(get_model_service),
):
    """Streaming twin of /voice/command/continue.

    Runs the post-tool-results LLM call as a stream and pipes sentences to
    TTS, returning audio/raw PCM. The user hears the first sentence within
    ~700ms instead of waiting for the full ~3s completion. Falls back to
    202 JSON when streaming isn't applicable so callers can retry against
    the blocking endpoint.
    """
    start_time = time.time()

    logger.info(
        f"Continue (stream) from node: {node_context_provider.node.node_id} "
        f"conversation={request.conversation_id}, tool_results={len(request.tool_results)}"
    )

    try:
        from app.core.clients.tts_client import TTSClient

        tts_client = TTSClient(
            household_id=node_context_provider.household_id,
            node_id=node_context_provider.node.node_id,
        )

        tool_results = [
            {"tool_call_id": tr.tool_call_id, "output": tr.output}
            for tr in request.tool_results
        ]

        # Push actions to inbox in parallel with the stream — same as the
        # blocking endpoint. Fire-and-forget so it doesn't block audio.
        await _maybe_push_actions_to_inbox(
            tool_results=tool_results,
            node_context_provider=node_context_provider,
        )

        streaming_audio = await model_service.try_stream_continue_with_tool_results(
            conversation_id=request.conversation_id,
            tool_results=tool_results,
            tts_client=tts_client,
        )

        if streaming_audio is None:
            # Streaming path not applicable — return 202 JSON so the node
            # falls back to calling /voice/command/continue (blocking).
            logger.info("Continue-stream: not applicable, signaling fallback")
            return JSONResponse(
                status_code=202,
                content={"fallback": "use_blocking_continue"},
            )

        try:
            audio_fmt = await tts_client.get_audio_format()
        except Exception:
            audio_fmt = {"sample_rate": "16000", "channels": "1", "sample_width": "2"}

        duration = time.time() - start_time
        logger.info(f"🚀 Continue-stream LLM→TTS dispatched in {duration:.2f}s")

        return StreamingResponse(
            streaming_audio,
            status_code=200,
            media_type="audio/raw",
            headers={
                "X-Audio-Sample-Rate": audio_fmt["sample_rate"],
                "X-Audio-Channels": audio_fmt["channels"],
                "X-Audio-Sample-Width": audio_fmt["sample_width"],
                "X-Assistant-Message": "",
            },
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Continue-stream failed after {duration:.2f}s: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to stream continuation: {str(e)}",
        )


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

    except HTTPException:
        # Whole-request rejection — surface the status, don't swallow to 200.
        raise
    except ConversationPreconditionError as e:
        # Unknown / expired conversation_id: reject the whole request as 422.
        raise HTTPException(status_code=422, detail=str(e))
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
