import json
import time
import os
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter, BackgroundTasks
from app.context_providers.node_context_provider import NodeContextProvider
from app.request_models.voice_command_request import VoiceCommandRequest
from app.request_models.conversation_start_request import ConversationStartRequest
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse
from app.debug_setup import setup_debugger
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from app.core.conversation_cache import conversation_cache
from app.core.llm_manager import LLMManager
from app.core.command_validation_service import CommandValidationService
from app.core.parameter_extraction_service import ParameterExtractionService
from . import admin, chat, date_context
from deps import verify_api_key

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn")

# Set up debugger if DEBUG environment variable is set
setup_debugger()

# Initialize malformed JSON extractor service
malformed_json_extractor = MalformedJsonExtractorService()



app = FastAPI(title="Jarvis Command Center", version="1.0.0")

# Create versioned routers
v0_router = APIRouter()

# Include admin router with versioning
app.include_router(admin.router, prefix="/api/v0/admin", tags=["admin"])

# Include chat router
app.include_router(chat.chat_router, prefix="/api/v0", tags=["chat"])

# Include date context router
app.include_router(date_context.date_context_router, prefix="/api/v0", tags=["date-context"])





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


@v0_router.get("/ping")
def ping():
    return {"message": "pong"}


@v0_router.get("/health")
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




@v0_router.post("/conversation/start")
async def start_conversation(
    request: ConversationStartRequest,
    background_tasks: BackgroundTasks,
    node_context_provider: NodeContextProvider = Depends(verify_api_key)
):
    """Start a new conversation and warm up the LLM with general context."""
    try:
        # Build complete command inference prompt containing all rules, commands, and examples
        from app.context_providers.general_prompt_provider import GeneralPromptProvider
        general_provider = GeneralPromptProvider()
        
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
        

        
        # Build complete static prompt with all rules, commands, and examples
        complete_static_prompt = general_provider.build_complete_static_prompt(
            node_context, 
            client_timezone, 
            request.available_commands,
            request.conversation_id
        )
        
        # Build warm-up messages - just establish base context
        warmup_messages = [
            {"role": "system", "content": complete_static_prompt },
            {"role": "user", "content": "This is a warm-up message to establish context."},
            {"role": "assistant", "content": "I understand. I'm ready to help with command inference and parameter extraction."}
        ]
        
        # Initialize LLM manager for background warmup
        llm_manager = LLMManager()
        
        # Add background task for LLM warm-up and cache population
        # The cache will be populated after the LLM warmup completes
        background_tasks.add_task(
            llm_manager.warmup_conversation_background,
            request.conversation_id,
            request.conversation_id,
            warmup_messages,
            request.available_commands,
            client_timezone
        )

        
        # Return success immediately - LLM warm-up and cache population will happen in background
        return {"status": "success", "conversation_id": request.conversation_id}
            
    except Exception as e:
        logger.error(f"❌ Error starting conversation {request.conversation_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")




    
    # async def run_date_extraction():
    #     date_start_time = time.time()
    #     try:
    #         logger.info(f"🚀 Starting date extraction request...")
    #         logger.info(f"📡 Date extraction URL: {chat_url}")
    #         logger.info(f"📡 Date extraction payload size: {len(str(date_llm_payload))} chars")
            
    #         # Time the actual HTTP request
    #         http_start = time.time()
    #         response = await post(url=chat_url, json_data=date_llm_payload)
    #         http_duration = time.time() - http_start
            
    #         date_duration = time.time() - date_start_time
            
    #         date_choices = response.get('choices', [{}])
    #         date_message = date_choices[0].get('message', {}) if date_choices else {}
    #         date_response_content = date_message.get('content', '')
            
    #         date_extraction_result = {}
    #         if date_response_content:
    #             date_json_valid, _ = await validate_and_clean_json_response(date_response_content, cleaned_voice_command)
    #             if date_json_valid:
    #                 json_start = date_response_content.find('{')
    #                 json_end = date_response_content.rfind('}')
    #                 if json_start >= 0 and json_end > json_start:
    #                     json_portion = date_response_content[json_start:json_end + 1]
    #                     try:
    #                         date_extraction_result = json.loads(json_portion)
    #                     except json.JSONDecodeError:
    #                         logger.warning(f"Failed to parse date extraction JSON, using empty result")
            
    #         logger.info(f"✅ Date extraction completed in {date_duration:.3f}s")
    #         logger.info(f"📡 HTTP request took: {http_duration:.3f}s")
    #         logger.info(f"📊 Date extraction result: {date_extraction_result}")
    #         return date_extraction_result
    #     except Exception as e:
    #         date_duration = time.time() - date_start_time
    #         logger.warning(f"❌ Date extraction failed after {date_duration:.3f}s: {e}, using empty result")
    #         return {}
    



@v0_router.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
):
    start_time = time.time()
    
    # Initialize services
    llm_manager = LLMManager()

    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")
    logger.info(f"Command: '{request.voice_command}'")

    try:
        # Get available commands from cache
        available_commands = None
        if request.conversation_id:
            available_commands = conversation_cache.get_available_commands(request.conversation_id)
        
        # Step 3: Command inference (first step)
        try:
            # Make LLM request for command inference using cache-managed method
            command_response_data, command_response_content = await llm_manager.command_inference_with_cache(
                message=request.voice_command,
                conversation_id=request.conversation_id,
                available_commands=available_commands
            )
            
            # Process the response
            commands = llm_manager.process_command_inference_response(command_response_data, command_response_content)
            
            # Note: llm_manager.command_inference() already handles adding to conversation cache
            
            # Step 5: Parameter inference (second step)
            final_commands = []
            
            # Initialize services
            command_validation_service = CommandValidationService()
            parameter_extraction_service = ParameterExtractionService()
            
            if not available_commands:
                logger.error(f"No available commands found in cache for conversation {request.conversation_id}")
                # DO NOT DELETE - Broken command handling for future client-side support
                for command in commands:
                    command_name = command.get("n", "unknown")
                    broken_command = {
                        "success": False,
                        "command_name": command_name,
                        "parameters": None,
                        "errors": {"type": "no_available_commands", "message": "No available commands found in cache"}
                    }
                    final_commands.append(broken_command)
                return VoiceCommandResponse(commands=final_commands)
            
            # Validate commands and separate valid from broken ones
            valid_commands, broken_commands = command_validation_service.validate_commands(
                commands, available_commands, request.conversation_id
            )
            
            # Add broken commands to final response (DO NOT DELETE - for future client-side support)
            for broken_command in broken_commands:
                final_commands.append(broken_command)
            
            # Process valid commands for parameter extraction
            for selected_command in valid_commands:
                # Build parameter extraction context
                node_context, cached_timezone = parameter_extraction_service.build_parameter_context(
                    selected_command=selected_command,
                    request_conversation_id=request.conversation_id,
                    conversation_cache=conversation_cache,
                    node_context_provider=node_context_provider,
                    voice_command=request.voice_command
                )
                
                # Extract parameters
                extracted_parameters, date_extraction_result = await parameter_extraction_service.extract_parameters_and_dates(
                    selected_command=selected_command,
                    voice_command=request.voice_command,
                    cached_timezone=cached_timezone,
                    conversation_id=request.conversation_id,
                    conversation_cache=conversation_cache
                )
                
                # Convert compressed format to full format
                command_name = selected_command.command_name
                final_command = {
                    "success": True,  # We know it's valid since it passed validation
                    "command_name": command_name,
                    "parameters": extracted_parameters,
                    "errors": None
                }
                final_commands.append(final_command)

            # Create the final response
            final_response_data = {
                "commands": final_commands
            }
            
            # Add request information if we have it
            if request.voice_command or request.conversation_id:
                final_response_data["request_information"] = {
                    "voice_command": request.voice_command,
                    "conversation_id": request.conversation_id
                }

            return VoiceCommandResponse(**final_response_data)
            
        except Exception as e:
            logger.error(f"❌ Command inference failed: {e}")
            raise HTTPException(status_code=500, detail=f"Command inference failed: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Voice command processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice command processing failed: {str(e)}")

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse command inference response: {e}")

            response_data = {
                "commands": [SingleCommandResponse(
                    success=False,
                    command_name=None,
                    parameters=None,
                    errors=VoiceCommandError(
                        type="parsing_error",
                        message=f"Failed to parse command inference response: {str(e)}",
                        clarification_question="Could you please rephrase your command?"
                    )
                )]
            }
            
            # Add request information if we have it
            if request.voice_command or request.conversation_id:
                response_data["request_information"] = {
                    "voice_command": request.voice_command,
                    "conversation_id": request.conversation_id
                }
            
            return VoiceCommandResponse(**response_data)

    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time

        logger.error(f"❌ Unexpected error processing voice command: {e}")
        logger.error(f"   Response time: {response_time:.2f}s")
        logger.error(f"   Node: {node_context_provider.node.node_id}")
        logger.error(f"   Command: '{request.voice_command}'")

        response_data = {
            "commands": [SingleCommandResponse(
                success=False,
                command_name=None,
                parameters=None,
                errors=VoiceCommandError(
                    type="system_error",
                    message=f"System error: {str(e)}",
                    clarification_question="Please try again in a moment."
                )
            )]
        }
        
        # Add request information if we have it
        if request.voice_command or request.conversation_id:
            response_data["request_information"] = {
                "voice_command": request.voice_command,
                "conversation_id": request.conversation_id
            }
        
        return VoiceCommandResponse(**response_data)







# Include versioned routes at the end after all routes are defined
app.include_router(v0_router, prefix="/api/v0", tags=["v0"])