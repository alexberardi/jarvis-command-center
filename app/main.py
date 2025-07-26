import json
import re
import time
import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.context_providers.standard_system_prompt_provider import StandardCommandInferenceSystemPromptProvider
from app.context_providers.node_context_provider import NodeContextProvider
from app.core.interfaces.ijarvis_context_provider import IJarvisContextProvider, ICommandInferenceSystemPromptProvider, ITranscriptionCleanupSystemPromptProvider
from app.request_models.voice_command_request import VoiceCommandRequest, CommandDefinition
from app.response_models.voice_command_response import VoiceCommandResponse, VoiceCommandError, SingleCommandResponse
from app.debug_setup import setup_debugger
from app.core.utils.rest_client import post
from app.core.malformed_json_extractor import MalformedJsonExtractorService
from .deps import verify_api_key, get_command_inference_system_prompt_provider, get_transcription_cleanup_system_prompt_provider
from . import admin

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("uvicorn")

# Set up debugger if DEBUG environment variable is set
setup_debugger()

# Initialize malformed JSON extractor service
malformed_json_extractor = MalformedJsonExtractorService()

# JSON validation statistics
json_validation_stats = {
    "total_requests": 0,
    "invalid_json_count": 0,
    "legacy_format_conversions": 0,
    "auto_fixes": 0,
    "fallback_extractions": 0,
    "completely_failed": 0
}

app = FastAPI(title="Jarvis Command Center", version="1.0.0")

# Create versioned routers
v0_router = APIRouter()

# Include admin router with versioning
app.include_router(admin.router, prefix="/api/v0/admin", tags=["admin"])


async def validate_and_clean_json_response(raw_response, original_voice_command=None):
    """
    Enhanced JSON validation and cleaning for production use
    Returns: (cleaned_json, issues_list, success_boolean)
    """
    global json_validation_stats
    issues = []
    
    # Remove any leading/trailing whitespace
    cleaned = raw_response.strip()
    
    # Try to extract JSON from the response
    json_start = cleaned.find('{')
    json_end = cleaned.rfind('}')
    
    if json_start == -1 or json_end == -1:
        issues.append("No JSON object found in response")
        json_validation_stats["invalid_json_count"] += 1
        json_validation_stats["completely_failed"] += 1
        return None, issues, False
    
    # Extract just the JSON portion
    json_portion = cleaned[json_start:json_end + 1]
    
    # Try to parse the JSON
    try:
        parsed_json = json.loads(json_portion)
        
        # Validate structure
        if not isinstance(parsed_json, dict):
            issues.append("Response is not a JSON object")
            return None, issues, False
        
        # Handle both compressed and full key formats
        commands_key = None
        if "commands" in parsed_json:
            commands_key = "commands"
        elif "c" in parsed_json:
            commands_key = "c"
            # Convert compressed format to full format for consistency
            converted_json = {"commands": parsed_json["c"]}
            for cmd in converted_json["commands"]:
                if isinstance(cmd, dict):
                    if "s" in cmd:
                        cmd["success"] = cmd.pop("s")
                    if "n" in cmd:
                        cmd["command_name"] = cmd.pop("n")
                    if "p" in cmd:
                        cmd["parameters"] = cmd.pop("p")
                    if "e" in cmd:
                        cmd["errors"] = cmd.pop("e")
            parsed_json = converted_json
            # Don't treat compressed format conversion as an error - it's working correctly
        else:
            # Check if this is a legacy single-command format
            if any(key in parsed_json for key in ["s", "success", "n", "command_name", "p", "parameters", "e", "errors"]):
                # Convert legacy single-command format to array format
                legacy_command = {}
                if "s" in parsed_json:
                    legacy_command["success"] = parsed_json["s"]
                elif "success" in parsed_json:
                    legacy_command["success"] = parsed_json["success"]
                else:
                    legacy_command["success"] = False
                
                if "n" in parsed_json:
                    legacy_command["command_name"] = parsed_json["n"]
                elif "command_name" in parsed_json:
                    legacy_command["command_name"] = parsed_json["command_name"]
                else:
                    legacy_command["command_name"] = None
                
                if "p" in parsed_json:
                    legacy_command["parameters"] = parsed_json["p"] if parsed_json["p"] is not None else {}
                elif "parameters" in parsed_json:
                    legacy_command["parameters"] = parsed_json["parameters"] if parsed_json["parameters"] is not None else {}
                else:
                    legacy_command["parameters"] = {}
                
                if "e" in parsed_json:
                    legacy_command["errors"] = parsed_json["e"]
                elif "errors" in parsed_json:
                    legacy_command["errors"] = parsed_json["errors"]
                else:
                    legacy_command["errors"] = None
                
                parsed_json = {"commands": [legacy_command]}
                issues.append("Converted legacy single-command format to array format")
                json_validation_stats["legacy_format_conversions"] += 1
            else:
                issues.append("Missing 'commands' or 'c' key in response")
                json_validation_stats["invalid_json_count"] += 1
                json_validation_stats["completely_failed"] += 1
                return None, issues, False
        
        if not isinstance(parsed_json["commands"], list):
            issues.append("'commands' is not a list")
            return None, issues, False
        
        # Validate each command
        for i, cmd in enumerate(parsed_json["commands"]):
            if not isinstance(cmd, dict):
                issues.append(f"Command {i+1} is not an object")
                continue
            
            required_keys = ["success", "command_name", "parameters", "errors"]
            for key in required_keys:
                if key not in cmd:
                    issues.append(f"Command {i+1} missing required key: {key}")
            
            # Validate success field
            if "success" in cmd and not isinstance(cmd["success"], bool):
                issues.append(f"Command {i+1} 'success' field is not boolean")
            
            # Validate parameters - must be object, never null
            if "parameters" in cmd:
                if cmd["parameters"] is None:
                    # Fix null parameters to empty object
                    cmd["parameters"] = {}
                    issues.append(f"Command {i+1} 'parameters' was null, fixed to empty object")
                elif not isinstance(cmd["parameters"], dict):
                    issues.append(f"Command {i+1} 'parameters' field is not an object")
            
            # Validate command_name - can be null for failed commands
            if "command_name" in cmd and cmd["command_name"] is not None and not isinstance(cmd["command_name"], str):
                issues.append(f"Command {i+1} 'command_name' field is not a string or null")
        
        # Check for extra content after JSON
        if json_end + 1 < len(cleaned):
            extra_content = cleaned[json_end + 1:].strip()
            if extra_content:
                issues.append(f"Extra content after JSON: '{extra_content[:50]}{'...' if len(extra_content) > 50 else ''}'")
        
        return parsed_json, issues, len(issues) == 0
        
    except json.JSONDecodeError as e:
        issues.append(f"JSON parsing error: {str(e)}")
        
        # Try to fix common JSON issues
        try:
            # Remove trailing commas
            fixed_json = json_portion.replace(',}', '}').replace(',]', ']')
            # Fix missing quotes around keys
            fixed_json = re.sub(r'(\w+):', r'"\1":', fixed_json)
            # Fix null parameters to empty objects
            fixed_json = re.sub(r'"parameters":\s*null', '"parameters": {}', fixed_json)
            fixed_json = re.sub(r'"p":\s*null', '"p": {}', fixed_json)
            
            parsed_json = json.loads(fixed_json)
            issues.append("JSON was automatically fixed")
            return parsed_json, issues, True
            
        except:
            issues.append("Could not automatically fix JSON")
            
            # Try fallback LLM extraction
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If we're in an async context, we need to use create_task
                    logger.info("Attempting fallback LLM extraction in async context")
                    # Create a task and wait for it with a timeout
                    task = asyncio.create_task(
                        malformed_json_extractor.extract_json_from_malformed_response(
                            raw_response, 
                            original_voice_command
                        )
                    )
                    try:
                        extracted_json, success, error_msg = await asyncio.wait_for(task, timeout=10.0)
                        if success and extracted_json:
                            issues.append("JSON extracted successfully with malformed JSON extractor")
                            return extracted_json, issues, True
                        else:
                            issues.append(f"Malformed JSON extraction failed: {error_msg}")
                    except asyncio.TimeoutError:
                        logger.warning("Fallback LLM extraction timed out")
                        issues.append("Fallback LLM extraction timed out")
                else:
                    extracted_json, success, error_msg = loop.run_until_complete(
                        malformed_json_extractor.extract_json_from_malformed_response(
                            raw_response, 
                            original_voice_command
                        )
                    )
                    if success and extracted_json:
                        issues.append("JSON extracted successfully with malformed JSON extractor")
                        return extracted_json, issues, True
                    else:
                        issues.append(f"Malformed JSON extraction failed: {error_msg}")
            except Exception as extraction_error:
                logger.error(f"Malformed JSON extraction error: {extraction_error}")
                issues.append(f"Malformed JSON extraction error: {extraction_error}")
            
            return None, issues, False


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


@v0_router.post("/voice/command", response_model=VoiceCommandResponse)
async def handle_voice(
    request: VoiceCommandRequest,
    node_context_provider: NodeContextProvider = Depends(verify_api_key),
    system_prompt_provider: ICommandInferenceSystemPromptProvider = Depends(get_command_inference_system_prompt_provider),
    transcription_cleanup_provider: ITranscriptionCleanupSystemPromptProvider = Depends(get_transcription_cleanup_system_prompt_provider)
):
    import time
    start_time = time.time()
    
    logger.info(f"Voice command from node: {node_context_provider.node.node_id} in room {node_context_provider.node.room}")
    logger.info(f"Command: '{request.voice_command}'")

    try:
        # Step 1: Transcription cleanup (if enabled)
        cleaned_voice_command = request.voice_command
        if transcription_cleanup_provider.name != "NO_OP":
            logger.info("🔄 Performing transcription cleanup...")
            cleanup_prompt = transcription_cleanup_provider.build_system_prompt(request.voice_command)
            
            cleanup_messages = [
                {"role": "system", "content": cleanup_prompt},
                {"role": "user", "content": request.voice_command}
            ]
            
            llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
            llm_api_version = os.getenv("JARVIS_LLM_PROXY_API_VERSION", "0")
            lightweight_chat_url = f"{llm_api_url}/api/v{llm_api_version}/lightweight/chat"
            
            try:
                cleanup_payload = {
                    "model": "jarvis-llm",
                    "temperature": 0,
                    "messages": cleanup_messages
                }
                cleanup_response = await post(url=lightweight_chat_url, json_data=cleanup_payload)
                cleanup_content = (
                    cleanup_response.get('choices', [{}])[0].get('message', {}).get('content', '') or  # OpenAI-style format
                    cleanup_response.get('content', '') or  # Alternative format
                    cleanup_response.get('response', '')  # Legacy format
                )
                
                if cleanup_content and cleanup_content.strip():
                    cleaned_voice_command = cleanup_content.strip()
                    logger.info(f"✅ Transcription cleaned: '{request.voice_command}' → '{cleaned_voice_command}'")
                else:
                    logger.warning("⚠️ Transcription cleanup returned empty response, using original")
            except Exception as e:
                logger.warning(f"⚠️ Transcription cleanup failed: {e}, using original command")
        else:
            logger.info("⏭️ Transcription cleanup disabled, using original command")

        # Step 2: Command inference
        # Build system prompt with optional voice command for preprocessing
        system_prompt = system_prompt_provider.build_system_prompt(
            request.node_context, 
            request.available_commands, 
            cleaned_voice_command
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": cleaned_voice_command}
        ]

        llm_api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
        llm_api_version = os.getenv("JARVIS_LLM_PROXY_API_VERSION", "0")
        chat_url = f"{llm_api_url}/api/v{llm_api_version}/chat"

        # Make LLM request
        llm_payload = {
            "model": "jarvis-llm",
            "temperature": 0,
            "messages": messages
        }
        llm_response = await post(url=chat_url, json_data=llm_payload)

        # Enhanced JSON validation and parsing
        try:
            # Extract response content from LLM response (OpenAI-style format)
            response_content = llm_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not response_content:
                logger.error("Empty response from LLM")
                raise ValueError("Empty response from LLM")
            
            # Use enhanced validation
            response_data, validation_issues, json_valid = await validate_and_clean_json_response(response_content, cleaned_voice_command)
            
            if not json_valid:
                # Log validation issues for debugging
                for issue in validation_issues:
                    logger.warning(f"JSON validation issue: {issue}")
                
                if response_data:
                    logger.info("✅ Successfully recovered JSON from malformed response")
                else:
                    logger.error("❌ Could not extract valid JSON from response")
                    raise ValueError(f"Invalid JSON response: {validation_issues[0] if validation_issues else 'Unknown error'}")
            
            # Validate that we have the expected structure
            if not isinstance(response_data, dict) or "commands" not in response_data:
                raise ValueError("Response missing 'commands' key")
            
            # Log success metrics
            end_time = time.time()
            response_time = end_time - start_time
            command_count = len(response_data.get("commands", []))
            
            logger.info(f"✅ Successfully processed command in {response_time:.2f}s")
            logger.info(f"   Commands extracted: {command_count}")
            logger.info(f"   JSON validation issues: {len(validation_issues)}")
            
            return VoiceCommandResponse(**response_data)
            
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.error(f"Raw response: {llm_response}")
            
            return VoiceCommandResponse(
                commands=[SingleCommandResponse(
                    success=False,
                    command_name=None,
                    parameters=None,
                    errors=VoiceCommandError(
                        type="parsing_error",
                        message=f"Failed to parse LLM response: {str(e)}",
                        clarification_question="Could you please rephrase your command?"
                    )
                )]
            )
            
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        
        logger.error(f"❌ Unexpected error processing voice command: {e}")
        logger.error(f"   Response time: {response_time:.2f}s")
        logger.error(f"   Node: {node_context_provider.node.node_id}")
        logger.error(f"   Command: '{request.voice_command}' → '{cleaned_voice_command}'")
        
        return VoiceCommandResponse(
            commands=[SingleCommandResponse(
                success=False,
                command_name=None,
                parameters=None,
                errors=VoiceCommandError(
                    type="system_error",
                    message=f"System error: {str(e)}",
                    clarification_question="Please try again in a moment."
                )
            )]
        )


# Include versioned routes at the end after all routes are defined
app.include_router(v0_router, prefix="/api/v0", tags=["v0"])