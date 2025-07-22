"""
Malformed JSON extractor service for fixing invalid LLM responses.
This provides a lightweight alternative for extracting valid JSON from malformed responses.
"""

import json
import logging
import os
from typing import Optional, Dict, Any, Tuple
from app.core.json_schema import create_fallback_prompt, get_json_format_examples
from app.core.utils.rest_client import post

logger = logging.getLogger(__name__)

class MalformedJsonExtractorService:
    """Service for using a lightweight LLM to extract valid JSON from malformed responses"""
    
    def __init__(self):
        # Use JARVIS_LLM_PROXY_API_URL from environment, fallback to localhost
        self.api_url = os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
        llm_api_version = os.getenv("JARVIS_LLM_PROXY_API_VERSION", "0")
        self.extract_endpoint = f"{self.api_url}/api/v{llm_api_version}/lightweight/chat"
        # Use lightweight model for JSON extraction, fallback to mistral-7b-q2
        self.lightweight_model = os.getenv("LIGHTWEIGHT_MODEL", "mistral-7b-q2")
        
    async def extract_json_from_malformed_response(
        self, 
        malformed_response: str,
        original_voice_command: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], bool, str]:
        """
        Attempt to extract valid JSON from a malformed LLM response.
        
        Args:
            malformed_response: The malformed JSON response from main LLM
            original_voice_command: Original voice command for context
            
        Returns:
            Tuple of (extracted_json, success, error_message)
        """
        try:
            # Get a good example to show the extractor LLM
            examples = get_json_format_examples()
            good_example = json.dumps(examples["success_example"], separators=(',', ':'))
            
            # Create the extraction prompt
            extraction_prompt = create_fallback_prompt(malformed_response, good_example)
            
            # Add context if available
            if original_voice_command:
                extraction_prompt += f"\n\nOriginal voice command: '{original_voice_command}'"
            
            # Call the lightweight chat endpoint
            payload = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a JSON extraction specialist. Your job is to fix malformed JSON responses."
                    },
                    {
                        "role": "user", 
                        "content": extraction_prompt
                    }
                ],
                "model": self.lightweight_model,  # Lightweight local model for speed
                "temperature": 0.1,  # Low temperature for consistent formatting
                "max_tokens": 300
            }
            
            logger.info(f"Attempting JSON extraction with malformed JSON extractor")
            
            response_data = await post(self.extract_endpoint, json_data=payload)
            
            # Try different response formats
            extracted_content = (
                response_data.get("choices", [{}])[0].get("message", {}).get("content", "") or  # OpenAI format
                response_data.get("response", "") or  # Direct response format
                response_data.get("content", "")  # Alternative format
            )
            
            if not extracted_content:
                logger.error("JSON extractor returned empty response")
                return None, False, "JSON extractor returned empty response"
            
            # Try to parse the extracted JSON
            try:
                # First try to parse the full content
                extracted_json = json.loads(extracted_content.strip())
                logger.info("Successfully extracted JSON with malformed JSON extractor")
                return extracted_json, True, ""
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the beginning of the response
                try:
                    # Find the first { and last } to extract just the JSON part
                    start = extracted_content.find('{')
                    end = extracted_content.rfind('}') + 1
                    if start != -1 and end > start:
                        json_part = extracted_content[start:end]
                        extracted_json = json.loads(json_part)
                        logger.info("Successfully extracted JSON part from response with explanatory text")
                        return extracted_json, True, ""
                except json.JSONDecodeError as e:
                    logger.error(f"JSON extractor returned invalid JSON: {e}")
                    return None, False, f"JSON extractor returned invalid JSON: {e}"
                
        except Exception as e:
            logger.error(f"Error in malformed JSON extraction: {e}")
            return None, False, f"Malformed JSON extraction error: {e}"
    
    async def is_available(self) -> bool:
        """Check if the JSON extractor service is available"""
        try:
            llm_api_version = os.getenv("JARVIS_LLM_PROXY_API_VERSION", "0")
            await post(f"{self.api_url}/api/v{llm_api_version}/health", timeout=5)
            return True
        except Exception:
            return False 