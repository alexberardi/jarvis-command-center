"""
LLM Proxy Client for handling all LLM API interactions.
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional
from app.core.utils.rest_client import post, get, build_jarvis_app_headers

logger = logging.getLogger("uvicorn")

class LLMProxyClient:
    """Client for interacting with the LLM proxy API."""
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the LLM proxy client.
        
        Args:
            base_url: Base URL for the LLM proxy. If None, uses environment variable.
        """
        self.base_url = base_url or os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:8000")
        self.app_headers = build_jarvis_app_headers()
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return f"{self.base_url.rstrip('/')}{endpoint}"

    def _build_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Combine required app headers with any caller-provided headers.
        """
        return {**self.app_headers, **(extra_headers or {})}
    
    async def chat_completion(
        self, 
        messages: list, 
        model: str = "full", 
        temperature: float = 0, 
        conversation_id: Optional[str] = None,
        tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """Make a chat completion request."""
        import time
        start_time = time.time()
        
        url = self._build_url("/v1/chat/completions")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        
        logger.debug(f"Making chat completion request to {url}")
        logger.info(f"🐛 DEBUG: LLM proxy client starting request at {start_time}")
        
        result = await post(url=url, json_data=payload, headers=self._build_headers())
        
        end_time = time.time()
        logger.info(f"🐛 DEBUG: LLM proxy client completed request at {end_time}, took {end_time - start_time:.3f}s")
        logger.info(f"🔍 LLM proxy returned keys: {list(result.keys()) if isinstance(result, dict) else 'NOT A DICT'}")
        logger.info(f"🔍 LLM proxy 'message' value type: {type(result.get('message', 'KEY NOT FOUND'))}")
        
        return result
    
    async def lightweight_chat(
        self, 
        messages: list, 
        model: str = "lightweight", 
        temperature: float = 0
    ) -> Dict[str, Any]:
        """Make a lightweight chat request."""
        url = self._build_url("/v1/chat/completions")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        
        logger.debug(f"Making lightweight chat request to {url}")
        return await post(url=url, json_data=payload)
    
    async def warmup_conversation(
        self, 
        conversation_id: str, 
        messages: list, 
        model: str = "full", 
        temperature: float = 0,
        tools: Optional[list] = None
    ) -> Dict[str, Any]:
        """Warm up a conversation with the LLM."""
        import time
        start_time = time.time()
        
        url = self._build_url("/v1/chat/completions")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "stream": False,
        }
        logger.info(f"🔥 POSTing warmup to {url} for conversation {conversation_id[:8]}")
        logger.debug(f"   Payload keys: {list(payload.keys())}, messages count: {len(messages)}")
        
        # Calculate approximate payload size for logging
        import json
        payload_size = len(json.dumps(payload))
        logger.info(f"   Payload size: ~{payload_size:,} bytes ({payload_size/1024:.1f} KB)")
        
        try:
            # Warmup requests can take longer due to larger system prompts, use longer timeout
            result = await post(url=url, json_data=payload, headers=self._build_headers(), timeout=120)
            duration = time.time() - start_time
            logger.info(f"✅ Warmup POST succeeded in {duration:.3f}s for {conversation_id[:8]}")
            logger.info(f"   Response type: {type(result)}")
            if isinstance(result, dict):
                logger.info(f"   Response keys: {list(result.keys())}")
                # Log the full response (truncated if too long)
                import json
                response_str = json.dumps(result, indent=2)
                if len(response_str) > 2000:
                    logger.info(f"   Response (first 2000 chars): {response_str[:2000]}...")
                else:
                    logger.info(f"   Response: {response_str}")
            else:
                logger.info(f"   Response (not a dict): {result}")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"❌ Warmup POST failed after {duration:.3f}s for {conversation_id[:8]}: {e}")
            logger.error(f"   Error type: {type(e).__name__}")
            raise
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """
        Conversation status endpoint is no longer available on the LLM proxy.
        We return a completed status after verifying the proxy health endpoint.
        """
        try:
            await get(url=self._build_url("/health"))
            return {"status": "completed"}
        except Exception as exc:
            logger.warning(f"LLM proxy health check failed while checking status: {exc}")
            return {"status": "unavailable", "error": str(exc)}
    
    async def passthrough_chat(
        self, 
        request_data: Dict[str, Any], 
        endpoint: str = "/v1/chat/completions"
    ) -> Dict[str, Any]:
        """Passthrough request to the LLM proxy."""
        url = self._build_url(endpoint)
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(100.0)) as client:
            try:
                resp = await client.post(url, json=request_data, headers=self._build_headers())
            except httpx.RequestError as e:
                logger.error(f"[passthrough {endpoint}] upstream request error: {e}")
                raise Exception(f"Failed to reach LLM proxy: {str(e)}")

            try:
                content = resp.json()
            except Exception:
                content = {"error": resp.text}

            if resp.status_code >= 400:
                logger.error(
                    f"[passthrough {endpoint}] upstream status={resp.status_code} body={resp.text[:600]}"
                )
            else:
                logger.info(f"[passthrough {endpoint}] upstream status={resp.status_code}")

            return {"status_code": resp.status_code, "content": content}
