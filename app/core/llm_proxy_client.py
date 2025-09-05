"""
LLM Proxy Client for handling all LLM API interactions.
"""

import os
import logging
import httpx
from typing import Dict, Any, Optional
from app.core.utils.rest_client import post, get

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
        self.api_version = os.getenv("JARVIS_LLM_PROXY_API_VERSION", "0")
    
    def _build_url(self, endpoint: str) -> str:
        """Build full URL from endpoint."""
        return f"{self.base_url}/api/v{self.api_version}{endpoint}"
    
    async def chat_completion(
        self, 
        messages: list, 
        model: str = "jarvis-llm", 
        temperature: float = 0, 
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Make a chat completion request."""
        url = self._build_url("/chat")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        if conversation_id:
            payload["conversation_id"] = conversation_id
        
        logger.debug(f"Making chat completion request to {url}")
        return await post(url=url, json_data=payload)
    
    async def lightweight_chat(
        self, 
        messages: list, 
        model: str = "jarvis-llm", 
        temperature: float = 0
    ) -> Dict[str, Any]:
        """Make a lightweight chat request."""
        url = self._build_url("/lightweight/chat")
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
        model: str = "jarvis-llm", 
        temperature: float = 0
    ) -> Dict[str, Any]:
        """Warm up a conversation with the LLM."""
        url = self._build_url(f"/chat/conversation/{conversation_id}/warmup")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        print(url)
        
        logger.debug(f"Making warmup request to {url}")
        return await post(url=url, json_data=payload)
    
    async def get_conversation_status(self, conversation_id: str) -> Dict[str, Any]:
        """Get the status of a conversation."""
        url = self._build_url(f"/conversation/{conversation_id}/status")
        
        logger.debug(f"Getting conversation status from {url}")
        return await get(url=url)
    
    async def passthrough_chat(
        self, 
        request_data: Dict[str, Any], 
        endpoint: str = "/chat"
    ) -> Dict[str, Any]:
        """Passthrough request to the LLM proxy."""
        url = self._build_url(endpoint)
        
        async with httpx.AsyncClient(timeout=httpx.Timeout(100.0)) as client:
            try:
                resp = await client.post(url, json=request_data)
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
