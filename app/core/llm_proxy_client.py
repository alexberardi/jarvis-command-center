"""
LLM Proxy Client for handling all LLM API interactions.
"""

import json
import os
import logging
from typing import Any, AsyncIterator, Dict, Optional

import httpx

from app.core.utils.rest_client import post, get, build_jarvis_app_headers, _get_shared_client

logger = logging.getLogger("uvicorn")


class LLMProxyClient:
    """Client for interacting with the LLM proxy API."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the LLM proxy client.

        Args:
            base_url: Base URL for the LLM proxy. If None, uses service discovery.
        """
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = self._get_base_url()
        self.app_headers = build_jarvis_app_headers()

    def _get_base_url(self) -> str:
        """Get base URL from service discovery or fallback to env var."""
        try:
            from app.core import service_config
            if service_config.is_initialized():
                return service_config.get_llm_proxy_url()
        except (ImportError, AttributeError):
            pass  # Service config not available, use env var fallback
        # Fallback to env var
        return os.getenv("JARVIS_LLM_PROXY_API_URL", "http://localhost:7704")
    
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
        model: str = "live",
        temperature: float = 0,
        conversation_id: Optional[str] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include_date_context: bool = True,
        adapter_settings: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make a chat completion request.

        Args:
            tools: Optional list of OpenAI-format tool definitions for native tool calling.
            tool_choice: Optional tool choice ("auto", "none", or specific tool).
            adapter_settings: Optional dict with adapter configuration.
                Expected keys: hash (str), scale (float, default 1.0), enabled (bool, default True)
            max_tokens: Optional max tokens for completion. When set, overrides
                the LLM proxy's server-side default.
            extra_body: Optional passthrough fields merged into the request payload
                (e.g. ``{"chat_template_kwargs": {"enable_thinking": False}}`` to bound
                a thinking model's reasoning). Merged last, so it can override defaults.
        """
        import time
        start_time = time.time()

        url = self._build_url("/v1/chat/completions")
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": messages
        }

        if conversation_id:
            payload["conversation_id"] = conversation_id
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        payload["include_date_context"] = include_date_context
        if adapter_settings is not None:
            payload["adapter_settings"] = adapter_settings
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_body:
            payload.update(extra_body)

        logger.debug(f"Making chat completion request to {url}")
        logger.info(f"🐛 DEBUG: LLM proxy client starting request at {start_time}")
        
        result = await post(url=url, json_data=payload, headers=self._build_headers())
        
        end_time = time.time()
        logger.info(f"🐛 DEBUG: LLM proxy client completed request at {end_time}, took {end_time - start_time:.3f}s")
        logger.info(f"🔍 LLM proxy returned keys: {list(result.keys()) if isinstance(result, dict) else 'NOT A DICT'}")
        logger.info(f"🔍 LLM proxy 'message' value type: {type(result.get('message', 'KEY NOT FOUND'))}")
        
        return result
    
    async def chat_completion_stream(
        self,
        messages: list,
        model: str = "live",
        temperature: float = 0,
        include_date_context: bool = True,
        adapter_settings: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat completion tokens via SSE.

        Yields dicts:
        - {"delta": "token"} for each streamed token
        - {"done": True, "content": "...", "usage": {...}, ...} as final event

        Falls back to non-streaming if the endpoint returns an error.
        """
        url = self._build_url("/v1/chat/completions")
        payload: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "include_date_context": include_date_context,
        }
        if adapter_settings is not None:
            payload["adapter_settings"] = adapter_settings
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        logger.debug(f"Starting streaming chat completion to {url}")

        client = _get_shared_client()
        async with client.stream(
            "POST", url, json=payload, headers=self._build_headers(),
            timeout=httpx.Timeout(60.0),
        ) as resp:
            if resp.status_code != 200:
                logger.error(f"Streaming request failed: {resp.status_code}")
                raise httpx.HTTPStatusError(
                    f"Streaming failed: {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # strip "data: " prefix
                try:
                    event = json.loads(data_str)
                    yield event
                    if event.get("done"):
                        return
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse SSE data: {data_str[:100]}")
                    continue

    async def lightweight_chat(
        self, 
        messages: list, 
        model: str = "live",
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
        model: str = "live",
        temperature: float = 0,
        tools: Optional[list] = None,
        adapter_settings: Optional[Dict[str, Any]] = None
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
        if adapter_settings is not None:
            payload["adapter_settings"] = adapter_settings
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
            except (ValueError, TypeError):
                content = {"error": resp.text}  # Response not valid JSON

            if resp.status_code >= 400:
                logger.error(
                    f"[passthrough {endpoint}] upstream status={resp.status_code} body={resp.text[:600]}"
                )
            else:
                logger.info(f"[passthrough {endpoint}] upstream status={resp.status_code}")

            return {"status_code": resp.status_code, "content": content}

    async def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the LLM inference engine.

        Returns engine type and capabilities like prefix caching support.
        Fetches fresh from the LLM proxy on every call.

        Returns:
            Dict with keys: inference_engine, backend_type, allows_caching, description
        """
        url = self._build_url("/v1/engine")
        try:
            result = await get(url=url, headers=self._build_headers())
            logger.info(f"🔧 LLM engine info: {result.get('inference_engine')} (allows_caching={result.get('allows_caching')})")
            return result
        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch engine info: {e}")
            return {
                "inference_engine": "unknown",
                "backend_type": "unknown",
                "allows_caching": True,
                "description": "Failed to fetch engine info, assuming caching is available"
            }

    async def get_date_keys(self) -> list[str]:
        """Fetch available date key vocabulary from llm-proxy."""
        url = self._build_url("/v1/adapters/date-keys")
        try:
            result = await get(url=url, headers=self._build_headers())
            keys: list[str] = result.get("static_keys", [])
            logger.info("Fetched %d date keys from llm-proxy", len(keys))
            return keys
        except Exception as e:
            logger.warning("Failed to fetch date keys: %s", e)
            return []

    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts via the LLM proxy (async).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, sorted by index.

        Raises:
            Exception: If the embedding request fails.
        """
        url = self._build_url("/v1/embeddings")
        payload = {"input": texts}
        result = await post(url=url, json_data=payload, headers=self._build_headers())

        # Parse OpenAI-format response and sort by index
        data = result.get("data", [])
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def create_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for the given texts via the LLM proxy (sync).

        Safe to call from synchronous tool code running inside FastAPI's
        event loop (where ``asyncio.new_event_loop()`` would fail).

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors, sorted by index.
        """
        url = self._build_url("/v1/embeddings")
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"input": texts}, headers=self._build_headers())
            resp.raise_for_status()
            data = resp.json().get("data", [])
            data.sort(key=lambda d: d["index"])
            return [d["embedding"] for d in data]

    async def allows_warmup_caching(self) -> bool:
        """
        Check if the LLM engine benefits from warmup messages (prefix caching).

        Returns True if warmup provides caching benefit, False if warmup can be skipped.
        """
        engine_info = await self.get_engine_info()
        return engine_info.get("allows_caching", True)
