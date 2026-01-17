"""
Date context generation endpoint.
"""

import logging
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.context_providers.node_context_provider import NodeContextProvider
from app.core.general_context import generate_date_context_object
from app.deps import verify_api_key

logger = logging.getLogger("uvicorn")

# Create router for date context endpoints
date_context_router = APIRouter()

@date_context_router.get("/generate/date-context")
async def generate_date_context(
    timezone: str = None,
    node_context_provider: NodeContextProvider = Depends(verify_api_key)
):
    """
    Generates a comprehensive date context object for the LLM.
    This is useful for date-related commands.
    
    Args:
        timezone: Optional timezone string (e.g., "America/New_York")
    """
    try:
        # Use provided timezone or try to get from node context
        if not timezone:
            # Check if there's a timezone in the node context
            timezone = getattr(node_context_provider.node, 'timezone', None)
        
        date_context = generate_date_context_object(timezone)
        logger.debug(f"Generated date context for timezone {timezone}: {date_context}")
        return JSONResponse(status_code=200, content=date_context)
    except Exception as e:
        logger.error(f"Error generating date context: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate date context: {str(e)}"})
