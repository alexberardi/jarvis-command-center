"""
Resolve Relative Date Tool.

Converts relative date/time expressions (like "tomorrow", "next week") to actual dates.
"""

import logging
from typing import Dict, Any, Optional

from app.core.interfaces.iserver_tool import IServerTool
from app.core.general_context import generate_date_context_object

logger = logging.getLogger("uvicorn")


class ResolveRelativeDateTool(IServerTool):
    """Tool for resolving relative date terms to actual dates."""
    
    @property
    def name(self) -> str:
        return "resolve_relative_date"
    
    @property
    def description(self) -> str:
        return (
            "Resolve ambiguous or non-absolute date/time phrases to absolute dates in the user's timezone. "
            "Call before any other tool when relative time is mentioned; do not guess. "
            "Only output resolved_datetimes if you called this tool."
        )

    @property
    def included_system_prompt_text(self) -> str:
        return ""
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "relative_term": {
                    "type": "string",
                    "description": (
                        "Relative date phrase from the user's utterance (e.g., 'today', 'tomorrow', 'next week', 'last weekend', 'yesterday', 'next Monday'). "
                        "Spaces or underscores are both allowed."
                    )
                },
                "timezone": {
                    "type": "string",
                    "description": "Timezone to use for resolution (optional - will use conversation timezone if available). Examples: 'America/New_York', 'UTC', 'Europe/London'"
                }
            },
            "required": ["relative_term"]
        }
    
    def execute(self, relative_term: str, timezone: Optional[str] = None, conversation_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Execute the date resolution.
        
        Args:
            relative_term: Relative date term (e.g., "tomorrow", "next week")
            timezone: Optional timezone string (if not provided, will try to get from conversation cache)
            conversation_id: Conversation ID (passed by tool executor, used to get cached timezone)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dict with resolved date(s)
        """
        # Always prefer timezone from conversation cache (it's the source of truth)
        # Only use provided timezone if cache doesn't have one
        if conversation_id:
            from app.core.conversation_cache import conversation_cache
            cached_timezone = conversation_cache.get_timezone(conversation_id)
            if cached_timezone:
                timezone = cached_timezone
                logger.info(f"      📅 Using timezone from cache: {timezone} (ignoring provided timezone if any)")
            elif timezone:
                logger.info(f"      📅 No timezone in cache, using provided: {timezone}")
            else:
                logger.warning(f"      📅 No timezone in cache or provided for conversation {conversation_id[:8]}, using server time")
        elif timezone:
            logger.info(f"      📅 Using provided timezone: {timezone} (no conversation_id to check cache)")
        else:
            logger.warning(f"      📅 No timezone provided and no conversation_id, using server time")
        
        logger.info(f"      📅 Resolving: '{relative_term}' (tz: {timezone or 'server'})")
        
        try:
            # Generate full date context
            date_context = generate_date_context_object(timezone)
            
            # Log the current date from the generated context for debugging
            current_date = date_context.get("current", {}).get("date_iso", "unknown")
            logger.info(f"      📅 Generated context shows current date: {current_date}")
            
            # Normalize the term: lowercase and replace spaces with underscores
            normalized_term = relative_term.lower().strip().replace(" ", "_")
            
            # Try to find the term in different sections of the date context
            result = None
            
            # Check current dates
            if normalized_term in ["today", "now"]:
                result = {
                    "term": normalized_term,
                    "date": date_context["current"]["date_iso"],
                    "datetime": date_context["current"]["datetime"],
                    "utc_start_of_day": date_context["current"]["utc_start_of_day"]
                }
            
            # Check relative_dates section
            elif normalized_term in date_context.get("relative_dates", {}):
                result = {
                    "term": normalized_term,
                    **date_context["relative_dates"][normalized_term]
                }
            
            # Check weekend section
            elif normalized_term in ["this_weekend", "next_weekend", "last_weekend"]:
                weekend_data = date_context["weekend"].get(normalized_term, [])
                result = {
                    "term": normalized_term,
                    "dates": [
                        {
                            "day": day["day"],
                            "date": day["date"],
                            "utc_start_of_day": day["utc_start_of_day"]
                        }
                        for day in weekend_data
                    ]
                }
            
            # Check weekdays section (next_monday, last_friday, etc.)
            elif normalized_term in date_context.get("weekdays", {}):
                result = {
                    "term": normalized_term,
                    **date_context["weekdays"][normalized_term]
                }
            
            # Check time_expressions section (tonight, tomorrow_morning, etc.)
            elif normalized_term in date_context.get("time_expressions", {}):
                result = {
                    "term": normalized_term,
                    "datetime": date_context["time_expressions"][normalized_term]
                }
            
            # Check weeks section
            elif normalized_term in ["this_week", "next_week", "last_week"]:
                week_data = date_context["weeks"].get(normalized_term, [])
                result = {
                    "term": normalized_term,
                    "dates": [
                        {
                            "day": day["day"],
                            "date": day["date"],
                            "utc_start_of_day": day["utc_start_of_day"]
                        }
                        for day in week_data
                    ]
                }

            # Check years section
            elif normalized_term in ["this_year", "next_year", "last_year"]:
                year_data = date_context["years"].get(normalized_term, [])
                result = {
                    "term": normalized_term,
                    "dates": [
                        {
                            "date": day["date"],
                            "utc_start_of_day": day["utc_start_of_day"]
                        }
                        for day in year_data
                    ]
                }
            
            if result:
                # Log concise version of result
                if "dates" in result:
                    logger.info(f"      └─ Resolved to {len(result['dates'])} dates")
                elif "date" in result:
                    logger.info(f"      └─ Resolved to: {result['date']}")
                elif "datetime" in result:
                    logger.info(f"      └─ Resolved to: {result['datetime']}")
                
                # Log full result before returning
                import json
                result_json = json.dumps(result, indent=2)
                logger.debug(f"      📤 Returning to LLM (first 1000 chars):\n{result_json[:1000]}")
                if len(result_json) > 1000:
                    logger.debug(f"      ... (truncated, total length: {len(result_json)} chars)")
                
                return result
            else:
                # Term not found: return a fallback candidate map so the LLM can decide
                logger.warning(f"      └─ ⚠️  Could not resolve term: '{relative_term}', providing candidates")
                result = {
                    "term": normalized_term,
                    "candidates": {
                        "relative_dates": date_context.get("relative_dates", {}),
                        "weekend": date_context.get("weekend", {}),
                        "weekdays": date_context.get("weekdays", {}),
                        "weeks": date_context.get("weeks", {}),
                        "time_expressions": date_context.get("time_expressions", {})
                    },
                    "message": f"Could not resolve '{relative_term}' exactly; choose the closest option."
                }
                
                # Log full result before returning
                import json
                result_json = json.dumps(result, indent=2)
                logger.debug(f"      📤 Returning to LLM (first 1000 chars):\n{result_json[:1000]}")
                if len(result_json) > 1000:
                    logger.debug(f"      ... (truncated, total length: {len(result_json)} chars)")
                
                return result
                
        except Exception as e:
            logger.error(f"      └─ ❌ Error: {e}")
            return {
                "error": str(e),
                "message": "Failed to resolve relative date"
            }

