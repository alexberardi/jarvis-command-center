"""
Resolve Relative Date Tool.

Converts relative date/time expressions (like "tomorrow", "next week") to actual dates.
Includes LLM fallback for unrecognized date expressions.
"""

import json
import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING

from app.core.interfaces.iserver_tool import IServerTool
from app.core.general_context import generate_date_context_object
from app.core.utils.think_block_stripper import (
    DEFAULT_THINK_DELIMITERS,
    ThinkBlockStripper,
)

if TYPE_CHECKING:
    from app.core.llm_proxy_client import LLMProxyClient

logger = logging.getLogger("uvicorn")

# Reasoning models (Qwen3, Llama 3.3 Thinking) sometimes leak ``<think>...``
# into the resolver's plain-text response — either as a balanced block before
# the answer or as an unclosed tail when max_tokens cuts them off mid-thought.
# Strip before validating so we don't reject a perfectly good "today" answer
# that just happens to follow some chain-of-thought.
_THINK_STRIPPER = ThinkBlockStripper.from_pair(DEFAULT_THINK_DELIMITERS)


class ResolveRelativeDateTool(IServerTool):
    """Tool for resolving relative date terms to actual dates."""
    
    @property
    def enabled(self) -> bool:
        """Disabled: date resolution is now handled automatically by _inject_date_keys()."""
        return False

    @property
    def name(self) -> str:
        return "resolve_relative_date"

    @property
    def description(self) -> str:
        return "Resolve ambiguous or non-absolute date/time phrases to absolute dates in the user's timezone."

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

    @staticmethod
    async def resolve_with_llm_fallback(
        unrecognized_key: str,
        available_keys: List[str],
        user_utterance: str,
        llm_client: "LLMProxyClient",
        timezone: Optional[str] = None
    ) -> Optional[str]:
        """
        Use LLM to resolve an unrecognized date key by picking the best match.

        This is a last-resort fallback when the date key from the primary model
        doesn't match any known keys in the date context.

        Args:
            unrecognized_key: The date key that wasn't found (e.g., "day_after_tomorrow")
            available_keys: List of valid date keys (e.g., ["today", "tomorrow", "next_week"])
            user_utterance: The user's original spoken message for context
            llm_client: LLM proxy client for making the isolated call
            timezone: Optional timezone string

        Returns:
            The best matching key from available_keys, or None if unable to resolve
        """
        if not available_keys:
            logger.warning("📅 LLM fallback called with empty available_keys")
            return None

        # Build a focused prompt for key selection
        prompt = f"""You are a date key matcher. Given a user's spoken message and an unrecognized date expression, pick the BEST matching key from the available options.

User said: "{user_utterance}"
Unrecognized date expression: "{unrecognized_key}"

Available date keys (pick ONE):
{json.dumps(available_keys, indent=2)}

Rules:
- Return ONLY the exact key string from the list, nothing else
- Pick the closest semantic match
- If no good match exists, return "today" as the default

Your answer (just the key):"""

        messages = [{"role": "user", "content": prompt}]

        try:
            logger.info(f"📅 LLM fallback: resolving '{unrecognized_key}' from user: '{user_utterance[:50]}...'")

            response = await llm_client.chat_completion(
                messages=messages,
                model="live",
                temperature=0,
                include_date_context=False  # Don't need date context for this simple task
            )

            # Extract the response content
            raw_content = response.get("message", "")
            if not raw_content:
                # Try OpenAI format
                choices = response.get("choices", [])
                if choices:
                    raw_content = choices[0].get("message", {}).get("content", "")

            if not raw_content:
                logger.warning("📅 LLM fallback returned empty response")
                return None

            # Reasoning models leak <think>...</think> here — strip before
            # parsing so we don't end up validating a whole chain-of-thought.
            stripped_content = _THINK_STRIPPER.strip_all(raw_content)
            # Strip quotes that LLM may include (e.g., "today" -> today)
            selected_key = stripped_content.strip().strip('"\'').lower().replace(" ", "_")

            # Validate it's in our available keys
            if selected_key in available_keys:
                logger.info(f"📅 LLM fallback resolved '{unrecognized_key}' → '{selected_key}'")
                return selected_key

            # Try fuzzy match - maybe LLM added/removed underscores
            for key in available_keys:
                if key.replace("_", "") == selected_key.replace("_", ""):
                    logger.info(f"📅 LLM fallback fuzzy matched '{unrecognized_key}' → '{key}'")
                    return key

            # Last-resort default: prefer "today" over passing the junk through.
            # The caller treats None as "couldn't resolve" and may forward the
            # unparseable key to the tool — which then crashes on strptime in
            # the command. Defaulting to today keeps the request flowing and
            # matches the fallback rule we already give the LLM in the prompt.
            if "today" in available_keys:
                logger.warning(
                    f"📅 LLM fallback returned invalid key '{selected_key}' "
                    f"(not in available_keys); defaulting to 'today'"
                )
                return "today"

            logger.warning(f"📅 LLM fallback returned invalid key '{selected_key}' (not in available_keys)")
            return None

        except Exception as e:
            logger.error(f"📅 LLM fallback error: {e}")
            return None

    @staticmethod
    def get_available_keys(date_context: Dict[str, Any]) -> List[str]:
        """
        Extract all available date keys from a date context object.

        Args:
            date_context: The date context from generate_date_context_object()

        Returns:
            Flat list of all valid date keys
        """
        keys: List[str] = []

        # Current
        keys.extend(["today", "now"])

        # Relative dates
        keys.extend(date_context.get("relative_dates", {}).keys())

        # Weekdays
        keys.extend(date_context.get("weekdays", {}).keys())

        # Weekend
        keys.extend(["this_weekend", "next_weekend", "last_weekend"])

        # Weeks
        keys.extend(["this_week", "next_week", "last_week"])

        # Years
        keys.extend(["this_year", "next_year", "last_year"])

        # Time expressions
        keys.extend(date_context.get("time_expressions", {}).keys())

        return keys

