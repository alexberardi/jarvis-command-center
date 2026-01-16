"""
Date Detection and Resolution Utility.

Automatically detects relative date phrases in user utterances and resolves them
before sending to the LLM.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

from app.core.general_context import generate_date_context_object

logger = logging.getLogger("uvicorn")


# Common relative date phrases to detect
RELATIVE_DATE_PATTERNS = [
    # Today/now
    r'\btoday\b',
    r'\bnow\b',
    
    # Tomorrow/yesterday
    r'\btomorrow\b',
    r'\byesterday\b',
    r'\bday after tomorrow\b',
    r'\bday before yesterday\b',
    
    # Weekends
    r'\bthis weekend\b',
    r'\bnext weekend\b',
    r'\blast weekend\b',
    
    # Weeks
    r'\bthis week\b',
    r'\bnext week\b',
    r'\blast week\b',
    
    # Weekdays
    r'\bnext (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\blast (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    r'\bthis (monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
    
    # Time expressions
    r'\btonight\b',
    r'\bthis morning\b',
    r'\bthis afternoon\b',
    r'\bthis evening\b',
    r'\btomorrow morning\b',
    r'\btomorrow afternoon\b',
    r'\btomorrow evening\b',
    r'\btomorrow night\b',
    r'\byesterday morning\b',
    r'\byesterday afternoon\b',
    r'\byesterday evening\b',
    r'\blast night\b',
    
    # Months/years
    r'\bthis month\b',
    r'\bnext month\b',
    r'\blast month\b',
    r'\bthis year\b',
    r'\bnext year\b',
    r'\blast year\b',
]


def detect_relative_dates(utterance: str) -> List[str]:
    """
    Detect relative date phrases in a user utterance.
    
    Args:
        utterance: The user's utterance text
        
    Returns:
        List of detected relative date phrases (normalized)
    """
    if not utterance:
        return []
    
    utterance_lower = utterance.lower()
    detected = []
    
    # Check each pattern
    for pattern in RELATIVE_DATE_PATTERNS:
        matches = re.finditer(pattern, utterance_lower)
        for match in matches:
            phrase = match.group(0).strip()
            # Normalize: replace spaces with underscores
            normalized = phrase.replace(" ", "_")
            if normalized not in detected:
                detected.append(normalized)
    
    return detected


def resolve_relative_date_term(relative_term: str, timezone: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Resolve a single relative date term to actual dates.
    
    Uses the same logic as ResolveRelativeDateTool.execute().
    
    Args:
        relative_term: Relative date term (e.g., "tomorrow", "last_weekend")
        timezone: Optional timezone string
        
    Returns:
        Dict with resolved date(s), or None if not found
    """
    try:
        # Generate full date context
        date_context = generate_date_context_object(timezone)
        
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
        
        return result
        
    except Exception as e:
        logger.error(f"Error resolving relative date term '{relative_term}': {e}")
        return None


def detect_and_resolve_relative_dates(
    utterance: str,
    timezone: Optional[str] = None
) -> Dict[str, Any]:
    """
    Detect and resolve all relative date phrases in an utterance.
    
    Args:
        utterance: The user's utterance text
        timezone: Optional timezone string (will use conversation timezone if available)
        
    Returns:
        Dict with:
            - detected_terms: List of detected relative date phrases
            - resolved_dates: Dict mapping terms to resolved date data
            - summary: Formatted string for inclusion in prompt
    """
    detected_terms = detect_relative_dates(utterance)
    
    if not detected_terms:
        return {
            "detected_terms": [],
            "resolved_dates": {},
            "summary": None
        }
    
    logger.info(f"📅 Detected {len(detected_terms)} relative date phrase(s): {detected_terms}")
    
    resolved_dates = {}
    for term in detected_terms:
        resolved = resolve_relative_date_term(term, timezone)
        if resolved:
            resolved_dates[term] = resolved
            logger.info(f"   ✓ Resolved '{term}' → {resolved.get('date') or resolved.get('datetime') or 'multiple dates'}")
        else:
            logger.warning(f"   ✗ Could not resolve '{term}'")
    
    # Build summary for prompt
    if resolved_dates:
        summary_lines = ["Detected Relative Dates:"]
        for term, data in resolved_dates.items():
            if "dates" in data:
                # Multiple dates (weekend, week, etc.)
                dates_list = [d.get("utc_start_of_day") or d.get("date") for d in data["dates"]]
                summary_lines.append(f"  - '{term}': {dates_list}")
            elif "utc_start_of_day" in data:
                summary_lines.append(f"  - '{term}': {data['utc_start_of_day']}")
            elif "datetime" in data:
                summary_lines.append(f"  - '{term}': {data['datetime']}")
            elif "date" in data:
                summary_lines.append(f"  - '{term}': {data['date']}")
        
        summary = "\n".join(summary_lines)
    else:
        summary = None
    
    return {
        "detected_terms": detected_terms,
        "resolved_dates": resolved_dates,
        "summary": summary
    }

