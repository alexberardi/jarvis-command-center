"""
JSON utilities for handling LLM responses and validation.
"""

import json
import logging
from typing import TypeVar, Type, Optional, Tuple, Dict, Any

logger = logging.getLogger("uvicorn")

# JSON validation statistics
json_validation_stats = {
    "total_requests": 0,
    "invalid_json_count": 0,
    "legacy_format_conversions": 0,
    "auto_fixes": 0,
    "fallback_extractions": 0,
    "completely_failed": 0
}

T = TypeVar('T')

def validate_and_clean_json_response(raw_response: str, original_voice_command: str = None) -> Tuple[bool, list]:
    """
    Pure JSON validation function - only validates structure, does not transform data
    Returns: (is_valid, issues_list)
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
        return False, issues

    # Extract just the JSON portion
    json_portion = cleaned[json_start:json_end + 1]

    # Try to parse the JSON
    try:
        parsed_json = json.loads(json_portion)

        # Validate structure
        if not isinstance(parsed_json, dict):
            issues.append("Response is not a JSON object")
            return False, issues

        # Check for required keys (either compressed or full format)
        has_compressed_format = "c" in parsed_json
        has_full_format = "commands" in parsed_json
        has_single_command = any(key in parsed_json for key in ["s", "success", "n", "command_name", "p", "parameters", "e", "errors"])
        
        if not has_compressed_format and not has_full_format and not has_single_command:
            issues.append("Missing 'commands', 'c', or single command keys in response")
            json_validation_stats["invalid_json_count"] += 1
            return False, issues

        # Check that commands is a list (for command inference) or validate single command structure
        if has_compressed_format or has_full_format:
            commands_key = "c" if has_compressed_format else "commands"
            if not isinstance(parsed_json[commands_key], list):
                issues.append(f"'{commands_key}' is not a list")
                return False, issues
        elif has_single_command:
            # For single command responses (parameter inference), validate the structure
            if "s" in parsed_json and not isinstance(parsed_json["s"], bool):
                issues.append("'s' field is not boolean")
                return False, issues
            if "n" in parsed_json and not isinstance(parsed_json["n"], str):
                issues.append("'n' field is not string")
                return False, issues
            if "p" in parsed_json and not isinstance(parsed_json["p"], dict):
                issues.append("'p' field is not object")
                return False, issues

        # Basic validation passed
        return True, issues

    except json.JSONDecodeError as e:
        issues.append(f"Invalid JSON: {str(e)}")
        json_validation_stats["invalid_json_count"] += 1
        return False, issues

def parse_json_to_type(raw_response: str, target_class: Type[T], original_voice_command: str = None) -> Optional[T]:
    """
    Parse JSON response and return an instance of the target class.
    
    Args:
        raw_response: Raw JSON string from LLM
        target_class: The class type to parse to
        original_voice_command: Original voice command for context
        
    Returns:
        Parsed instance of target_class or None if parsing fails
    """
    try:
        # First validate the JSON structure
        is_valid, issues = validate_and_clean_json_response(raw_response, original_voice_command)
        if not is_valid:
            logger.warning(f"JSON validation failed: {issues}")
            return None
        
        # Extract JSON portion
        cleaned = raw_response.strip()
        json_start = cleaned.find('{')
        json_end = cleaned.rfind('}')
        json_portion = cleaned[json_start:json_end + 1]
        
        # Parse to dict first
        parsed_dict = json.loads(json_portion)
        
        # Try to instantiate the target class
        # This assumes the target class can be constructed from the parsed dict
        # You might need to adjust this based on your specific classes
        return target_class(**parsed_dict)
        
    except Exception as e:
        logger.error(f"Failed to parse JSON to {target_class.__name__}: {e}")
        return None

def get_json_validation_stats() -> Dict[str, Any]:
    """Get current JSON validation statistics."""
    return json_validation_stats.copy()
