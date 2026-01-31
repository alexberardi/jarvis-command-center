"""
Mock LLM responses for testing the command center.

These factory functions create response dictionaries that match what the
LLMProxyClient returns from the LLM proxy API.
"""

import json
from typing import Any, Dict, List, Optional


def create_llm_tool_response(
    name: str,
    arguments: Dict[str, Any],
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an LLM response with a single tool call.

    Args:
        name: Tool name to call
        arguments: Tool arguments as a dictionary
        message: Optional assistant message

    Returns:
        Dict matching OpenAI chat completion format with JSON content
    """
    content = {
        "message": message or "",
        "tool_calls": [
            {
                "name": name,
                "arguments": arguments
            }
        ]
    }

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_multi_tool_response(
    tool_calls: List[Dict[str, Any]],
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an LLM response with multiple tool calls.

    Args:
        tool_calls: List of tool calls, each with 'name' and 'arguments'
        message: Optional assistant message

    Returns:
        Dict matching OpenAI chat completion format
    """
    content = {
        "message": message or "",
        "tool_calls": [
            {
                "name": tc["name"],
                "arguments": tc.get("arguments", {})
            }
            for tc in tool_calls
        ]
    }

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_complete_response(message: str) -> Dict[str, Any]:
    """
    Create an LLM response with no tool calls (complete).

    Args:
        message: The assistant's final message

    Returns:
        Dict matching OpenAI chat completion format with empty tool_calls
    """
    content = {
        "message": message,
        "tool_calls": []
    }

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_openai_format_response(
    name: str,
    arguments: Dict[str, Any],
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an LLM response with OpenAI function wrapper format.

    Some models return tool calls in the format:
    {"function": {"name": "...", "arguments": {...}}}

    Args:
        name: Tool name
        arguments: Tool arguments
        message: Optional message

    Returns:
        Dict with OpenAI function wrapper format
    """
    content = {
        "message": message or "",
        "tool_calls": [
            {
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            }
        ]
    }

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_string_arguments_response(
    name: str,
    arguments_json: str,
    message: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an LLM response with arguments as JSON string (not dict).

    Some models return arguments as a JSON string instead of a dict.

    Args:
        name: Tool name
        arguments_json: Arguments as a JSON string
        message: Optional message

    Returns:
        Dict with string arguments
    """
    content = {
        "message": message or "",
        "tool_calls": [
            {
                "name": name,
                "arguments": arguments_json  # String, not dict
            }
        ]
    }

    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content),
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_markdown_response(json_content: str) -> Dict[str, Any]:
    """
    Create an LLM response with JSON wrapped in markdown code block.

    Some models wrap their JSON output in ```json ... ``` blocks.

    Args:
        json_content: The JSON string to wrap in markdown

    Returns:
        Dict with markdown-wrapped content
    """
    markdown_content = f"Here is the response:\n\n```json\n{json_content}\n```"

    return {
        "choices": [
            {
                "message": {
                    "content": markdown_content,
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_response_with_comments(json_with_comments: str) -> Dict[str, Any]:
    """
    Create an LLM response with JSON containing comments.

    Some models add # comments to their JSON output.

    Args:
        json_with_comments: JSON string that may contain # comments

    Returns:
        Dict with content containing comments
    """
    return {
        "choices": [
            {
                "message": {
                    "content": json_with_comments,
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


def create_llm_plain_text_response(text: str) -> Dict[str, Any]:
    """
    Create an LLM response with plain text (not JSON).

    When the LLM doesn't follow the JSON format.

    Args:
        text: Plain text response

    Returns:
        Dict with plain text content
    """
    return {
        "choices": [
            {
                "message": {
                    "content": text,
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }
