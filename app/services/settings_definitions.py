"""Settings definitions for jarvis-command-center.

Defines all configurable settings with their types, defaults, and metadata.
Uses the shared jarvis-settings-client library.
"""

from jarvis_settings_client import SettingDefinition


# Command-center settings definitions
SETTINGS_DEFINITIONS: list[SettingDefinition] = [
    # Model/LLM settings
    SettingDefinition(
        key="llm.interface",
        category="llm",
        value_type="string",
        default="HermesMediumUntrained",
        description="LLM interface type for command processing",
    ),
    SettingDefinition(
        key="llm.proxy.url",
        category="llm",
        value_type="string",
        default="http://localhost:7704",
        description="URL of the LLM proxy API service",
        env_fallback="JARVIS_LLM_PROXY_API_URL",
        requires_reload=True,
    ),

    # Tool classifier settings
    SettingDefinition(
        key="tool_classifier.enabled",
        category="tool_classifier",
        value_type="bool",
        default=True,
        description="Enable/disable the tool classifier for pre-filtering",
        env_fallback="JARVIS_TOOL_CLASSIFIER_ENABLED",
    ),
    SettingDefinition(
        key="tool_classifier.min_confidence",
        category="tool_classifier",
        value_type="float",
        default=0.6,
        description="Minimum confidence threshold for tool classification",
        env_fallback="JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE",
    ),

    # Tool router settings
    SettingDefinition(
        key="tool_router.filter_min_confidence",
        category="tool_router",
        value_type="float",
        default=0.85,
        description="Minimum confidence for tool router filtering",
        env_fallback="JARVIS_TOOL_ROUTER_FILTER_MIN_CONFIDENCE",
    ),

    # Transcription settings
    SettingDefinition(
        key="transcription.cleanup_enabled",
        category="transcription",
        value_type="bool",
        default=False,
        description="Enable/disable transcription cleanup post-processing",
        env_fallback="JARVIS_TRANSCRIPTION_CLEANUP_ENABLED",
    ),

    # Prompt generation settings
    SettingDefinition(
        key="prompt.include_antipatterns",
        category="prompt",
        value_type="bool",
        default=True,
        description="Include antipatterns in prompt generation",
        env_fallback="JARVIS_PROMPT_INCLUDE_ANTIPATTERNS",
    ),
    SettingDefinition(
        key="prompt.include_param_descriptions",
        category="prompt",
        value_type="bool",
        default=True,
        description="Include parameter descriptions in prompts",
        env_fallback="JARVIS_PROMPT_INCLUDE_PARAM_DESCRIPTIONS",
    ),

    # Model mode settings
    SettingDefinition(
        key="model.small_model_mode",
        category="model",
        value_type="bool",
        default=True,
        description="Optimize prompts for smaller models",
        env_fallback="JARVIS_SMALL_MODEL_MODE",
    ),

    # Conversation settings
    SettingDefinition(
        key="conversation.max_turns",
        category="conversation",
        value_type="int",
        default=20,
        description="Maximum number of conversation turns",
        env_fallback="JARVIS_CONVERSATION_MAX_TURNS",
    ),
    SettingDefinition(
        key="conversation.cache_ttl_seconds",
        category="conversation",
        value_type="int",
        default=3600,
        description="Conversation cache TTL in seconds",
        env_fallback="JARVIS_CONVERSATION_CACHE_TTL",
    ),

    # Memory settings
    SettingDefinition(
        key="memory.enabled",
        category="memory",
        value_type="bool",
        default=True,
        description="Master toggle for remember/forget/recall tools",
    ),
    SettingDefinition(
        key="memory.recall_enabled",
        category="memory",
        value_type="bool",
        default=True,
        description="Enable semantic search via the recall tool",
    ),
    SettingDefinition(
        key="memory.pinned_max_chars",
        category="memory",
        value_type="int",
        default=500,
        description="Maximum characters for pinned memories in the system prompt",
    ),
    SettingDefinition(
        key="memory.recall_similarity_threshold",
        category="memory",
        value_type="float",
        default=0.3,
        description="Minimum cosine similarity for recall results (0-1)",
    ),
    SettingDefinition(
        key="memory.recall_max_results",
        category="memory",
        value_type="int",
        default=5,
        description="Maximum number of recall results returned",
    ),

    # Admin settings
    SettingDefinition(
        key="admin.api_key",
        category="admin",
        value_type="string",
        default="",
        description="Admin API key for protected endpoints",
        env_fallback="ADMIN_API_KEY",
        is_secret=True,
    ),
]
