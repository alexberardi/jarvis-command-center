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
        default="Qwen25MediumUntrained",
        description="LLM interface type for command processing",
        requires_reload=True,
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
    SettingDefinition(
        key="model.advanced_thinking",
        category="model",
        value_type="bool",
        default=False,
        description=(
            "Enable chain-of-thought reasoning and proactive context injection "
            "(weather, calendar, news). Adds ~2s latency but improves response "
            "quality for complex queries."
        ),
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

    # Passive memory extraction settings
    SettingDefinition(
        key="memory.extraction_enabled",
        category="memory",
        value_type="bool",
        default=True,
        description="Enable passive memory extraction from conversation transcripts",
    ),
    SettingDefinition(
        key="memory.extraction_interval_seconds",
        category="memory",
        value_type="int",
        default=300,
        description="Interval in seconds between extraction batch runs (e.g., 30 for dev, 3600 for prod)",
    ),
    SettingDefinition(
        key="memory.transcript_ttl_days",
        category="memory",
        value_type="int",
        default=7,
        description="Days to retain conversation transcripts before cleanup",
    ),

    # Agent context injection settings (background agents → prompt)
    SettingDefinition(
        key="memory.agent_context_enabled",
        category="memory",
        value_type="bool",
        default=True,
        description="Enable agent-injected context retrieval during voice commands",
    ),
    SettingDefinition(
        key="memory.agent_context_max_results",
        category="memory",
        value_type="int",
        default=5,
        description="Maximum number of agent context items to inject into the prompt",
    ),
    SettingDefinition(
        key="memory.agent_context_max_chars",
        category="memory",
        value_type="int",
        default=500,
        description="Maximum characters for agent context in the system prompt",
    ),
    SettingDefinition(
        key="memory.agent_context_similarity_threshold",
        category="memory",
        value_type="float",
        default=0.25,
        description="Minimum cosine similarity for agent context vector search (0-1)",
    ),

    # Phase 5 — Auto-deploy LoRA adapter loop
    SettingDefinition(
        key="adapter.auto_train_enabled",
        category="adapter",
        value_type="bool",
        default=False,
        description="Run the periodic adapter-training loop (scheduler + auto-deploy)",
    ),
    SettingDefinition(
        key="adapter.auto_train_interval_seconds",
        category="adapter",
        value_type="int",
        default=3600,
        description="Interval between scheduler ticks that check whether training should fire",
    ),
    SettingDefinition(
        key="adapter.auto_train_min_examples",
        category="adapter",
        value_type="int",
        default=50,
        description="Minimum new positive examples since last training before a new run is triggered",
    ),
    SettingDefinition(
        key="adapter.auto_train_min_interval_hours",
        category="adapter",
        value_type="int",
        default=6,
        description="Minimum hours between successful training runs per household (rate limiter)",
    ),
    SettingDefinition(
        key="adapter.base_model_id",
        category="adapter",
        value_type="string",
        default="",
        description="Base model id/path passed to llm-proxy when the scheduler enqueues training",
    ),
    SettingDefinition(
        key="adapter.commands_spec_path",
        category="adapter",
        value_type="string",
        default="",
        description="Filesystem path to the commands spec JSON used for synthetic expansion + prompt variants",
    ),
    SettingDefinition(
        key="adapter.llm_proxy_url",
        category="adapter",
        value_type="string",
        default="",
        description="LLM proxy base URL for training enqueue (leave blank to use service discovery)",
        env_fallback="JARVIS_LLM_PROXY_API_URL",
    ),
    SettingDefinition(
        key="adapter.eval_container",
        category="adapter",
        value_type="string",
        default="jarvis-node",
        description="Docker container name for docker-exec eval gate",
    ),
    SettingDefinition(
        key="adapter.expansion_enabled",
        category="adapter",
        value_type="bool",
        default=False,
        description="Expand each baked adapter example via llm-proxy before training (Phase 3.5)",
    ),
    SettingDefinition(
        key="adapter.expansion_count_per_canonical",
        category="adapter",
        value_type="int",
        default=2,
        description="Number of rephrasings to generate per baked canonical example when expansion is enabled",
    ),
    SettingDefinition(
        key="adapter.expansion_max_canonicals",
        category="adapter",
        value_type="int",
        default=10,
        description="Cap on canonical examples used as expansion seeds per command",
    ),

    # Phase 7 — User-approved adapter deployment
    SettingDefinition(
        key="adapter.auto_approve_deployments",
        category="adapter",
        value_type="bool",
        default=False,
        description=(
            "Bypass the proposal flow and deploy new adapters automatically "
            "on eval PASS (Phase 5 behavior). Default off — the scheduler "
            "posts a proposal to the inbox instead."
        ),
    ),
    SettingDefinition(
        key="adapter.proposal_expiry_days",
        category="adapter",
        value_type="int",
        default=7,
        description=(
            "How long a pending adapter proposal stays valid before the "
            "janitor flips it to expired. Users can still apply up until "
            "this cutoff."
        ),
    ),

    # Network / callback settings
    SettingDefinition(
        key="network.public_url",
        category="network",
        value_type="string",
        default="",
        description="LAN-reachable URL for this service (used in callbacks to remote workers)",
        env_fallback="CC_PUBLIC_URL",
    ),

    # OAuth settings
    SettingDefinition(
        key="oauth.relay_url",
        category="oauth",
        value_type="string",
        default="",
        description="Relay URL for OAuth bounce (external providers like Google)",
        env_fallback="JARVIS_RELAY_URL",
    ),
    SettingDefinition(
        key="oauth.external_url",
        category="oauth",
        value_type="string",
        default="",
        description="Public base URL for this service (used as OAuth redirect URI base)",
        env_fallback="JARVIS_EXTERNAL_URL",
    ),

    # Smart home settings
    SettingDefinition(
        key="smart_home.use_home_assistant",
        category="smart_home",
        value_type="bool",
        default=False,
        description="Use Home Assistant for device control and aggregation (vs direct WiFi only)",
    ),
    SettingDefinition(
        key="smart_home.device_manager",
        category="smart_home",
        value_type="string",
        default="jarvis_direct",
        description="Active device manager for device listing",
        options=["jarvis_direct", "home_assistant"],
    ),
    SettingDefinition(
        key="smart_home.primary_node_id",
        category="smart_home",
        value_type="string",
        default="",
        description="Node that handles device management (discovery, listing) for the household",
    ),
    SettingDefinition(
        key="smart_home.use_external_devices",
        category="smart_home",
        value_type="bool",
        default=False,
        description="Show devices from the node's device manager instead of the CC database",
    ),

    # Voice / speaker stickiness
    SettingDefinition(
        key="voice.stickiness_min_confidence",
        category="voice",
        value_type="float",
        default=0.55,
        description=(
            "Minimum speaker-recognition confidence to record a node's "
            "speaker for short follow-up inheritance. When the next short "
            "utterance comes in unidentified, this node's most recent "
            "high-confidence speaker is reused if within the TTL. Tune to "
            "match the encoder's score distribution — ECAPA same-speaker "
            "scores typically cluster 0.45-0.70; resemblyzer 0.75-0.90."
        ),
    ),
    SettingDefinition(
        key="voice.stickiness_ttl_seconds",
        category="voice",
        value_type="float",
        default=30.0,
        description=(
            "Seconds an identified speaker stays sticky for a node. "
            "After this, a fresh identification is required."
        ),
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
