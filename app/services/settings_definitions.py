"""Settings definitions for jarvis-command-center.

Defines all configurable settings with their types, defaults, and metadata.
Uses the shared jarvis-settings-client library.
"""

from jarvis_settings_client import SettingDefinition

from app.services.persona_presets import DEFAULT_PERSONA


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

    # Signal Bus settings
    SettingDefinition(
        key="signals.enabled",
        category="signals",
        value_type="bool",
        default=True,
        description="Enable the Signal Bus ingress + reactive rendering for this household",
    ),
    SettingDefinition(
        key="signals.automations",
        category="signals",
        value_type="string",
        default="{}",
        description=(
            "Per-household signal automations as a JSON object mapping a signal "
            "kind to {\"instruction\": str, \"enabled\": bool} — the user's "
            "free-text instruction for what should happen when that signal fires "
            "(interpreted at fire time). Edited from mobile via the "
            "signal-automations endpoint; the authorable kinds come from "
            "signal_catalog."
        ),
    ),

    # Tool router settings
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
        key="model.advanced_context",
        category="model",
        value_type="bool",
        default=False,
        description=(
            "Enable proactive context injection — the background agents' weather, "
            "calendar, news, and reminder memories are accepted and injected into the "
            "prompt so the assistant can answer 'how's today looking?' proactively. "
            "Decoupled from chain-of-thought (see model.include_thinking)."
        ),
    ),
    SettingDefinition(
        key="model.include_thinking",
        category="model",
        value_type="bool",
        default=False,
        description=(
            "Allow the model to emit chain-of-thought (<think>) before answering "
            "(/think vs /no_think on Qwen3). Improves hard queries but adds significant "
            "latency (~500 tokens, several seconds), so it defaults off."
        ),
    ),

    # Conversation settings
    SettingDefinition(
        key="conversation.max_turns",
        category="conversation",
        value_type="int",
        default=10,
        description=(
            "Maximum user/assistant exchanges kept in conversation history "
            "(sliding window — oldest turns are dropped first, atomically with "
            "their tool calls/results; keeps prompts bounded for the LLM)"
        ),
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

    # Household speaking-voice persona (the "voice" layer — VOICE/tone only,
    # walled off from tools + safety). Per-household so it rides the cached
    # prefix. Default = warm & folksy, so households that never touch the box
    # still get a voice warmer than the old flat "function calling assistant".
    SettingDefinition(
        key="persona.household_prompt",
        category="persona",
        value_type="string",
        default=DEFAULT_PERSONA,
        description=(
            "Household speaking-voice persona injected into the voice prompt as a "
            "fenced <personality> block. Shapes tone and word choice ONLY — never "
            "tool-calling or safety, which stay non-overridable. Per-household; "
            "editable from mobile (starter presets + free text). Empty = no voice "
            "layer (falls back to the flat identity line)."
        ),
    ),

    # Characterization — Jarvis's evolving synthesized "view of a person", built
    # in the background from their facts + recent transcripts. Synthesis and
    # injection are gated SEPARATELY so a household can build + inspect the view
    # long before it ever shapes a live prompt. Both default OFF.
    SettingDefinition(
        key="characterization.synthesis_enabled",
        category="characterization",
        value_type="bool",
        default=False,
        description=(
            "Enable the background job that synthesizes a per-person characterization "
            "from their facts + recent transcripts. Off the hot path; default OFF."
        ),
    ),
    SettingDefinition(
        key="characterization.synthesis_interval_seconds",
        category="characterization",
        value_type="int",
        default=3600,
        description=(
            "Interval in seconds between characterization synthesis batch runs "
            "(e.g., 300 for dev, 3600 for prod)."
        ),
    ),
    SettingDefinition(
        key="characterization.max_transcripts",
        category="characterization",
        value_type="int",
        default=50,
        description=(
            "Max recent transcripts fed into a single characterization synthesis "
            "prompt (bounds the background model's input)."
        ),
    ),
    SettingDefinition(
        key="characterization.injection_enabled",
        category="characterization",
        value_type="bool",
        default=False,
        description=(
            "Allow the synthesized characterization to be injected into the live "
            "voice prompt as a <person_view> tail. Default OFF, fail-closed — "
            "separate from synthesis so the view can be built + inspected first."
        ),
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
    SettingDefinition(
        key="ambient_context.enabled",
        category="memory",
        value_type="bool",
        default=False,
        description=(
            "Inject an always-on ambient situational block (current time, weather, "
            "today's calendar) into the CACHED prompt prefix — snapshotted once at "
            "conversation start — so the assistant can answer 'how's today looking?' "
            "proactively without a tool call. Opt-in; when on, consider turning "
            "memory.agent_context_enabled off for this household to avoid double-"
            "injecting weather/calendar."
        ),
    ),

    # Web search settings
    SettingDefinition(
        key="web_search.enabled",
        category="web_search",
        value_type="bool",
        default=False,
        description=(
            "Master toggle for web search. Gates the live quick_search lookups "
            "and deep_research tools, which make outbound requests to the "
            "internet. Default OFF so local-only households never egress."
        ),
    ),
    # Proposable actions (agent → command confirm cards)
    SettingDefinition(
        key="proposals.enabled",
        category="proposals",
        value_type="bool",
        default=False,
        description=(
            "Master toggle for agent-proposed action cards. Gates the "
            "proposable-action dispatcher: when off (default), a tapped 'agent "
            "proposes a command' card (e.g. an email agent's 'Add to calendar?') "
            "is refused server-side. Fail-closed — any settings error disables it "
            "— because it lets background agents originate real writes."
        ),
    ),
    SettingDefinition(
        key="proposals.proactive_enabled",
        category="proposals",
        value_type="bool",
        default=False,
        description=(
            "Enable the PROACTIVE Signal reasoner (situation matcher). When off "
            "(default), Signal edges never trigger a background reason pass; the "
            "reactive 'who's home' rendering still works. Requires proposals.enabled "
            "as well. Runs on the background model, only ever proposes a card. Keep "
            "off until the offline precision harness clears the model. Fail-closed."
        ),
    ),
    SettingDefinition(
        key="errands.autonomous_enabled",
        category="errands",
        value_type="bool",
        default=False,
        description=(
            "Allow a Signal to AUTORUN a low-blast multi-step plan without a "
            "tap-to-confirm card (e.g. an upcoming appointment auto-checks drive time "
            "and sets a leave-by reminder). When off (default), a signal-triggered "
            "plan is proposed as a card instead of executed. Even when on, only plans "
            "that pass the plan-start blast gate (allowlisted, non-risky, no outbound "
            "counterparty, few steps) autorun; anything else falls back to a card. "
            "Fail-closed — any settings error disables autorun — because it lets a "
            "background signal originate real writes with no human confirmation."
        ),
    ),
    SettingDefinition(
        key="web_scraping.allow_external",
        category="web_search",
        value_type="bool",
        default=False,
        description=(
            "Permit the deep_research scraper to fall back to the third-party "
            "r.jina.ai reader proxy when a page can't be fetched directly. This "
            "leaks which pages the household reads to a third party. Default OFF; "
            "shares the web_search mobile screen."
        ),
    ),

    # Update check settings
    SettingDefinition(
        key="updates.allow_check",
        category="updates",
        value_type="bool",
        default=False,
        description=(
            "Allow outbound update-version lookups to api.github.com (node "
            "release checks). Default OFF so local-only households never egress "
            "to GitHub; an explicit version can still be installed without it."
        ),
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
    SettingDefinition(
        key="voice.wake_verification_mode",
        category="voice",
        value_type="string",
        default="off",
        description=(
            "Wake-clip verification: transcribe the ~2s clip that fired the "
            "wake word and check it actually contains the wake phrase. "
            "'off' = disabled; 'bias' = an unverified wake injects a mild "
            "misfire-leaning prompt hint (one signal, never a suppression "
            "instruction); 'enforce' = an unverified wake is silently "
            "dropped (like the STT-noise prefilter). Catches "
            "high-confidence openWakeWord misfires that score/VAD cannot "
            "(prod 2026-08-15: a 0.95 misfire marked a medication as taken "
            "off overheard family talk). Fail SAFE: missing rows and "
            "unrecognized values both mean 'off', and missing/pending/"
            "errored verdicts always fail OPEN."
        ),
    ),
    SettingDefinition(
        key="voice.followup_doubt_max_rounds",
        category="voice",
        value_type="int",
        default=2,
        description=(
            "Answered (non-sentinel) rounds a DOUBTED conversation — one "
            "whose wake-clip verdict was 'unverified' — gets before "
            "follow-up turns gain a strong wrap-up lean (answer briefly "
            "and close with <exchange_complete/>, or <not_for_me/> if not "
            "addressed). A prompt lean, never a hard cut; conversations "
            "with a verified wake are exempt. Breaks the kitchen runaway "
            "(2026-08-15): an answered false wake opened the follow-up "
            "window onto the family's conversation and Jarvis answered "
            "non-stop."
        ),
    ),
    SettingDefinition(
        key="voice.tool_dedupe_window_seconds",
        category="voice",
        value_type="float",
        default=120.0,
        description=(
            "Seconds an issued CLIENT tool call (name + canonicalized "
            "args) blocks an IDENTICAL re-issue in the same conversation. "
            "A repeat inside the window is answered with a one-shot system "
            "nudge ('the results are already in this conversation above — "
            "answer from them') instead of a 202; whatever the model "
            "returns next is accepted, and a second identical request "
            "after the nudge passes through (fail-open — the model may "
            "genuinely want a refresh). 0 disables. Server tools are "
            "excluded. Breaks the 2026-08-15 calendar loop: three "
            "identical get_calendar_events in 25s, each follow-up "
            "re-reading the whole calendar aloud."
        ),
    ),
    SettingDefinition(
        key="voice.wake_verification_phrase",
        category="voice",
        value_type="string",
        default="jarvis",
        description=(
            "The word the wake clip must (fuzzily) contain to count as "
            "verified. Matches within edit distance 2, so whisper's "
            "manglings of short clips ('travis', 'jervis') still verify."
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

    # Routines: scheduler + execution audit retention
    SettingDefinition(
        key="routines.scheduler_enabled",
        category="routines",
        value_type="bool",
        default=False,
        description="Enable the CC-side cron/interval scheduler that fires scheduled routines",
    ),
    SettingDefinition(
        key="routines.scheduler_interval_seconds",
        category="routines",
        value_type="int",
        default=30,
        description="How often the routine scheduler wakes to check for due routines",
    ),
    SettingDefinition(
        key="routines.execution_ttl_days",
        category="routines",
        value_type="int",
        default=7,
        description="Days to retain routine_executions audit rows before cleanup",
    ),

    # Attention broker: deterministic governance of proactive notifications
    # (prds/attention-broker.md). Off by default; per-household opt-in.
    SettingDefinition(
        key="attention.enabled",
        category="attention",
        value_type="bool",
        default=False,
        description="Route proactive notifications through the attention broker (journal, dedup, budgets, quiet hours)",
    ),
    SettingDefinition(
        key="attention.daily_push_budget",
        category="attention",
        value_type="int",
        default=8,
        description="Max broker-routed push notifications per household per local day (exhausted budget demotes to inbox)",
    ),
    SettingDefinition(
        key="attention.daily_inbox_budget",
        category="attention",
        value_type="int",
        default=30,
        description="Max broker-routed inbox cards per household per local day (exhausted budget demotes to journal)",
    ),
    SettingDefinition(
        key="attention.source_daily_cap",
        category="attention",
        value_type="int",
        default=4,
        description="Max delivered items per source per household per local day",
    ),
    SettingDefinition(
        key="attention.dedupe_window_hours",
        category="attention",
        value_type="int",
        default=24,
        description="Window in which a repeated (source, dedupe_key) is journaled as a duplicate instead of re-delivered",
    ),
    SettingDefinition(
        key="attention.quiet_hours",
        category="attention",
        value_type="string",
        default="22:00-07:00",
        description="Household-local window (HH:MM-HH:MM) during which push demotes to inbox; safety categories exempt",
    ),
    SettingDefinition(
        key="attention.timezone",
        category="attention",
        value_type="string",
        default="UTC",
        description="IANA timezone for quiet hours and daily budget resets (set per household)",
    ),
    SettingDefinition(
        key="attention.safety_categories",
        category="attention",
        value_type="string",
        default="medication,reminder,security,safety",
        description="Comma-separated categories that bypass tiers, budgets, and quiet hours (dedup still applies); mute is refused",
    ),
    SettingDefinition(
        key="attention.journal_ttl_days",
        category="attention",
        value_type="int",
        default=30,
        description="Days to retain attention events/deliveries before cleanup",
    ),
    SettingDefinition(
        key="attention.journal_card_enabled",
        category="attention",
        value_type="bool",
        default=True,
        description="Post the daily attention-journal inbox card (delivered + withheld summary)",
    ),
    SettingDefinition(
        key="attention.journal_card_cron",
        category="attention",
        value_type="string",
        default="0 21 * * *",
        description="Cron (household tz) for the daily attention-journal card",
    ),

    # Phone calls (phone-calls PRD) — all fail-closed, household-scoped.
    SettingDefinition(
        key="phone_calls.enabled",
        category="phone_calls",
        value_type="bool",
        default=False,
        description="Master toggle for AI phone calls (default OFF, fail-closed)",
    ),
    SettingDefinition(
        key="phone_calls.call_context",
        category="phone_calls",
        value_type="string",
        default="",
        description=(
            "Per-user details the call agent may use (JSON). Scoped by "
            "user_id, not household: insurance and callback numbers are "
            "personal. See app/services/call_context.py for the shape, the "
            "category (is it loaded at all) and tier (may it be volunteered) "
            "controls. Contains PII — anything reaching a call can be spoken "
            "aloud and lands in the transcript and recording."
        ),
        is_secret=True,
    ),
    SettingDefinition(
        key="phone_calls.plan_ttl_minutes",
        category="phone_calls",
        value_type="int",
        default=20,
        description="Minutes a call plan (confirm card) stays valid before expiring",
    ),
    SettingDefinition(
        key="phone_calls.audio_retention_days",
        category="phone_calls",
        value_type="int",
        default=30,
        description="Days to retain call audio AND transcript before reaping",
    ),
    SettingDefinition(
        key="phone_calls.max_call_seconds",
        category="phone_calls",
        value_type="int",
        default=600,
        description="Hard cap on a single call's duration (reaper enforces)",
    ),
    SettingDefinition(
        key="phone_calls.attempt_cap",
        category="phone_calls",
        value_type="int",
        default=2,
        description="Max dial attempts per task",
    ),
    SettingDefinition(
        key="phone_calls.calls_per_day",
        category="phone_calls",
        value_type="int",
        default=10,
        description="Per-household daily call/plan cap (also caps pending plans)",
    ),
    SettingDefinition(
        key="phone_calls.monthly_minutes_cap",
        category="phone_calls",
        value_type="int",
        default=60,
        description="Per-household monthly call minutes cap (fail-closed when exceeded)",
    ),
    SettingDefinition(
        key="household.location",
        category="household",
        value_type="string",
        default="",
        description=(
            "Household locality used to disambiguate business searches — "
            "e.g. 'Springfield, IL 62704', 'Springfield, IL', or a bare ZIP. Deliberately "
            "NOT a street address: this picks WHICH business, it is not used "
            "for directions and is never spoken on a call. Empty means "
            "searches run unbiased (the pre-2026-07 behavior)."
        ),
    ),
    SettingDefinition(
        key="phone_calls.max_concurrent_calls",
        category="phone_calls",
        value_type="int",
        default=1,
        description="Max simultaneous active calls per household",
    ),
]
