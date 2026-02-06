"""Seed default settings

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-02-05 17:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

revision = 'f6a7b8c9d0e1'
down_revision = 'e5f6a7b8c9d0'
branch_labels = None
depends_on = None


# Settings definitions from app/services/settings_definitions.py
# EXCLUDED: llm.proxy.url (use config-service), admin.api_key (is_secret=True)
SETTINGS = [
    # LLM settings (only interface, NOT url)
    {
        "key": "llm.interface",
        "value": "JarvisAdapterModel",
        "value_type": "string",
        "category": "llm",
        "description": "LLM interface type for command processing",
        "env_fallback": "JARVIS_MODEL_INTERFACE",
        "requires_reload": False,
        "is_secret": False,
    },
    # Tool classifier settings
    {
        "key": "tool_classifier.enabled",
        "value": "true",
        "value_type": "bool",
        "category": "tool_classifier",
        "description": "Enable/disable the tool classifier for pre-filtering",
        "env_fallback": "JARVIS_TOOL_CLASSIFIER_ENABLED",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "tool_classifier.min_confidence",
        "value": "0.6",
        "value_type": "float",
        "category": "tool_classifier",
        "description": "Minimum confidence threshold for tool classification",
        "env_fallback": "JARVIS_TOOL_CLASSIFIER_MIN_CONFIDENCE",
        "requires_reload": False,
        "is_secret": False,
    },
    # Tool router settings
    {
        "key": "tool_router.filter_min_confidence",
        "value": "0.85",
        "value_type": "float",
        "category": "tool_router",
        "description": "Minimum confidence for tool router filtering",
        "env_fallback": "JARVIS_TOOL_ROUTER_FILTER_MIN_CONFIDENCE",
        "requires_reload": False,
        "is_secret": False,
    },
    # Transcription settings
    {
        "key": "transcription.cleanup_enabled",
        "value": "false",
        "value_type": "bool",
        "category": "transcription",
        "description": "Enable/disable transcription cleanup post-processing",
        "env_fallback": "JARVIS_TRANSCRIPTION_CLEANUP_ENABLED",
        "requires_reload": False,
        "is_secret": False,
    },
    # Prompt generation settings
    {
        "key": "prompt.include_antipatterns",
        "value": "true",
        "value_type": "bool",
        "category": "prompt",
        "description": "Include antipatterns in prompt generation",
        "env_fallback": "JARVIS_PROMPT_INCLUDE_ANTIPATTERNS",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "prompt.include_param_descriptions",
        "value": "true",
        "value_type": "bool",
        "category": "prompt",
        "description": "Include parameter descriptions in prompts",
        "env_fallback": "JARVIS_PROMPT_INCLUDE_PARAM_DESCRIPTIONS",
        "requires_reload": False,
        "is_secret": False,
    },
    # Model mode settings
    {
        "key": "model.small_model_mode",
        "value": "true",
        "value_type": "bool",
        "category": "model",
        "description": "Optimize prompts for smaller models",
        "env_fallback": "JARVIS_SMALL_MODEL_MODE",
        "requires_reload": False,
        "is_secret": False,
    },
    # Conversation settings
    {
        "key": "conversation.max_turns",
        "value": "20",
        "value_type": "int",
        "category": "conversation",
        "description": "Maximum number of conversation turns",
        "env_fallback": "JARVIS_CONVERSATION_MAX_TURNS",
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "conversation.cache_ttl_seconds",
        "value": "3600",
        "value_type": "int",
        "category": "conversation",
        "description": "Conversation cache TTL in seconds",
        "env_fallback": "JARVIS_CONVERSATION_CACHE_TTL",
        "requires_reload": False,
        "is_secret": False,
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == 'postgresql'

    for setting in SETTINGS:
        if is_postgres:
            conn.execute(
                sa.text("""
                    INSERT INTO settings (key, value, value_type, category, description,
                                         env_fallback, requires_reload, is_secret,
                                         household_id, node_id, user_id, created_at, updated_at)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL, NOW(), NOW())
                    ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
                """),
                setting
            )
        else:
            conn.execute(
                sa.text("""
                    INSERT OR IGNORE INTO settings (key, value, value_type, category, description,
                                                   env_fallback, requires_reload, is_secret,
                                                   household_id, node_id, user_id, created_at, updated_at)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL, datetime('now'), datetime('now'))
                """),
                setting
            )


def downgrade() -> None:
    conn = op.get_bind()
    for setting in SETTINGS:
        conn.execute(
            sa.text("""
                DELETE FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
            """),
            {"key": setting["key"]}
        )
