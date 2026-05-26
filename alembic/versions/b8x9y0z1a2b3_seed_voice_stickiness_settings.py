"""seed voice.stickiness_* settings

Revision ID: b8x9y0z1a2b3
Revises: a7w8x9y0z1a2
Create Date: 2026-05-26

Seeds default values for the per-node speaker-stickiness knobs added
in app/services/settings_definitions.py. SettingDefinition only
declares metadata; without a row in the settings table the admin UI
shows nothing to edit. This migration inserts the defaults so they
appear in the voice category immediately after deploy.
"""

from alembic import op
import sqlalchemy as sa


revision = "b8x9y0z1a2b3"
down_revision = "a7w8x9y0z1a2"
branch_labels = None
depends_on = None


SETTINGS = [
    {
        "key": "voice.stickiness_min_confidence",
        "value": "0.55",
        "value_type": "float",
        "category": "voice",
        "description": (
            "Minimum speaker-recognition confidence to record a node's "
            "speaker for short follow-up inheritance. Tune to match the "
            "encoder's score distribution (ECAPA: ~0.45-0.70 same-speaker; "
            "resemblyzer: ~0.75-0.90)."
        ),
        "env_fallback": None,
        "requires_reload": False,
        "is_secret": False,
    },
    {
        "key": "voice.stickiness_ttl_seconds",
        "value": "30.0",
        "value_type": "float",
        "category": "voice",
        "description": (
            "Seconds an identified speaker stays sticky for a node. "
            "After this, a fresh identification is required."
        ),
        "env_fallback": None,
        "requires_reload": False,
        "is_secret": False,
    },
]


def upgrade() -> None:
    conn = op.get_bind()
    is_postgres = conn.dialect.name == "postgresql"

    for setting in SETTINGS:
        if is_postgres:
            conn.execute(
                sa.text(
                    """
                    INSERT INTO settings (key, value, value_type, category, description,
                                         env_fallback, requires_reload, is_secret,
                                         household_id, node_id, user_id, created_at, updated_at)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL, NOW(), NOW())
                    ON CONFLICT (key, household_id, node_id, user_id) DO NOTHING
                    """
                ),
                setting,
            )
        else:
            conn.execute(
                sa.text(
                    """
                    INSERT OR IGNORE INTO settings (key, value, value_type, category, description,
                                                   env_fallback, requires_reload, is_secret,
                                                   household_id, node_id, user_id, created_at, updated_at)
                    VALUES (:key, :value, :value_type, :category, :description,
                           :env_fallback, :requires_reload, :is_secret,
                           NULL, NULL, NULL, datetime('now'), datetime('now'))
                    """
                ),
                setting,
            )


def downgrade() -> None:
    conn = op.get_bind()
    for setting in SETTINGS:
        conn.execute(
            sa.text(
                """
                DELETE FROM settings
                WHERE key = :key
                  AND household_id IS NULL
                  AND node_id IS NULL
                  AND user_id IS NULL
                """
            ),
            {"key": setting["key"]},
        )
