"""Split model.advanced_thinking into model.advanced_context + model.include_thinking

The old ``model.advanced_thinking`` flag coupled two unrelated things: proactive
context injection (weather/calendar/news/reminder agent memories) AND chain-of-thought
``<think>`` generation. Thinking is slow (~500 tokens / several seconds), so households
that just want proactive context were forced to eat the latency. This decouples them:

  * ``model.advanced_context``  — accept + inject the agents' context (default OFF).
  * ``model.include_thinking``  — allow /think chain-of-thought on Qwen3 (default OFF).

Both default OFF (matches the old default). The deprecated key is deleted.

Revision ID: k3g4ctxthink007
Revises: j2f3sched006
"""
from alembic import op

revision = "k3g4ctxthink007"
down_revision = "j2f3sched006"
branch_labels = None
depends_on = None

_SEED = (
    "INSERT INTO settings (key, value, value_type, category, created_at, updated_at) "
    "SELECT :k, 'false', 'bool', 'model', NOW(), NOW() "
    "WHERE NOT EXISTS (SELECT 1 FROM settings WHERE key = :k "
    "AND household_id IS NULL AND node_id IS NULL AND user_id IS NULL)"
)


def upgrade() -> None:
    # Drop the deprecated coupled setting (every scope).
    op.execute("DELETE FROM settings WHERE key = 'model.advanced_thinking'")
    # Seed the two decoupled settings as system defaults (both OFF), idempotently.
    op.execute(_SEED.replace(":k", "'model.advanced_context'"))
    op.execute(_SEED.replace(":k", "'model.include_thinking'"))


def downgrade() -> None:
    op.execute(
        "DELETE FROM settings WHERE key IN ('model.advanced_context', 'model.include_thinking')"
    )
    op.execute(_SEED.replace(":k", "'model.advanced_thinking'"))
