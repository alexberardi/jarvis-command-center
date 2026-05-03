"""allow null user_id on user_memories for household-wide agent context

Revision ID: y5u6v7w8x9y0
Revises: x4t5u6v7w8x9
Create Date: 2026-05-02

Background agents (calendar, news, weather) inject context that applies
to all speakers in a household, not a specific user.  A NULL user_id
means the memory is household-wide.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "y5u6v7w8x9y0"
down_revision = "x4t5u6v7w8x9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "user_memories", "user_id", existing_type=sa.Integer(), nullable=True
    )


def downgrade() -> None:
    # Assign a placeholder user_id to household-wide memories before
    # restoring NOT NULL.
    op.execute("UPDATE user_memories SET user_id = 0 WHERE user_id IS NULL")
    op.alter_column(
        "user_memories", "user_id", existing_type=sa.Integer(), nullable=False
    )
