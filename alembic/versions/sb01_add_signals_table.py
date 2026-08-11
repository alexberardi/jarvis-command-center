"""add signals table for the Signal Bus

Revision ID: sb01signals
Revises: pa02suppressions
Create Date: 2026-08-10

The generic world-state store: one self-describing Signal per (household,
source_key), TTL-swept. Deliberately NOT user_memories — situational/location
facts must never reach the recall tool or the embedding sweep.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "sb01signals"
down_revision = "pa02suppressions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "signals",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("household_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("node_id", sa.String(255), nullable=True),
        sa.Column("room", sa.String(255), nullable=True),
        sa.Column("kind", sa.String(255), nullable=False),
        sa.Column("subject", sa.String(255), nullable=True),
        sa.Column("source_key", sa.String(512), nullable=False),
        sa.Column("summary", sa.Text, nullable=True),
        sa.Column("facts", sa.Text, nullable=True),
        sa.Column("source_agent", sa.String(255), nullable=True),
        sa.Column("cacheable", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("salience", sa.Float, nullable=True),
        sa.Column("observed_at", sa.DateTime, nullable=True),
        sa.Column("expires_at", sa.DateTime, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("household_id", "source_key", name="uq_signals_household_source"),
    )
    op.create_index("ix_signals_household_id", "signals", ["household_id"])
    op.create_index("ix_signals_kind", "signals", ["kind"])
    op.create_index("ix_signals_user_id", "signals", ["user_id"])
    op.create_index("ix_signals_expires_at", "signals", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_signals_expires_at", table_name="signals")
    op.drop_index("ix_signals_user_id", table_name="signals")
    op.drop_index("ix_signals_kind", table_name="signals")
    op.drop_index("ix_signals_household_id", table_name="signals")
    op.drop_table("signals")
