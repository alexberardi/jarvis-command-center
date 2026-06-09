"""add assistant_message column to request_traces

Revision ID: e1f2g3h4i5j6
Revises: d0a1b2c3d4e5
Create Date: 2026-06-09

Captures Jarvis's spoken reply alongside the user command on each trace
row so the admin trace list can show both halves of the exchange without
having to parse spans or look up ConversationTranscript per row.
"""

from alembic import op
import sqlalchemy as sa

revision = "e1f2g3h4i5j6"
down_revision = "d0a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "request_traces",
        sa.Column("assistant_message", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("request_traces", "assistant_message")
