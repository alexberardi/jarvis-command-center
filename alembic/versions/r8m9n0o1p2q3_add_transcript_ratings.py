"""add rating columns to conversation_transcripts

Revision ID: r8m9n0o1p2q3
Revises: q7l8m9n0o1p2
Create Date: 2026-04-18

"""
from alembic import op
import sqlalchemy as sa

revision = "r8m9n0o1p2q3"
down_revision = "q7l8m9n0o1p2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "conversation_transcripts",
        sa.Column("user_rating", sa.Integer(), nullable=True),
    )
    op.add_column(
        "conversation_transcripts",
        sa.Column("rating_notes", sa.Text(), nullable=True),
    )
    op.add_column(
        "conversation_transcripts",
        sa.Column("rated_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("conversation_transcripts", "rated_at")
    op.drop_column("conversation_transcripts", "rating_notes")
    op.drop_column("conversation_transcripts", "user_rating")
