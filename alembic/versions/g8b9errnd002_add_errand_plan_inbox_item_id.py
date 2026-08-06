"""errand_plans.inbox_item_id — the plan card's inbox item, for in-place updates

A drafted errand posts a plan card to the inbox; storing that card's inbox item
id lets a Revise/re-plan UPDATE the SAME card in place (PATCH) instead of posting
a duplicate.

Revision ID: g8b9errnd002
Revises: f7a8errnd001
Create Date: 2026-07-29 01:40:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "g8b9errnd002"
down_revision = "f7a8errnd001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("errand_plans", sa.Column("inbox_item_id", sa.String(64), nullable=True))


def downgrade() -> None:
    op.drop_column("errand_plans", "inbox_item_id")
