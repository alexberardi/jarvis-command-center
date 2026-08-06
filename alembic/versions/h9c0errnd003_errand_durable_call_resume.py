"""Durable wait-and-decide errand execution: link phone calls to errands + resume cursor

An errand that places a phone call must SUSPEND until that call finishes, then
decide whether to run the next step (fail-fast if the call failed). To do that:

- phone_call_sessions gains ``errand_id`` + ``errand_step`` so a finalizing call
  (done/failed/declined/expired) can find the errand waiting on it and resume it.
- errand_plans gains ``cursor`` (next step index to run on resume) + ``results_json``
  (the accumulated per-step outcomes, so the final completion card can summarize
  everything across suspends). ``state`` gains a new value ``waiting`` (no schema
  change — it's already a free String).

Revision ID: h9c0errnd003
Revises: g8b9errnd002
Create Date: 2026-07-30 00:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "h9c0errnd003"
down_revision = "g8b9errnd002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("phone_call_sessions", sa.Column("errand_id", sa.String(40), nullable=True))
    op.add_column("phone_call_sessions", sa.Column("errand_step", sa.Integer(), nullable=True))
    op.create_index(
        "ix_phone_call_sessions_errand_id", "phone_call_sessions", ["errand_id"]
    )
    op.add_column(
        "errand_plans", sa.Column("cursor", sa.Integer(), nullable=False, server_default="0")
    )
    op.add_column(
        "errand_plans", sa.Column("results_json", sa.Text(), nullable=False, server_default="[]")
    )


def downgrade() -> None:
    op.drop_column("errand_plans", "results_json")
    op.drop_column("errand_plans", "cursor")
    op.drop_index("ix_phone_call_sessions_errand_id", table_name="phone_call_sessions")
    op.drop_column("phone_call_sessions", "errand_step")
    op.drop_column("phone_call_sessions", "errand_id")
