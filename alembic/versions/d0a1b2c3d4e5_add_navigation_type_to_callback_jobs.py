"""add navigation_type to callback_jobs

Revision ID: d0a1b2c3d4e5
Revises: c9y0z1a2b3c4
Create Date: 2026-06-01

Lets a command's interactive element pick how the mobile app renders the
callback result — push a screen onto the navigation stack (`stack`), show
a modal sheet (`popover`), or leave the current async behavior where a
fresh inbox item lands later (`new_notification`, default). Drives both
the mobile dispatch and CC's choice of whether to also post an inbox item
when the node's result lands.
"""

from alembic import op
import sqlalchemy as sa


revision = "d0a1b2c3d4e5"
down_revision = "c9y0z1a2b3c4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "callback_jobs",
        sa.Column(
            "navigation_type",
            sa.String(32),
            nullable=False,
            server_default="new_notification",
        ),
    )


def downgrade() -> None:
    op.drop_column("callback_jobs", "navigation_type")
