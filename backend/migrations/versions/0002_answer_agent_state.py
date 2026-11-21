"""Persist answer-agent recovery state separately from verified answers."""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("conversation_turns", sa.Column("agent_state",
        sa.JSON().with_variant(postgresql.JSONB(), "postgresql"), nullable=False, server_default="{}"))


def downgrade():
    op.drop_column("conversation_turns", "agent_state")
