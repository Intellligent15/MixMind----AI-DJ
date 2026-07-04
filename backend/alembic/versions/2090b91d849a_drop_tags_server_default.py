"""drop tags server default

Revision ID: 2090b91d849a
Revises: 5841e9b70b62
Create Date: 2026-07-02 00:43:00.679217

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '2090b91d849a'
down_revision: Union[str, None] = '5841e9b70b62'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The '{}' server default (left over from the NOT NULL introduction
    # of `tags`) made every new Analysis row read as "already tagged" —
    # NULL is the real "not tagged yet" marker.
    op.alter_column(
        'analyses', 'tags',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        server_default=None,
        existing_nullable=True,
    )
    op.execute("UPDATE analyses SET tags = NULL WHERE tags = '{}'::jsonb")


def downgrade() -> None:
    op.alter_column(
        'analyses', 'tags',
        existing_type=postgresql.JSONB(astext_type=sa.Text()),
        server_default=sa.text("'{}'::jsonb"),
        existing_nullable=True,
    )
