"""add qa columns to mix_plans

Revision ID: 65cfac6c314d
Revises: 2090b91d849a
Create Date: 2026-07-03 00:47:03.752526

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '65cfac6c314d'
down_revision: Union[str, None] = '2090b91d849a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'mix_plans',
        sa.Column('qa_metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column('mix_plans', sa.Column('qa_verdict', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('mix_plans', 'qa_verdict')
    op.drop_column('mix_plans', 'qa_metrics')
