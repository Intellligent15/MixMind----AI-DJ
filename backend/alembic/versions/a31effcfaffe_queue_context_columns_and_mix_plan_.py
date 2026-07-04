"""queue context columns and mix_plan energy_bias

Revision ID: a31effcfaffe
Revises: 65cfac6c314d
Create Date: 2026-07-03 12:59:38.051231

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a31effcfaffe'
down_revision: Union[str, None] = '65cfac6c314d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('queues', sa.Column('occasion', sa.String(), nullable=True))
    op.add_column('queues', sa.Column('vibe_note', sa.String(length=300), nullable=True))
    op.add_column('queues', sa.Column('arc_template', sa.String(), nullable=True))
    op.add_column(
        'queues',
        sa.Column('tease_hooks', sa.Boolean(), nullable=False,
                  server_default=sa.text('false')),
    )
    op.add_column('mix_plans', sa.Column('energy_bias', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('mix_plans', 'energy_bias')
    op.drop_column('queues', 'tease_hooks')
    op.drop_column('queues', 'arc_template')
    op.drop_column('queues', 'vibe_note')
    op.drop_column('queues', 'occasion')
