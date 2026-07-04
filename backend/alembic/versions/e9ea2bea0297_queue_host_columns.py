"""queue host columns

Revision ID: e9ea2bea0297
Revises: 42b991246c47
Create Date: 2026-07-03 21:26:28.169232

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e9ea2bea0297'
down_revision: Union[str, None] = '42b991246c47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('queues', sa.Column('host_frequency', sa.String(), nullable=True))
    op.add_column('queues', sa.Column('host_persona', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('queues', 'host_persona')
    op.drop_column('queues', 'host_frequency')
