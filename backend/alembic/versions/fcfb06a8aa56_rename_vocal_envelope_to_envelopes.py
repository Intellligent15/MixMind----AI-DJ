"""rename vocal_envelope to envelopes

Revision ID: fcfb06a8aa56
Revises: b7c8d9e0f1a2
Create Date: 2026-06-28 16:53:54.364060

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'fcfb06a8aa56'
down_revision: Union[str, None] = 'b7c8d9e0f1a2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column('stems', 'vocal_envelope_path', new_column_name='envelopes_path')


def downgrade() -> None:
    op.alter_column('stems', 'envelopes_path', new_column_name='vocal_envelope_path')
