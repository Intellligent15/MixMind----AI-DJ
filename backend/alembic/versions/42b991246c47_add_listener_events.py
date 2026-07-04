"""add listener_events

Revision ID: 42b991246c47
Revises: a31effcfaffe
Create Date: 2026-07-03 13:05:23.058295

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '42b991246c47'
down_revision: Union[str, None] = 'a31effcfaffe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'listener_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('queue_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('mix_plan_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            'kind',
            sa.Enum('thumbs_up', 'thumbs_down', 'skip', 'replay',
                    name='listener_event_kind'),
            nullable=False,
        ),
        sa.Column('style', sa.String(), nullable=True),
        sa.Column('from_genre', sa.String(), nullable=True),
        sa.Column('to_genre', sa.String(), nullable=True),
        sa.Column('position_seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True),
                  server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['queue_id'], ['queues.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['mix_plan_id'], ['mix_plans.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_listener_events_queue_id'), 'listener_events', ['queue_id']
    )


def downgrade() -> None:
    op.drop_index(op.f('ix_listener_events_queue_id'), table_name='listener_events')
    op.drop_table('listener_events')
    sa.Enum(name='listener_event_kind').drop(op.get_bind(), checkfirst=True)
