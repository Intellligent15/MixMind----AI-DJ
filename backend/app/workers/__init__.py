from celery import Celery

from app.core.config import settings

# Priority ladder on Redis (0 lowest, 9 highest). Downstream stages
# preempt upstream stages of OTHER songs, so the first songs in a
# 12-song queue race ahead to "ready" while later songs are still
# downloading. Steps of 3 leave headroom to insert a stage between
# two existing ones without renumbering everything.
PRI_DOWNLOAD = 0
PRI_ANALYZE = 3
PRI_SEPARATE = 6
PRI_TRANSCRIBE = 9
PRI_TAG = 6

celery_app = Celery(
    "ai_dj",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.ping",
        "app.workers.download",
        "app.workers.analyze",
        "app.workers.separate",
        "app.workers.transcribe",
        "app.workers.tagging",
        "app.workers.plan_set",
        "app.workers.render_transition",
        "app.workers.stitch_queue",
        "app.workers.fetch_lyrics",
        "app.workers.align_lyrics",
        "app.workers.evict_cache",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Redis priority support requires both: prefetch=1 stops the worker
    # from greedily grabbing 4 low-priority tasks while a high-priority
    # one waits; acks_late means a crashed task is retried on another
    # worker (or after restart) rather than silently lost.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    broker_transport_options={
        "queue_order_strategy": "priority",
        "priority_steps": list(range(10)),
    },
)
