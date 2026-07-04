from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import feedback as feedback_api
from app.api import mix_plans as mix_plans_api
from app.api import queues as queues_api
from app.api import search as search_api
from app.api import songs as songs_api
from app.core.db import check_db
from app.core.redis_client import check_redis

app = FastAPI(title="MixMind Backend", version="0.0.0")

app.add_middleware(
    CORSMiddleware,
    # Local-dev frontend + the deployed droplet's public address. If we
    # ever grow more deploy targets, expose this via settings; for a
    # single droplet the static list is fine.
    allow_origins=[
        "http://localhost:3000",
        "http://137.184.211.233:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_api.router)
app.include_router(songs_api.router)
app.include_router(queues_api.router)
app.include_router(mix_plans_api.router)
app.include_router(feedback_api.router)


@app.get("/health")
def health() -> dict[str, str]:
    db_ok = check_db()
    redis_ok = check_redis()
    overall = "ok" if (db_ok and redis_ok) else "degraded"
    return {
        "status": overall,
        "db": "ok" if db_ok else "down",
        "redis": "ok" if redis_ok else "down",
    }
