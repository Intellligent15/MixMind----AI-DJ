#!/usr/bin/env bash
# MixMind — bring up the full dev stack.
#
# - Docker: postgres, redis, backend, frontend
# - Native (host): Celery worker (needs MPS access on macOS)
#
# The worker runs in the foreground; Ctrl-C stops it. Run ./stop-dev.sh to tear
# down the Docker stack afterwards.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# yt-dlp uses ffmpeg for the WAV extraction postprocessor on the host worker.
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH. Install with: brew install ffmpeg" >&2
  exit 1
fi

echo "==> Starting Docker services (postgres, redis, backend, frontend)..."
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build

echo "==> Waiting for postgres + redis to be healthy..."
for i in {1..30}; do
  pg_status=$(docker compose ps --format json postgres | grep -o '"Health":"[^"]*"' | head -1 || true)
  redis_status=$(docker compose ps --format json redis | grep -o '"Health":"[^"]*"' | head -1 || true)
  if [[ "$pg_status" == '"Health":"healthy"' && "$redis_status" == '"Health":"healthy"' ]]; then
    echo "    postgres + redis healthy."
    break
  fi
  sleep 1
done

cd backend

# Make the host worker share the same cache directory the dockerized backend
# mounts (./cache at the repo root). Without this, the worker's CWD changes
# to backend/ and a relative LOCAL_STORAGE_PATH would land in backend/cache/
# — invisible to the container that serves the audio.
export LOCAL_STORAGE_PATH="${LOCAL_STORAGE_PATH:-$REPO_ROOT/cache}"

# Demucs downloads the htdemucs_ft weights on first run via urllib. macOS
# Python doesn't trust the system root CAs by default, so point urllib +
# requests at certifi's bundle (which is already on the dep tree).
CERTIFI_PEM=$(uv run python -c "import certifi; print(certifi.where())" 2>/dev/null || true)
if [[ -n "$CERTIFI_PEM" ]]; then
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$CERTIFI_PEM}"
  export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-$CERTIFI_PEM}"
fi

echo "==> Applying database migrations..."
uv run alembic upgrade head

echo "==> Starting native Celery worker (Ctrl-C to stop)..."
echo "    Backend:  http://localhost:8000/health"
echo "    Frontend: http://localhost:3000"
echo "    Stop docker stack with: ./stop-dev.sh"
echo

# --group worker: pulls torch + demucs + torchaudio (the optional dep
# group defined in pyproject.toml). The dockerized backend skips this
# group entirely so its image stays small; only the native worker needs
# the ML stack.
#
# --pool=threads --concurrency=4: lets the worker run up to 4 tasks
# concurrently in OS threads. Safe now that heavy ML runs on Modal —
# the local worker mostly orchestrates I/O (yt-dlp download, S3
# transfers, blocking modal.Function.remote() calls), all of which
# release the GIL or are network-bound. We avoid --pool=prefork
# specifically because torch + MPS crashes (SIGABRT) under fork on
# macOS; threads sidestep that since they share the parent's memory.
# Bump --concurrency higher only if you want more songs in-flight than
# Modal's free tier wants to spin up at once (each parallel separation
# is its own GPU container).
exec uv run --group worker celery -A app.workers worker --loglevel=info --pool=threads --concurrency=8
