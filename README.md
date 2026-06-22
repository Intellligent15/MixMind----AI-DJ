# MixMind

MixMind turns a queue of YouTube songs into a single, continuously beat‑mixed DJ set. For each pair of adjacent tracks it analyses the music (tempo, key, beat grid, song structure, vocals) and uses an LLM to design a musical transition — beatmatched crossfades, key‑safe blends, filter sweeps, vinyl stops, EQ kills — then renders the whole queue into one seamless audio file you can play in the browser.

It's a local‑first, single‑user web app: everything runs on your own machine.

> Full design details live in [mixmind-spec.md](mixmind-spec.md).

## What it does

1. **Search & queue** — search YouTube, add songs, drag to order, lock the queue.
2. **Analyse each track** — download audio, detect BPM / key / beat grid / sections / energy, split into 4 stems (vocals, drums, bass, other), transcribe the vocal, and align lyrics.
3. **Design transitions** — for every adjacent pair, an LLM plans a transition as a sequence of mixing "tool calls," guided by both songs' structure and the regions where vocals are/aren't present (so cuts never chop a word).
4. **Render & stitch** — a deterministic mixer interprets those tool calls (time‑stretch to beatmatch, downbeat‑aligned per‑stem crossfades, pitch handling, effects) and stitches all transitions and songs into one continuous FLAC.
5. **Play** — the player streams the continuous mix with a live indicator of the current track and the transition in progress, plus a per‑song hard‑cut fallback.

## How it works (high level)

```
YouTube search ─▶ Queue ─▶ per-song pipeline ─▶ per-pair LLM plan ─▶ render ─▶ stitch ─▶ Player
                            (download →            (tool-call list)    (numpy +    (one FLAC)
                             analyze →                                  rubberband)
                             stem-separate →
                             transcribe →
                             lyrics align)
```

- **Audio analysis** uses `librosa` for tempo / beats / downbeats / key / sections / energy. Optionally, installing the `allin1` extra (`pip install -e backend[pro-analysis]`) and setting `SECTION_DETECTOR=allin1` switches beats / downbeats / tempo / sections to the All‑In‑One neural model, which also produces *real* section labels (intro / verse / chorus / …) that flow into the planner; any failure falls back to librosa.
- **Stem separation** uses Demucs (`htdemucs`); **transcription** uses Whisper. Both are GPU‑heavy and run on a native worker (Apple Silicon MPS, or CUDA on Linux).
- **Vocal‑safe regions** are derived from the vocal stem's energy envelope cross‑referenced with Whisper + aligned Genius lyrics, so the planner only places hard cuts where there are no vocals.
- **Transition planning (planner v2)** goes through a pluggable `LLMProvider` (Gemini / Groq / OpenAI‑compatible), but the LLM no longer does any timestamp math. The pipeline pre‑computes a small menu of *valid‑by‑construction* seam candidates for each song (downbeat‑snapped, headroom‑checked, vocal‑safety‑annotated), and the model — seeing each song's **title and artist** plus its section/energy structure — picks an exit point, an entry point, and one of eight **transition archetypes** (`smooth_blend`, `drop_swap`, `drum_bridge`, `wash_out`, `stutter_buildup`, `vinyl_stop`, plus two acapella-layering styles: `acapella_out` rides A's vocals over B's incoming instrumental before handing off to B's vocals at a vocal-safe phrase boundary, and `acapella_in` teases B's vocals over A's still-playing backing before the beat swaps — both require compatible effective keys and are auto-downgraded to a blend on a clash unless the user pins them) with a couple of bounded knobs. A deterministic expander turns that decision into the exact tool‑call list (`crossfade_stem`, `set_tempo_ramp`, `temporary_pitch_shift`, `filter_sweep`, `apply_reverb`, `turntable_stop`, `volume_fade`, …), so the executor's invariants hold by construction. A **set‑level pass** runs once per locked queue and assigns each pair a suggested style so the whole mix has an arc. Malformed LLM output is *repaired* where mechanically possible instead of silently discarded, and every plan records its provenance (`plan_source`) — LLM, repaired, or fallback — surfaced in the UI. Each transition can be **re‑rolled** (with an optional pinned style) from the player; re‑rolls bump a nonce in the LLM cache key so they are genuinely fresh samples. The v1 free‑form prompt remains available via `PLANNER_VERSION=legacy`.
- **Whole‑song key handling (no more pitch glides).** Mid‑song key glides — holding the incoming track in the outgoing track's key, then bending it back after the crossfade — are the most audible artifact a mixer can produce, so by default (`PITCH_MODE=whole_song`) MixMind never does them. Instead, a deterministic set‑level resolver walks the locked queue once and assigns each song a constant semitone offset (capped at ±2, reset to 0 whenever neighboring keys are already compatible) applied to its **entire** play — the professional "Pitch 'n Time" approach; a constant ±1–2 semitone shift is essentially inaudible while a bend is always audible. Pair renders pre‑shift the song's stems and master (cached per song+offset in storage), so the two renders sharing a song agree exactly and stitch junctions stay seamless. Clashes beyond the cap are *accepted* and the planner routes them to short or washy transition styles where keys barely overlap. `PITCH_MODE=temporary` restores the v1 glide behavior; `PITCH_MODE=off` never shifts. Tempo is unaffected — beatmatching still uses a (now longer, gentler) post‑crossfade ramp, since tempo glides don't change key.
- **Gain staging.** Every transition includes automatic **seam loudness matching**: the incoming song's perceived level (from the analysis energy curve) is aligned to the outgoing one at the seam — boost capped at +4 dB, cut at −6 dB, gaps under ~1.25 dB untouched — then glides back to its native level after the crossfade, returning to exactly unity before the next stitch junction. Disable with `LOUDNESS_MATCH=false`. The acapella styles additionally bring the backing track in **staged**, the way a DJ would: drums crossfade first (the groove morphs), the bass quick‑swaps in one bar (two basslines overlapping is instant mud), and the melodic stem arrives **ducked** under the riding vocal, restoring to full only when the vocal hands over.
- **The mixer executor** turns that tool list into audio with `numpy` / `soundfile` / `pyrubberband`: it time‑stretches the incoming track to the outgoing track's tempo, aligns downbeats at the seam, and applies per‑stem crossfades and effects. The stitcher equal‑power crossfades the per‑pair renders into one file.
- **Storage** is behind a `StorageBackend` protocol (local filesystem by default; S3‑compatible swap is a config change). Generated artifacts live under `cache/` and are evicted LRU once they exceed a size budget.

## Tech stack

- **Backend:** FastAPI, Celery + Redis, PostgreSQL, SQLAlchemy + Alembic (Python 3.11)
- **Audio/ML:** librosa, Demucs, Whisper (mlx‑whisper on Apple Silicon), pyrubberband (`rubberband-cli`), numpy / scipy / soundfile, yt‑dlp
- **LLM:** pluggable provider (Gemini / Groq / DigitalOcean / OpenAI‑compatible)
- **Frontend:** Next.js 15 (App Router), React 19, TypeScript, Tailwind, TanStack Query, wavesurfer.js, dnd‑kit; Playwright for e2e
- **Infra:** Docker Compose (Postgres, Redis, backend, frontend); the Celery worker runs natively for GPU access

## Repository layout

```
backend/    FastAPI app, Celery workers, Alembic migrations
frontend/   Next.js app (App Router)
cache/      Generated audio, stems, analyses, mixes (gitignored)
```

## Prerequisites

```bash
brew install uv fnm docker ffmpeg rubberband
fnm install 20
```

- [`uv`](https://docs.astral.sh/uv/) manages Python 3.11 + deps; [`fnm`](https://github.com/Schniz/fnm) manages Node 20.
- `ffmpeg` is used by yt‑dlp for audio extraction; `rubberband` (rubberband‑cli) powers time‑stretch / pitch‑shift.
- Stem separation and transcription want a GPU: Apple Silicon (MPS) out of the box, or CUDA on Linux.

### Why a native worker?

On macOS, Docker containers can't see the Apple MPS GPU. So Postgres, Redis, the FastAPI backend, and the Next.js frontend run in Docker, while the **Celery worker runs natively on the host** (it needs MPS for Demucs/Whisper and `rubberband-cli` for time‑stretching). On a Linux/CUDA host the worker can move into Compose with no other changes.

## Configuration

Copy the example env and fill in the keys you want:

```bash
cp .env.example .env
```

- **LLM planning:** set `LLM_PROVIDER` and the matching key (`GEMINI_API_KEY`, `GROQ_API_KEY`, or `DO_INFERENCE_API_KEY`). Without one, set `USE_LLM_PLANNER=False` to use the deterministic planner.
- **Lyrics:** `GENIUS_ACCESS_TOKEN` enables Genius lyric fetch + alignment (optional; the pipeline degrades gracefully without it).
- **Storage / cache:** `STORAGE_BACKEND=local` and `LOCAL_STORAGE_PATH=./cache`; `CACHE_MAX_SIZE_GB` caps the on‑disk cache.

`.env` is gitignored — never commit real keys.

## Run it

From the repo root:

```bash
./start-dev.sh    # brings up the Docker stack, then runs the native worker in the foreground
```

Then open the app and check health:

```bash
open http://localhost:3000          # the app
curl localhost:8000/health          # -> {"status":"ok","db":"ok","redis":"ok"}
```

Stop everything:

```bash
# Ctrl‑C the worker, then:
./stop-dev.sh
docker compose down -v # To full wipe past data from container
```

## Using the app

1. On the home page, **search** for a song and add it to the queue. Add a few and **drag to reorder**.
2. Click **Done** to lock the queue. The **Processing** view shows each song moving through the pipeline and each transition rendering; failures surface inline with a **Retry**.
3. When the continuous mix is ready it advances to the **Player**, which streams the stitched set and shows the active transition. You can also download the mix as a FLAC, or switch to per‑song "Queue Mode."

## Tests

```bash
# backend (needs Postgres + Redis up: docker compose up -d postgres redis)
cd backend && uv run pytest -q

# frontend
cd frontend && npx tsc --noEmit        # type-check
cd frontend && npm run test:e2e        # Playwright (stubbed backend)
```

## Scope

Single‑user, local‑first, no authentication. Not intended for multi‑tenant or public deployment as‑is.
