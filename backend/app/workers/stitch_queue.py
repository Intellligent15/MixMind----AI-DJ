import asyncio
import json
import logging
import uuid
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf

from sqlalchemy import select, update

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import (
    Analysis,
    MixPlan,
    MixPlanStatus,
    Queue,
    QueueItem,
    QueueRender,
    QueueRenderStatus,
    Song,
    Stems,
    Transcription,
)
from app.models.lyrics import Lyrics, LyricsAlignmentStatus
from app.services.host.overlay import apply_host_overlay
from app.services.vocal_safety.safety import vocal_safe_regions
from app.services.mixer.tempo_map import (
    a_output_sample,
    a_ramp_from_plan,
    map_sample,
    ramp_time_map,
)
from app.services.storage import get_storage
from app.workers import celery_app

logger = logging.getLogger(__name__)

CLAIMABLE_STATUSES = (
    QueueRenderStatus.pending,
    QueueRenderStatus.failed,
)


def _get_mix0_sample(
    plan_json: list[dict],
    rate_A: float,
    T_orig: float,
    sr: int = 44100,
    a_bpm: float | None = None,
) -> int:
    """Map a middle-song time (B of render0) to a sample in render0's
    output. Shares tempo-map math with the executor so junctions can't
    drift from where the audio actually sits: B pre-ramp plays at rate_A,
    glides to native across the B-side ramp; A's seam is mapped through
    the A-side meet-in-the-middle ramp when the plan carries one."""
    window = next(c for c in plan_json if c["tool"] == "set_transition_window")
    b_ramp = next(
        (c for c in plan_json
         if c["tool"] == "set_tempo_ramp" and c.get("song") != "A"),
        None,
    )

    a_seam_orig = window["from_song_time_start"]
    b_seam_orig = window["to_song_time_start"]

    a_seam_samp = a_output_sample(plan_json, a_bpm, float(a_seam_orig), sr)
    b_seam_samp_post = int(b_seam_orig * sr / rate_A)

    if b_ramp:
        start = int(float(b_ramp["start_time"]) * sr)
        end = int(float(b_ramp["end_time"]) * sr)
        total = max(end, int(T_orig * sr)) + sr
        pairs = ramp_time_map(total, start, end, rate_A, 1.0)
        stretched_B_sample = map_sample(pairs, T_orig * sr)
    else:
        stretched_B_sample = int(T_orig * sr / rate_A)

    return stretched_B_sample - b_seam_samp_post + a_seam_samp


def _get_mix1_sample(T_orig: float, sr: int = 44100) -> int:
    return int(T_orig * sr)


def _snap_downbeat(t: float, downbeats: list[float]) -> float:
    """First downbeat at/after `t` (mirrors the executor's seam snap)."""
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _describe_transition(plan: list[dict]) -> dict:
    """Summarise a per-pair plan_json for the player's transition indicator:
    a human label, the per-stem A→B routing, and any layered effects."""
    tools = [c.get("tool") for c in plan]
    stems = [
        {"stem": c.get("stem"), "from": c.get("from_song"), "to": c.get("to_song")}
        for c in plan
        if c.get("tool") == "crossfade_stem"
    ]
    effects: list[str] = []
    if "filter_sweep" in tools:
        effects.append("filter sweep")
    if "echo_out" in tools:
        effects.append("echo tail")
    if "loop_section" in tools:
        effects.append("loop & build")
    if "swap_stem" in tools:
        effects.append("stem swap")
    if "temporary_pitch_shift" in tools:
        effects.append("key lift")
    if "set_tempo_ramp" in tools:
        effects.append("tempo ramp")
    if "apply_reverb" in tools:
        effects.append("reverb wash")
    if "turntable_stop" in tools:
        effects.append("vinyl stop")
    if "volume_fade" in tools:
        effects.append("volume fade")

    if "turntable_stop" in tools:
        label = "Vinyl stop"
    elif "swap_stem" in tools:
        label = "Stem swap"
    elif "apply_reverb" in tools:
        label = "Reverb wash"
    elif "filter_sweep" in tools:
        label = "Filter sweep"
    elif "echo_out" in tools:
        label = "Echo tail-out"
    elif "loop_section" in tools:
        label = "Loop & build"
    else:
        label = "Crossfade"

    reasoning = next(
        (c.get("text") for c in plan if c.get("tool") == "set_reasoning"), None
    )
    return {"label": label, "stems": stems, "effects": effects, "reasoning": reasoning}


def _build_timeline(
    song_ids: list,
    song_meta: dict,
    mix_plans: list,
    analyses: dict,
    render_body_start: list[int],
    head_full_index: list[int],
    total_samples: int,
    sr: int,
) -> dict:
    """Map the stitched output back to per-song + per-transition time spans.

    Uses the SAME accumulated offsets the audio loop produced
    (`render_body_start` / `head_full_index`), so the timeline can't drift
    from where the audio actually sits. Each render r is the transition from
    song r to song r+1; its seam (= A's downbeat-snapped seam, in render r's
    own samples) maps to output sample
    ``render_body_start[r] + (a_seam_sample_r - head_full_index[r])``.
    """
    n = len(song_ids)
    transitions: list[dict] = []
    for r in range(len(mix_plans)):
        plan = mix_plans[r].plan_json or []
        window = next(
            (c for c in plan if c.get("tool") == "set_transition_window"), None
        )
        an_a = analyses[song_ids[r]]
        if window is None or not an_a.bpm:
            continue
        a_seam_orig = _snap_downbeat(
            float(window["from_song_time_start"]), list(an_a.downbeats or [])
        )
        # Through the A-side meet ramp when present (identity otherwise).
        a_seam_sample = a_output_sample(plan, an_a.bpm, a_seam_orig, sr)
        seam_out = render_body_start[r] + (a_seam_sample - head_full_index[r])
        a_ramp = a_ramp_from_plan(plan)
        window_bpm = (
            float(a_ramp.get("end_bpm") or an_a.bpm) if a_ramp else an_a.bpm
        )
        sec_per_bar_a = (60.0 / window_bpm) * an_a.time_signature
        trans_len = int(round(int(window.get("duration_bars", 0)) * sec_per_bar_a * sr))
        seam_out = max(0, min(seam_out, total_samples))
        end_out = max(seam_out, min(seam_out + trans_len, total_samples))
        desc = _describe_transition(plan)
        transitions.append(
            {
                "index": r,
                "from_song_id": str(song_ids[r]),
                "to_song_id": str(song_ids[r + 1]),
                "start": round(seam_out / sr, 3),
                "end": round(end_out / sr, 3),
                "label": desc["label"],
                "stems": desc["stems"],
                "effects": desc["effects"],
                "reasoning": desc["reasoning"],
            }
        )

    # Per-song spans: song k owns output from the previous transition's end
    # to its own outgoing transition's seam. First/last songs bookend.
    total_sec = round(total_samples / sr, 3)
    seam_by_idx = {t["index"]: t for t in transitions}
    songs: list[dict] = []
    for k in range(n):
        prev_t = seam_by_idx.get(k - 1)
        cur_t = seam_by_idx.get(k)
        start = prev_t["end"] if prev_t else 0.0
        end = cur_t["start"] if cur_t else total_sec
        end = max(start, min(end, total_sec))
        meta = song_meta.get(song_ids[k], {})
        songs.append(
            {
                "index": k,
                "song_id": str(song_ids[k]),
                "title": meta.get("title"),
                "artist": meta.get("artist"),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    return {"duration": total_sec, "songs": songs, "transitions": transitions}


@celery_app.task(name="app.workers.stitch_queue.stitch_queue")
def stitch_queue(queue_id: str) -> str | None:
    queue_uuid = uuid.UUID(queue_id)
    storage = get_storage()

    with SessionLocal() as db:
        render_row = db.scalar(select(QueueRender).where(QueueRender.queue_id == queue_uuid))
        if render_row is None:
            logger.warning("stitch_queue: no QueueRender found for %s", queue_id)
            return None

        claim = db.execute(
            update(QueueRender)
            .where(QueueRender.id == render_row.id)
            .where(QueueRender.status.in_(CLAIMABLE_STATUSES))
            .values(status=QueueRenderStatus.rendering, error_text=None)
        )
        db.commit()
        if claim.rowcount == 0:
            db.refresh(render_row)
            logger.info("stitch_queue: %s already %s, skipping", queue_id, render_row.status.value)
            return None

        render_row_id = render_row.id

        # Fetch all queue items
        items = db.scalars(
            select(QueueItem)
            .where(QueueItem.queue_id == queue_uuid)
            .order_by(QueueItem.position)
        ).all()
        
        if len(items) < 2:
            _mark_failed(render_row_id, "Queue must have at least 2 songs")
            return None

        # Fetch MixPlans
        mix_plans = []
        for i in range(len(items) - 1):
            mp = db.scalar(
                select(MixPlan)
                .where(MixPlan.queue_id == queue_uuid)
                .where(MixPlan.from_song_id == items[i].song_id)
                .where(MixPlan.to_song_id == items[i+1].song_id)
            )
            if not mp or mp.status != MixPlanStatus.ready or not mp.rendered_audio_path:
                _mark_failed(render_row_id, f"MixPlan for pair {i} not ready")
                return None
            mix_plans.append(mp)

        # Fetch Analyses for BPMs
        analyses = {}
        for item in items:
            an = db.scalar(select(Analysis).where(Analysis.song_id == item.song_id))
            if not an:
                _mark_failed(render_row_id, f"Analysis missing for song {item.song_id}")
                return None
            analyses[item.song_id] = an

        # Phase 10: capture per-song display metadata for the player timeline
        # (title/artist) while the session is open — the loop below runs after
        # the session closes and can't lazy-load relationships.
        song_ids = [item.song_id for item in items]
        song_meta = {}
        for sid in song_ids:
            s = db.get(Song, sid)
            if s is not None:
                song_meta[sid] = {"title": s.title, "artist": s.artist}

        # F11 host snapshot: queue-level voice settings plus song 1's
        # vocal-safety inputs (the intro clip is placed in its first
        # vocal-free span). All optional; the host degrades to silence.
        queue_row = db.get(Queue, queue_uuid)
        host_ctx = {
            "occasion": queue_row.occasion if queue_row else None,
            "vibe_note": queue_row.vibe_note if queue_row else None,
            "frequency": queue_row.host_frequency if queue_row else None,
            "persona": queue_row.host_persona if queue_row else None,
        }
        host_enabled = settings.tts_provider != "off" and (
            (host_ctx["frequency"] or settings.host_frequency) != "off"
        )
        s1_segments = s1_aligned = s1_env_path = None
        s1_duration = 0.0
        if host_enabled:
            s1_id = items[0].song_id
            s1_song = db.get(Song, s1_id)
            s1_duration = (s1_song.duration_seconds or 0.0) if s1_song else 0.0
            s1_tr = db.scalar(
                select(Transcription).where(Transcription.song_id == s1_id)
            )
            s1_segments = s1_tr.segments if s1_tr else None
            s1_ly = db.scalar(select(Lyrics).where(Lyrics.song_id == s1_id))
            if s1_ly and s1_ly.alignment_status == LyricsAlignmentStatus.success:
                s1_aligned = s1_ly.aligned_words
            s1_st = db.scalar(select(Stems).where(Stems.song_id == s1_id))
            s1_env_path = s1_st.envelopes_path if s1_st else None

    # Now stitch!
    sr = 44100
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        
        # Download all WAVs
        wav_paths = []
        for i, mp in enumerate(mix_plans):
            dest = tmp / f"mix_{i}.wav"
            asyncio.run(storage.download_file(mp.rendered_audio_path, dest))
            wav_paths.append(dest)

        # Load all audio into memory (each is ~40MB, total ~800MB for 20 songs, very safe)
        audios = []
        for p in wav_paths:
            y, _ = sf.read(str(p), dtype=np.float32)
            if y.ndim == 1:
                y = np.column_stack((y, y))
            audios.append(y)

        stitched = [audios[0]]
        # How many samples of the current `stitched[-1]` were trimmed off the
        # front of its source `audios[i]` in the previous iteration's xfade.
        # _get_mix0_sample returns an index into the full audios[i], so we
        # must subtract this offset when slicing the (already head-trimmed)
        # buffer. Without this, each iter past the first re-plays a chunk of
        # the middle song before the next junction.
        head_offset = 0

        # Phase 10 timeline accumulators. render_body_start[r] = output sample
        # where render r's kept body begins; head_full_index[r] = the index
        # into audios[r] at which that body starts (how much head was trimmed).
        render_body_start = [0]
        head_full_index = [0]

        for i in range(len(mix_plans) - 1):
            mix0 = stitched[-1] # The accumulated mix so far (or we just accumulate chunks)
            mix1 = audios[i+1]
            
            mp0 = mix_plans[i]
            mp1 = mix_plans[i+1]
            
            # The song in the middle is items[i+1].song_id
            mid_song_id = items[i+1].song_id
            an_a = analyses[items[i].song_id]
            an_b = analyses[mid_song_id]

            plan0 = mp0.plan_json
            plan1 = mp1.plan_json

            tempo_ramp = next(
                (c for c in plan0
                 if c["tool"] == "set_tempo_ramp" and c.get("song") != "A"),
                None,
            )
            window0 = next(c for c in plan0 if c["tool"] == "set_transition_window")
            window1 = next(c for c in plan1 if c["tool"] == "set_transition_window")

            # Half/double-time beatmatch (window tempo_ratio) and the
            # meet-in-the-middle A ramp both change B's stretch rate in
            # render0 — the junction math must mirror the executor's.
            ratio0 = float(window0.get("tempo_ratio") or 1.0)
            a_ramp0 = a_ramp_from_plan(plan0)
            crossfade_bpm0 = (
                float(a_ramp0.get("end_bpm") or an_a.bpm) if a_ramp0 else an_a.bpm
            )
            rate_A = (crossfade_bpm0 * ratio0) / an_b.bpm
            
            a_seam1 = window1["from_song_time_start"]
            
            if tempo_ramp:
                safe_start_T = tempo_ramp["end_time"]
            else:
                # Estimate crossfade end (one A-grid bar consumes ratio0
                # B-bars of original audio under a half-time beatmatch).
                b_seam0 = window0["to_song_time_start"]
                dur_bars = window0["duration_bars"]
                sec_per_bar_b = (60.0 / an_b.bpm) * an_b.time_signature
                safe_start_T = b_seam0 + dur_bars * sec_per_bar_b * ratio0

            safe_end_T = a_seam1
            T_orig = (safe_start_T + safe_end_T) / 2.0
            
            # Prevent overlap failure
            if T_orig > safe_end_T:
                T_orig = safe_end_T - 1.0
                
            S0 = _get_mix0_sample(plan0, rate_A, T_orig, sr, a_bpm=an_a.bpm)
            S1 = _get_mix1_sample(T_orig, sr)
            
            # To accumulate cleanly, we replace stitched[-1] with its sliced version
            mix0_sliced = mix0[: S0 - head_offset]
            mix1_sliced = mix1[S1:]

            # Record render r's kept length (before the xfade trims a few ms
            # off either end — negligible vs the indicator's resolution) so
            # the timeline knows where render r+1's body lands in the output.
            render_body_start.append(render_body_start[-1] + mix0_sliced.shape[0])
            head_full_index.append(S1)
            
            # 50ms crossfade
            xfade_samples = int(0.050 * sr)
            if mix0_sliced.shape[0] < xfade_samples or mix1_sliced.shape[0] < xfade_samples:
                xfade_samples = min(mix0_sliced.shape[0], mix1_sliced.shape[0])
                
            if xfade_samples > 0:
                t = np.linspace(0.0, 1.0, xfade_samples, endpoint=False, dtype=np.float32)
                gain0 = np.cos(t * (np.pi / 2.0))
                gain1 = np.sin(t * (np.pi / 2.0))

                xfade_region = (
                    gain0[:, None] * mix0_sliced[-xfade_samples:] +
                    gain1[:, None] * mix1_sliced[:xfade_samples]
                )
                mix0_sliced = mix0_sliced[:-xfade_samples]
                mix1_sliced = mix1_sliced[xfade_samples:]
                stitched[-1] = mix0_sliced
                stitched.append(xfade_region)
                stitched.append(mix1_sliced)
                head_offset = S1 + xfade_samples
            else:
                stitched[-1] = mix0_sliced
                stitched.append(mix1_sliced)
                head_offset = S1

        final_audio = np.concatenate(stitched)

        # Phase 10: build the player timeline from the accumulated offsets.
        # Best-effort — a timeline glitch must never fail an otherwise-good
        # mix render.
        try:
            timeline = _build_timeline(
                song_ids, song_meta, mix_plans, analyses,
                render_body_start, head_full_index, final_audio.shape[0], sr,
            )
        except Exception:
            logger.exception("stitch_queue: timeline build failed for %s", queue_id)
            timeline = None

        # F11: the host voice — set intro, occasional mic drops. Purely
        # additive and failure-tolerant; the mix ships voiceless on any
        # problem.
        if host_enabled:
            song1_regions = None
            if s1_segments and s1_env_path:
                try:
                    env = json.loads(
                        asyncio.run(storage.read(s1_env_path)).decode("utf-8")
                    )
                    song1_regions = vocal_safe_regions(
                        transcription_segments=s1_segments,
                        envelope=env,
                        aligned_words=s1_aligned,
                        duration_seconds=s1_duration,
                    )
                except Exception:
                    song1_regions = None
            host_events = apply_host_overlay(
                final_audio, sr, timeline, storage,
                songs=[song_meta.get(sid, {}) for sid in song_ids],
                occasion=host_ctx["occasion"],
                vibe_note=host_ctx["vibe_note"],
                queue_frequency=host_ctx["frequency"],
                queue_persona=host_ctx["persona"],
                song1_safe_regions=song1_regions,
            )
            if host_events and timeline is not None:
                timeline["host"] = host_events

        wav_dest = tmp / "final.wav"
        out_dest = tmp / "final.m4a"
        sf.write(str(wav_dest), final_audio, sr, format="WAV", subtype="PCM_16")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(wav_dest),
            "-c:a", "aac",
            "-b:a", "256k",
            "-movflags", "+faststart",
            str(out_dest)
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            # CalledProcessError's message omits stderr — log it or the
            # actual encoder failure is invisible.
            logger.error(
                "stitch_queue: ffmpeg AAC encode failed for %s: %s",
                queue_id, (exc.stderr or b"").decode(errors="replace").strip(),
            )
            raise

        with open(out_dest, "rb") as f:
            audio_bytes = f.read()

    key = f"queue_mixes/{queue_id}.m4a"
    asyncio.run(storage.write(key, audio_bytes))

    with SessionLocal() as db:
        row = db.get(QueueRender, render_row_id)
        if row:
            row.rendered_audio_path = key
            row.status = QueueRenderStatus.ready
            row.error_text = None
            row.timeline = timeline
            db.commit()

    # Stitching just wrote the largest single artifact (the whole-queue
    # M4A) — nudge the LRU evictor (no-op under budget). By name to stay
    # decoupled from the evictor module.
    try:
        celery_app.send_task("app.workers.evict_cache.enforce_cache_budget")
    except Exception:  # pragma: no cover - broker hiccup must not fail stitch
        logger.warning("stitch_queue: failed to dispatch cache eviction", exc_info=True)

    return queue_id


def _mark_failed(render_id: uuid.UUID, message: str) -> None:
    with SessionLocal() as db:
        row = db.get(QueueRender, render_id)
        if row is None:
            return
        row.status = QueueRenderStatus.failed
        row.error_text = message[:1000]
        db.commit()
