"""Transition eval harness.

Renders the SAME song pair through (a) the deterministic v1 planner and
(b) planner v2 (LLM decision → archetype expansion), writes both WAVs
side by side, and prints objective seam metrics so plan changes can be
compared with numbers instead of vibes:

* low-band onset correlation at the seam (the executor's own
  pair-phase metric — higher = kick grids locked);
* loudness continuity: RMS over the 2 bars before vs after the seam
  (closer to 0 dB delta = no energy pothole);
* peak / clipping stats.

Usage (inside the backend container / venv, songs already processed):

    python -m scripts.eval_transitions --from <song_uuid> --to <song_uuid>
    python -m scripts.eval_transitions --queue <queue_uuid>   # every pair
    python -m scripts.eval_transitions --from ... --to ... --style drop_swap

Outputs land in `<local_storage_path>/eval/`.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import tempfile
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from sqlalchemy import select

from app.core.config import settings
from app.core.db import SessionLocal
from app.models import Analysis, Queue, Song, Stems, Transcription
from app.models.lyrics import Lyrics, LyricsAlignmentStatus
from app.services.llm import get_llm_provider
from app.services.mixer.executor import (
    REQUIRED_SAMPLE_RATE,
    _low_band_onset,
    _norm_corr,
    render,
)
from app.services.mixer.pitch_resolver import SongKey, resolve_pitch_offsets
from app.services.mixer.plan import build_pair_plan
from app.services.mixer.qa import score_render
from app.services.mixer.planner_v2 import SongMeta, build_plan_v2
from app.services.mixer.preshift import ensure_pitched_inputs
from app.services.mixer.types import AnalysisBundle, SongRenderInputs
from app.services.mixer.validation import (
    enforce_revert_after_crossfade,
    strip_pitch_tools,
)
from app.services.storage import get_storage
from app.services.vocal_safety.safety import vocal_safe_regions
from app.workers.render_transition import _to_bundle


def _seam_metrics(
    wav_bytes: bytes, plan: list[dict], a_bundle: AnalysisBundle,
    style: str | None = None,
) -> dict:
    """QA report (single source of truth: services/mixer/qa.py) plus the
    eval-only low-band onset correlation across the seam."""
    report = score_render(wav_bytes, plan, a_bundle, style=style)

    audio, sr = sf.read(io.BytesIO(wav_bytes), always_2d=True, dtype="float32")
    seam_samp = int(float(report.metrics.get("seam_s", 0.0)) * sr)
    spb = (60.0 / a_bundle.bpm) * a_bundle.time_signature
    two_bars = int(2 * spb * sr)
    pre_env = _low_band_onset(audio[max(0, seam_samp - two_bars): seam_samp], sr)
    post_env = _low_band_onset(audio[seam_samp: seam_samp + two_bars], sr)
    onset_corr = (
        _norm_corr(pre_env, post_env)
        if pre_env is not None and post_env is not None else float("nan")
    )

    return {
        **report.metrics,
        "seam_lowband_onset_corr": round(float(onset_corr), 3),
        "peak": round(float(np.max(np.abs(audio))), 4),
        "qa_verdict": report.verdict,
        "qa_flags": report.flags,
    }


async def _load_song(db, song_id: uuid.UUID, storage, tmp: Path, prefix: str):
    song = db.get(Song, song_id)
    analysis = db.scalar(select(Analysis).where(Analysis.song_id == song_id))
    stems = db.scalar(select(Stems).where(Stems.song_id == song_id))
    if not (song and analysis and stems):
        raise SystemExit(f"song {song_id} not fully processed")
    bundle = _to_bundle(analysis, song.duration_seconds)

    paths = {}
    for k, key in (("vocals", stems.vocals_path), ("drums", stems.drums_path),
                   ("bass", stems.bass_path), ("other", stems.other_path)):
        dest = tmp / f"{prefix}_{k}.wav"
        await storage.download_file(key, dest)
        paths[k] = str(dest)
    orig = None
    if song.audio_path:
        dest = tmp / f"{prefix}_orig.wav"
        await storage.download_file(song.audio_path, dest)
        orig = str(dest)

    transcription = db.scalar(
        select(Transcription).where(Transcription.song_id == song_id)
    )
    lyrics = db.scalar(select(Lyrics).where(Lyrics.song_id == song_id))
    regions = []
    if transcription and stems.envelopes_path:
        try:
            env = json.loads(
                (await storage.read(stems.envelopes_path)).decode("utf-8")
            )
            aligned = (
                lyrics.aligned_words
                if lyrics and lyrics.alignment_status == LyricsAlignmentStatus.success
                else None
            )
            regions = vocal_safe_regions(
                transcription_segments=transcription.segments,
                envelope=env, aligned_words=aligned,
                duration_seconds=song.duration_seconds or 0.0,
            )
        except Exception:
            regions = []

    meta = SongMeta(
        title=song.title, artist=song.artist, bundle=bundle,
        energy_curve=list(analysis.energy_curve or []), safe_regions=regions,
    )
    inputs = SongRenderInputs(
        stem_paths=paths, analysis=bundle, original_audio_path=orig
    )
    return meta, inputs


async def _eval_pair(from_id: uuid.UUID, to_id: uuid.UUID,
                     style: str | None, nonce: int,
                     pitch_mode: str) -> None:
    storage = get_storage()
    out_dir = Path(settings.local_storage_path) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    with SessionLocal() as db, tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        a_meta, a_inputs = await _load_song(db, from_id, storage, tmp, "a")
        b_meta, b_inputs = await _load_song(db, to_id, storage, tmp, "b")

        pair_tag = f"{str(from_id)[:8]}__{str(to_id)[:8]}"
        print(f"\n=== {a_meta.title} → {b_meta.title} (pitch_mode={pitch_mode}) ===")

        # Whole-song mode: resolve the pair's offsets exactly like the
        # set-level pass would (A is the walk's first song → always 0),
        # pre-shift the local copies, and carry the offsets into planning.
        if pitch_mode == "whole_song":
            offsets = resolve_pitch_offsets([
                SongKey(a_meta.bundle.key, a_meta.bundle.camelot_key),
                SongKey(b_meta.bundle.key, b_meta.bundle.camelot_key),
            ])
            print(f"  whole-song offsets: A {offsets[0]:+d} st, B {offsets[1]:+d} st")
            await ensure_pitched_inputs(
                storage, str(from_id), offsets[0], a_inputs.stem_paths,
                a_inputs.original_audio_path,
            )
            await ensure_pitched_inputs(
                storage, str(to_id), offsets[1], b_inputs.stem_paths,
                b_inputs.original_audio_path,
            )
            from dataclasses import replace as _dc_replace
            a_meta = _dc_replace(a_meta, pitch_offset=offsets[0])
            b_meta = _dc_replace(b_meta, pitch_offset=offsets[1])

        # (a) deterministic v1
        det_plan = enforce_revert_after_crossfade(
            build_pair_plan(a_meta.bundle, b_meta.bundle), b_meta.bundle
        )
        det = render(det_plan, a_inputs, b_inputs)
        det_path = out_dir / f"{pair_tag}__deterministic.wav"
        det_path.write_bytes(det.wav_bytes)
        det_metrics = _seam_metrics(det.wav_bytes, det_plan, a_meta.bundle)
        print(f"[deterministic] {det_path.name}")
        print(json.dumps(det_metrics, indent=2))

        # (b) planner v2
        outcome = await build_plan_v2(
            get_llm_provider(), a_meta, b_meta,
            style_override=style, nonce=nonce, pitch_mode=pitch_mode,
        )
        v2_plan = outcome.plan
        if pitch_mode != "temporary":
            v2_plan = strip_pitch_tools(v2_plan)
        v2_plan = enforce_revert_after_crossfade(v2_plan, b_meta.bundle)
        v2 = render(v2_plan, a_inputs, b_inputs)
        v2_path = out_dir / f"{pair_tag}__v2_{outcome.style or 'fallback'}.wav"
        v2_path.write_bytes(v2.wav_bytes)
        v2_metrics = _seam_metrics(
            v2.wav_bytes, v2_plan, a_meta.bundle, style=outcome.style
        )
        print(f"[planner v2] source={outcome.source} style={outcome.style}")
        if outcome.rationale:
            print(f"  rationale: {outcome.rationale}")
        print(f"  {v2_path.name}")
        print(json.dumps(v2_metrics, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_id", type=uuid.UUID)
    parser.add_argument("--to", dest="to_id", type=uuid.UUID)
    parser.add_argument("--queue", dest="queue_id", type=uuid.UUID)
    parser.add_argument("--style", default=None,
                        help="pin a transition style for the v2 render")
    parser.add_argument("--nonce", type=int, default=0,
                        help="re-roll nonce (busts the LLM cache)")
    parser.add_argument("--pitch-mode", default=settings.pitch_mode,
                        choices=("whole_song", "temporary", "off"),
                        help="key-clash handling for the v2 render "
                             "(deterministic baseline always renders native)")
    args = parser.parse_args()

    pairs: list[tuple[uuid.UUID, uuid.UUID]] = []
    if args.queue_id:
        with SessionLocal() as db:
            queue = db.get(Queue, args.queue_id)
            if queue is None:
                raise SystemExit("queue not found")
            items = sorted(queue.items, key=lambda it: it.position)
            pairs = [
                (a.song_id, b.song_id) for a, b in zip(items[:-1], items[1:])
            ]
    elif args.from_id and args.to_id:
        pairs = [(args.from_id, args.to_id)]
    else:
        raise SystemExit("provide --from/--to or --queue")

    for f, t in pairs:
        asyncio.run(_eval_pair(f, t, args.style, args.nonce, args.pitch_mode))


if __name__ == "__main__":
    main()
