"""Whole-song pitch pre-shifting for pair renders.

Under `pitch_mode = "whole_song"` every pair render must see a song's
audio ALREADY shifted by its resolved offset — that way the two renders
that share a song (A→B and B→C both contain B) agree sample-for-sample
and the stitch junction between them is invisible. Pitch shifting
preserves duration, so none of the stitcher's sample math changes.

Shifting a full song's five files (4 stems + master) through rubberband
is expensive, and the same (song, offset) pair is needed by two renders
— so results are cached in storage under

    stems_pitched/{song_id}/{offset:+d}/{vocals|drums|bass|other|original}.wav

and reused. The cache key includes the offset, so a re-resolved queue
(different neighbors → different offset) computes a fresh variant
without invalidating the old one.

Worker-side module: imports audio libs, does storage I/O. Not for the
pure-unit-test surface.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyrubberband.pyrb as pyrb
import soundfile as sf

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
STEM_NAMES = ("vocals", "drums", "bass", "other")


def _cache_key(song_id: str, offset: int, name: str) -> str:
    return f"stems_pitched/{song_id}/{offset:+d}/{name}.wav"


def _shift_file(path: str, semitones: int) -> None:
    """Pitch-shift a WAV in place, preserving duration and channel layout."""
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    shifted = pyrb.pitch_shift(audio, sr, semitones).astype(np.float32)
    # rubberband can drift by a handful of samples; pin the length so
    # downstream sample math stays exact.
    if shifted.shape[0] > audio.shape[0]:
        shifted = shifted[: audio.shape[0]]
    elif shifted.shape[0] < audio.shape[0]:
        pad = np.zeros(
            (audio.shape[0] - shifted.shape[0], shifted.shape[1]),
            dtype=np.float32,
        )
        shifted = np.vstack([shifted, pad])
    sf.write(path, shifted, sr)


async def ensure_pitched_inputs(
    storage,
    song_id: str,
    offset: int,
    stem_paths: dict[str, str],
    original_path: str | None,
) -> None:
    """Make the local files at `stem_paths`/`original_path` reflect the
    whole-song offset, via the storage cache when possible.

    No-op when offset == 0. Mutates the files in place (they're the
    worker's tempdir copies, downloaded fresh per render).
    """
    if offset == 0:
        return

    targets: list[tuple[str, str]] = [
        (name, stem_paths[name]) for name in STEM_NAMES if name in stem_paths
    ]
    if original_path:
        targets.append(("original", original_path))

    missing: list[tuple[str, str]] = []
    for name, local in targets:
        key = _cache_key(song_id, offset, name)
        if await storage.exists(key):
            await storage.download_file(key, Path(local))
        else:
            missing.append((name, local))

    if not missing:
        logger.info(
            "preshift: song %s offset %+d fully served from cache",
            song_id, offset,
        )
        return

    logger.info(
        "preshift: shifting song %s by %+d semitone(s) (%d file(s))",
        song_id, offset, len(missing),
    )
    for name, local in missing:
        _shift_file(local, offset)
        try:
            data = Path(local).read_bytes()
            await storage.write(_cache_key(song_id, offset, name), data)
        except Exception as exc:  # cache write failure must not kill the render
            logger.warning(
                "preshift: cache write failed for %s/%s: %s", song_id, name, exc
            )
