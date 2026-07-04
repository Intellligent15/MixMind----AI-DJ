"""Host overlay orchestration: script → TTS → placement.

Called by ``stitch_queue`` right before the ffmpeg encode. Every step is
failure-tolerant: no script, no clips, or no fitting slots all mean the
mix ships exactly as it would have without a host.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from app.core.config import settings
from app.services.host.mixdown import place_clips
from app.services.host.script import generate_script
from app.services.host.tts import synthesize_cached
from app.services.mixer.occasions import OCCASIONS

logger = logging.getLogger(__name__)

# Fallback intro start when song 1 has no usable vocal-safety data.
DEFAULT_INTRO_START_S = 2.0
INTRO_PAD_S = 1.0


def resolve_host_settings(
    queue_frequency: str | None,
    queue_persona: str | None,
    occasion: str | None,
) -> tuple[str, str]:
    """(frequency, persona) after defaults: queue setting → occasion's
    persona → global config."""
    frequency = queue_frequency or settings.host_frequency
    persona = queue_persona or (
        OCCASIONS[occasion].host_persona if occasion in OCCASIONS else "radio"
    )
    return frequency, persona


def pick_intro_start(
    safe_regions: list[dict] | None, clip_seconds: float
) -> float:
    """Earliest vocal-safe span in song 1 that fits the intro clip;
    DEFAULT_INTRO_START_S when safety data is missing or nothing fits."""
    if not safe_regions:
        return DEFAULT_INTRO_START_S
    need = clip_seconds + INTRO_PAD_S
    for region in safe_regions:
        if "safe" in region and not region.get("safe"):
            continue
        lo = max(float(region["start"]), 0.5)
        hi = float(region["end"])
        if hi - lo >= need:
            return round(lo, 3)
    return DEFAULT_INTRO_START_S


def apply_host_overlay(
    final_audio: np.ndarray,
    sr: int,
    timeline: dict | None,
    storage,
    *,
    songs: list[dict],
    occasion: str | None,
    vibe_note: str | None,
    queue_frequency: str | None,
    queue_persona: str | None,
    song1_safe_regions: list[dict] | None,
) -> list[dict]:
    """Mutates ``final_audio`` in place; returns the placed host events
    (possibly empty). Never raises."""
    try:
        if settings.tts_provider == "off":
            return []
        frequency, persona = resolve_host_settings(
            queue_frequency, queue_persona, occasion
        )
        if frequency == "off":
            return []
        transitions = (timeline or {}).get("transitions") or []

        segments = generate_script(
            songs, occasion, vibe_note, persona, frequency
        )
        if not segments:
            return []

        clips: list[dict] = []
        for seg in segments:
            audio = asyncio.run(synthesize_cached(storage, seg["text"]))
            if audio is None:
                logger.info("host: synthesis failed for slot %r; skipping",
                            seg["slot"])
                continue
            clips.append({**seg, "audio": audio})
        if not clips:
            return []

        intro_clip = next((c for c in clips if c["slot"] == "intro"), None)
        intro_start = pick_intro_start(
            song1_safe_regions,
            (intro_clip["audio"].shape[0] / sr) if intro_clip is not None else 0.0,
        )
        events = place_clips(final_audio, sr, clips, transitions, intro_start)
        if events:
            logger.info(
                "host: placed %d segment(s) (%s persona, %s)",
                len(events), persona, frequency,
            )
        return events
    except Exception:
        logger.exception("host: overlay failed; shipping the mix voiceless")
        return []
