"""Voice-over mixdown: place host clips into the stitched mix, ducked
like a radio DJ's sidechain.

Pure numpy — placement policy and gain math only. Slot rules:

  * "intro"  — at the caller-supplied safe start (vocal-safety-derived),
    must fully land before the first transition begins;
  * pair k   — just after transition k ends (B's entry, typically the
    least vocal stretch), must clear the next transition comfortably;
  * "outro"  — ends a few seconds before the mix does.

Any clip that doesn't fit its slot is skipped — silence beats talking
over a vocal.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

DUCK_GAIN = 0.45          # music level under speech
DUCK_ATTACK_S = 0.2
DUCK_RELEASE_S = 0.5
PAIR_SLOT_OFFSET_S = 0.5  # after the transition end
PAIR_SLOT_CLEARANCE_S = 5.0   # must end this far before the next transition
INTRO_CLEARANCE_S = 2.0
OUTRO_TAIL_S = 3.0


def duck_and_add(
    mix: np.ndarray, sr: int, clip: np.ndarray, start_sample: int
) -> None:
    """In place: duck the music around [start, start+len(clip)] with soft
    attack/release, then add the voice."""
    n = mix.shape[0]
    start = max(0, start_sample)
    end = min(n, start + clip.shape[0])
    if end <= start:
        return
    attack = int(DUCK_ATTACK_S * sr)
    release = int(DUCK_RELEASE_S * sr)

    env_lo = max(0, start - attack)
    env_hi = min(n, end + release)
    env = np.ones(env_hi - env_lo, dtype=np.float32)
    hold_lo = start - env_lo
    hold_hi = end - env_lo
    env[hold_lo:hold_hi] = DUCK_GAIN
    if hold_lo > 0:
        env[:hold_lo] = np.linspace(1.0, DUCK_GAIN, hold_lo, dtype=np.float32)
    tail = env.shape[0] - hold_hi
    if tail > 0:
        env[hold_hi:] = np.linspace(DUCK_GAIN, 1.0, tail, dtype=np.float32)

    mix[env_lo:env_hi] *= env[:, None]
    mix[start:end] += clip[: end - start]
    np.clip(mix[start:end], -0.999, 0.999, out=mix[start:end])


def place_clips(
    mix: np.ndarray,
    sr: int,
    clips: list[dict],           # [{"slot", "text", "audio"}]
    transitions: list[dict],     # timeline transitions (output seconds)
    intro_start_s: float,
) -> list[dict]:
    """Mutates `mix`; returns timeline "host" events for the placed ones."""
    events: list[dict] = []
    total_s = mix.shape[0] / sr
    by_index = {t["index"]: t for t in transitions}
    first_transition_start = min(
        (t["start"] for t in transitions), default=total_s
    )

    for item in clips:
        audio = item["audio"]
        clip_s = audio.shape[0] / sr
        slot = item["slot"]
        start_s: float | None = None

        if slot == "intro":
            if intro_start_s + clip_s <= first_transition_start - INTRO_CLEARANCE_S:
                start_s = intro_start_s
        elif slot == "outro":
            candidate = total_s - clip_s - OUTRO_TAIL_S
            if candidate > 0:
                start_s = candidate
        elif isinstance(slot, int) and slot in by_index:
            candidate = by_index[slot]["end"] + PAIR_SLOT_OFFSET_S
            next_start = min(
                (t["start"] for t in transitions if t["start"] > candidate),
                default=total_s,
            )
            if candidate + clip_s <= next_start - PAIR_SLOT_CLEARANCE_S:
                start_s = candidate

        if start_s is None:
            logger.info("host mixdown: slot %r doesn't fit; skipping", slot)
            continue
        duck_and_add(mix, sr, audio, int(start_s * sr))
        events.append({
            "slot": "intro" if slot == "intro" else
                    "outro" if slot == "outro" else int(slot),
            "start": round(start_s, 3),
            "end": round(start_s + clip_s, 3),
            "text": item["text"],
        })
    return events
