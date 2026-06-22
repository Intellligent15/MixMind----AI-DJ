"""Whole-song pitch resolution for a locked queue.

THE PROBLEM: gliding a song's key back to native after a crossfade
(`temporary_pitch_shift`'s fade_out) is the most audible artifact in the
pipeline — human hearing barely notices a song played 1 semitone off
from its first beat, but a key that *bends* mid-song sounds like a tape
machine dying. Real DJs either pick compatible tracks, mask clashes
with short/washy transitions, or shift the incoming track for its
ENTIRE play (the Pitch 'n Time workflow). They never glide.

THE FIX: decide ONCE per queue, before any pair renders, a constant
semitone offset per song. Every pair render that touches a song
pre-shifts its stems/master by that song's offset, so the two renders
that share a song agree exactly and the stitch junction is invisible.
Key-clash logic downstream then compares EFFECTIVE keys (native +
offset), which this resolver makes compatible wherever it can within
the artifact budget.

Algorithm (greedy, deterministic, no LLM):

    walk songs in queue order, tracking the previous song's EFFECTIVE key
      - if this song's NATIVE key is compatible with the previous
        effective key -> offset 0 (reset point: prevents drift from
        accumulating across the set)
      - else compute the smallest shift that matches, clamp to the
        artifact cap (|offset| <= 2 semitones); if the needed shift
        exceeds the cap -> offset 0 and the clash is ACCEPTED (the
        planner masks it with a short or washy transition style)

Example — queue keys: Am(8A), Em(9A), Cm(5A), C#m(12A)
    Am   -> 0   (first song)
    Em   -> 0   (9A adjacent to 8A: compatible)
    Cm   -> +2? compute_pitch_shift(Em, Cm) wants Cm -> Em = +4 — over
            the cap -> 0, clash accepted (planner picks e.g. wash_out)
    C#m  -> compute_pitch_shift(Cm, C#m) = -1 -> offset -1; this song
            plays one semitone flat for its entire duration —
            imperceptible to listeners, and the blend with Cm is clean.

Pure module: no DB, no settings, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from app.services.mixer.archetypes import camelot_compatible
from app.services.mixer.plan import (
    _FLAT_TO_SHARP,
    _PITCH_CLASSES,
    compute_pitch_shift,
)
from app.services.mixer.types import AnalysisBundle

# Whole-song shifts past this many semitones trade an invisible key
# match for audible pyrubberband artifacts and a "wrong version" feel on
# familiar vocals. Past the cap, accepting the clash sounds better.
WHOLE_SONG_PITCH_CAP = 2


@dataclass(frozen=True)
class SongKey:
    key: str | None          # e.g. "Am", "F#"
    camelot_key: str | None  # e.g. "8A"


def transpose_key(key: str | None, semitones: int) -> str | None:
    """Transpose a key string by N semitones ('Am' +2 -> 'Bm')."""
    if not key or semitones == 0:
        return key
    is_minor = key.endswith("m")
    root = key[:-1] if is_minor else key
    root = _FLAT_TO_SHARP.get(root, root)
    if root not in _PITCH_CLASSES:
        return key  # unknown spelling — leave untouched
    idx = (_PITCH_CLASSES.index(root) + semitones) % 12
    return _PITCH_CLASSES[idx] + ("m" if is_minor else "")


def transpose_camelot(camelot: str | None, semitones: int) -> str | None:
    """Transpose a Camelot code by N semitones.

    One semitone up moves SEVEN steps around the Camelot wheel (the
    wheel is ordered by fifths, and 7 semitones = a fifth), letter
    (mode) unchanged: '8A' +1 -> '3A'.
    """
    if not camelot or semitones == 0:
        return camelot
    try:
        num, letter = int(camelot[:-1]), camelot[-1].upper()
    except (ValueError, IndexError):
        return camelot
    new_num = ((num - 1 + 7 * semitones) % 12) + 1
    return f"{new_num}{letter}"


def effective_bundle(bundle: AnalysisBundle, offset: int) -> AnalysisBundle:
    """A copy of `bundle` whose key/camelot reflect the whole-song offset.
    BPM and timing fields are untouched — pitch shifting preserves time."""
    if offset == 0:
        return bundle
    return replace(
        bundle,
        key=transpose_key(bundle.key, offset),
        camelot_key=transpose_camelot(bundle.camelot_key, offset),
    )


def resolve_pitch_offsets(
    keys: list[SongKey],
    cap: int = WHOLE_SONG_PITCH_CAP,
) -> list[int]:
    """Greedy queue walk -> one whole-song offset per song (see module
    docstring for the algorithm and an example)."""
    offsets: list[int] = []
    prev_eff: SongKey | None = None
    for sk in keys:
        if prev_eff is None:
            offsets.append(0)
            prev_eff = sk
            continue

        offset = 0
        if not camelot_compatible(prev_eff.camelot_key, sk.camelot_key):
            if prev_eff.key and sk.key:
                try:
                    shift = compute_pitch_shift(prev_eff.key, sk.key)
                except ValueError:
                    shift = 0
                if shift != 0 and abs(shift) <= cap:
                    offset = shift
                # else: clash accepted — planner masks it with style choice
        offsets.append(offset)
        prev_eff = SongKey(
            key=transpose_key(sk.key, offset),
            camelot_key=transpose_camelot(sk.camelot_key, offset),
        )
    return offsets
