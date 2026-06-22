"""Unit tests for whole-song pitch resolution (pure, no DB/audio)."""

from __future__ import annotations

from app.services.mixer.pitch_resolver import (
    SongKey,
    effective_bundle,
    resolve_pitch_offsets,
    transpose_camelot,
    transpose_key,
)
from app.services.mixer.types import AnalysisBundle


def _bundle(key: str, camelot: str) -> AnalysisBundle:
    return AnalysisBundle(
        bpm=120.0, key=key, camelot_key=camelot, time_signature=4,
        beat_grid=[], downbeats=[], sections=[], duration=200.0,
    )


# ----------------------------------------------------------- transposition

def test_transpose_key_wraps_and_keeps_mode():
    assert transpose_key("Am", 2) == "Bm"
    assert transpose_key("B", 1) == "C"          # wraps the octave
    assert transpose_key("C", -1) == "B"
    assert transpose_key("F#m", 0) == "F#m"      # no-op
    assert transpose_key(None, 2) is None
    assert transpose_key("weird", 2) == "weird"  # unknown spelling untouched


def test_transpose_camelot_moves_seven_per_semitone():
    # +1 semitone = +7 around the wheel (a fifth), same letter.
    assert transpose_camelot("8A", 1) == "3A"
    assert transpose_camelot("8B", 1) == "3B"
    assert transpose_camelot("3A", -1) == "8A"   # inverse
    assert transpose_camelot("12A", 1) == "7A"   # wraps 1..12
    assert transpose_camelot(None, 1) is None
    assert transpose_camelot("8A", 0) == "8A"


def test_transpose_round_trip_consistency():
    # Key and Camelot transposition must agree: Am(8A) +2 -> Bm, and Bm's
    # Camelot is 10A = 8A moved by 2 * 7 = 14 ≡ +2 positions.
    assert transpose_key("Am", 2) == "Bm"
    assert transpose_camelot("8A", 2) == "10A"


def test_effective_bundle_only_touches_keys():
    b = _bundle("Am", "8A")
    eff = effective_bundle(b, -1)
    assert eff.key == "G#m" and eff.camelot_key == "1A"
    assert eff.bpm == b.bpm and eff.duration == b.duration
    assert effective_bundle(b, 0) is b  # zero offset = same object


# --------------------------------------------------------------- resolver

def test_compatible_neighbors_get_zero():
    # 8A -> 8B (relative) -> 9B (adjacent): all compatible, all native.
    keys = [SongKey("Am", "8A"), SongKey("C", "8B"), SongKey("G", "9B")]
    assert resolve_pitch_offsets(keys) == [0, 0, 0]


def test_small_clash_resolved_within_cap():
    # Am(8A) -> G#m(1A): one semitone apart, wheel-distant -> shift +1.
    keys = [SongKey("Am", "8A"), SongKey("G#m", "1A")]
    offsets = resolve_pitch_offsets(keys)
    assert offsets == [0, 1]


def test_clash_beyond_cap_is_accepted_not_forced():
    # Am(8A) -> C#m(12A): needs ±4 — over the ±2 cap -> accept the clash.
    keys = [SongKey("Am", "8A"), SongKey("C#m", "12A")]
    assert resolve_pitch_offsets(keys) == [0, 0]


def test_chain_compares_against_effective_key():
    # Song2 shifts +1 to match song1; song3 is then judged against
    # song2's EFFECTIVE key (Am), not its native G#m.
    keys = [
        SongKey("Am", "8A"),    # native
        SongKey("G#m", "1A"),   # -> +1, plays as Am(8A)
        SongKey("Em", "9A"),    # 9A adjacent to effective 8A -> 0
    ]
    assert resolve_pitch_offsets(keys) == [0, 1, 0]


def test_compatible_native_key_resets_drift():
    # Even mid-chain, a song whose NATIVE key already works gets 0 —
    # offsets never accumulate across the set.
    keys = [
        SongKey("Am", "8A"),
        SongKey("Bm", "10A"),   # 10A vs 8A: clash; Bm->Am = -2 within cap
        SongKey("F#m", "11A"),  # native 11A adjacent to effective... 
    ]
    offsets = resolve_pitch_offsets(keys)
    assert offsets[0] == 0 and offsets[1] == -2
    # song2 plays as Am(8A); F#m(11A) vs 8A clashes; F#m->Am = +3 > cap -> 0
    assert offsets[2] == 0


def test_unknown_keys_never_crash_and_never_shift():
    keys = [SongKey(None, None), SongKey("Am", "8A"), SongKey("??", "??")]
    assert resolve_pitch_offsets(keys) == [0, 0, 0]
