"""Tests for `services.mixer.executor.render`.

We avoid loading real audio by stubbing `soundfile.read` to return
synthetic numpy arrays, and stubbing pyrubberband so we don't shell out
to rubberband-cli. The executor's actual time-stretch / pitch-shift
behavior is left for the integration smoke test (Task 8) where we
exercise the real rubberband binary against a known clip.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from app.services.mixer.executor import (
    SOFT_CLIP_CEILING,
    SOFT_KNEE_THRESHOLD,
    _apply_filter_sweep,
    _apply_reverb,
    _soft_knee_limit,
    render,
)
from app.services.mixer.types import (
    AnalysisBundle,
    MixerPreconditionError,
    SongRenderInputs,
)


SR = 44100
DUR = 5.0  # seconds; short for fast tests
N = int(SR * DUR)


def _bundle(bpm: float = 120.0, key: str = "C") -> AnalysisBundle:
    sec_per_beat = 60.0 / bpm
    sec_per_bar = sec_per_beat * 4
    return AnalysisBundle(
        bpm=bpm,
        key=key,
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * sec_per_beat for i in range(int(DUR / sec_per_beat))],
        downbeats=[i * sec_per_bar for i in range(int(DUR / sec_per_bar))],
        sections=[{"start": 0.0, "end": DUR, "label": "body"}],
        duration=DUR,
    )


def _stems_dict(prefix: str) -> dict[str, str]:
    """4-stem path map; values are arbitrary strings since soundfile is mocked."""
    return {s: f"stems/{prefix}/{s}.wav" for s in ("vocals", "drums", "bass", "other")}


def _inputs(bpm: float = 120.0, key: str = "C", prefix: str = "A") -> SongRenderInputs:
    return SongRenderInputs(stem_paths=_stems_dict(prefix), analysis=_bundle(bpm, key))


class _FakeStorage:
    def path(self, key: str):
        return f"/fake/{key}"


def _fake_sf_read(quiet: float = 0.0):
    """Returns a soundfile.read stub. Each stem returns a constant-tone
    stereo float32 array; summing them gives a recognisable per-song mix.
    `quiet` knocks each stem down so the 4-stem sum stays within ±1.0."""
    def _read(path, always_2d=True, dtype="float32"):
        # Choose tone by stem name to keep mixes distinguishable.
        amp = 0.2 - quiet
        sig = amp * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR
    return _read


def test_render_happy_path_no_stretch_no_shift():
    """Identical BPM + key. 1-bar crossfade. Output exists, length is
    correct, seam region has non-zero content.

    NOTE: plan-text said duration_bars=4 but at 120bpm that's 8s while
    the synthetic signal is only 5s — crossfade would extend past A's
    end. Reduced to 1 bar (2s) so the crossfade fits.
    """
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "crossfade_stem", "stem": "vocals",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "bass",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "other",
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)

    assert result.sample_rate == SR
    assert not result.pitch_shift_warning
    # Decode output WAV bytes to verify shape.
    decoded, sr = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert sr == SR
    # Output length = N (A's full length, identical-BPM means no stretch).
    # Output is a_seam (=0 samples for seam=0.0) + crossfade + B's tail.
    # With seam_a=0, the entire output is the B side after crossfade.
    # For more useful coverage we want seam_a > 0; we exercise that in
    # test_render_seam_alignment below.


def test_render_seam_alignment_identical_bpm():
    """Seam in middle of A: pre-seam is from A, post-crossfade is from B."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar,    # seam at 2.0s in A
         "to_song_time_start": 0.0,              # seam at 0.0s in B
         "duration_bars": 1},                    # 1-bar crossfade (2s = 88200 samples)
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)

    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    seam_sample = int(round(sec_per_bar * SR))
    crossfade_samples = int(round(sec_per_bar * SR))
    expected_len = seam_sample + (N - 0)  # full B after b_seam=0
    assert decoded.shape[0] == expected_len
    # First sample should equal A pre-seam (per fake_sf_read, A's vocals
    # sums to 0.2-0.2+0.2+0.2 = 0.4 stereo).
    assert decoded[0, 0] == pytest.approx(0.4, abs=1e-5)
    # Mid-crossfade (sample seam_sample + crossfade_samples/2): equal blend.
    mid = seam_sample + crossfade_samples // 2
    assert -1.0 < decoded[mid, 0] < 1.0
    # Post-crossfade: pure B.
    post = seam_sample + crossfade_samples + 100
    assert decoded[post, 0] == pytest.approx(0.4, abs=1e-5)


def test_render_calls_time_stretch_with_correct_rate():
    """A=120, B=128 → rate = 120/128 = 0.9375 (slow B down to A's tempo)."""
    a = _inputs(bpm=120.0, prefix="A")
    b = _inputs(bpm=128.0, prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    captured = {}

    def fake_stretch(audio, sr, rate):
        captured["rate"] = rate
        # Return audio of approximately rate-scaled length.
        new_len = int(audio.shape[0] / rate)
        return np.zeros((new_len, audio.shape[1]), dtype=audio.dtype)

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.time_stretch", side_effect=fake_stretch),
    ):
        render(plan, a, b)

    assert captured["rate"] == pytest.approx(120.0 / 128.0)


def test_render_calls_pitch_shift_with_correct_semitones():
    a = _inputs(key="C", prefix="A")
    b = _inputs(key="C", prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "pitch_shift", "song": "B", "semitones": -2},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    captured = {}

    def fake_shift(audio, sr, n_steps):
        captured["n_steps"] = n_steps
        return audio

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=fake_shift),
    ):
        result = render(plan, a, b)

    assert captured["n_steps"] == -2
    # |δ|=2 is NOT > 2 → no warning. (Plan text used -3 with a "not warning"
    # assertion, which contradicts LARGE_SHIFT_THRESHOLD=2 / "|δ| > 2 → warn".
    # Reduced to -2 so the no-warn assertion holds against the locked threshold.)
    assert not result.pitch_shift_warning


def test_render_pitch_shift_warning_flag_when_large():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "pitch_shift", "song": "B", "semitones": 5},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n_steps: y),
    ):
        result = render(plan, a, b)
    assert result.pitch_shift_warning


def test_render_temporary_pitch_mutes_vocal_then_fades_it_in():
    """Temporary-pitch path: B's vocal is silent through the ENTIRE pitch
    return (including the crossfade-back window) and only fades in once the
    key has fully settled. We make only the vocal stem non-zero so we can
    read its presence directly out of the rendered B body."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0  # 120 bpm, 4/4

    def _bundle_long(prefix_bpm: float = 120.0) -> AnalysisBundle:
        return AnalysisBundle(
            bpm=prefix_bpm, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    def _read_vocal_only(path, always_2d=True, dtype="float32"):
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    # Pitch shift starts at 4 s and returns over a 2-bar (4 s) window → native
    # by 8 s. The vocal stays silent through that whole return and only fades
    # in afterwards, over [8 s, 12 s], full from 12 s on. The tempo ramp is
    # longer (4–10 s) and intentionally decoupled.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 10.0, "start_bpm": 120.0, "end_bpm": 120.0},
        {"tool": "temporary_pitch_shift", "song": "B", "start_time": 4.0,
         "semitones": -4, "fade_in_bars": 0, "hold_bars": 0, "fade_out_bars": 2},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_read_vocal_only),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n: y),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)

    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Before the pitch return (~3 s): vocal silent.
    pre = np.max(np.abs(out[int(3.0 * SR): int(3.1 * SR)]))
    assert pre < 0.05, f"vocal should be muted before pitch return, got {pre}"
    # DURING the pitch return (~6 s, inside [4 s, 8 s]): still silent — this is
    # the behavior we want (no vocal over the detuned/blended instrumental).
    during = np.max(np.abs(out[int(6.0 * SR): int(6.1 * SR)]))
    assert during < 0.05, f"vocal should be muted during pitch return, got {during}"
    # After the key has fully settled (~14 s): vocal at full level.
    present = np.mean(np.abs(out[int(14.0 * SR): int(14.1 * SR)]))
    assert present > 0.4, f"vocal should be present once key settles, got {present}"
    assert result.pitch_shift_warning  # |-4| > 2


def test_render_tempo_ramp_alone_keeps_vocal_present():
    """Tempo-only path (no temp pitch shift): B's vocal is NOT muted. The
    ramp lands after the crossfade and B is a clean constant-tempo stretch
    through the blend, so holding the vocal out until the ramp settles would
    make it arrive jarringly late. Only a temp *pitch* shift gates the vocal
    (see test_render_temp_pitch_*). The vocal rides in with the crossfade and
    stays present across the ramp window."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0  # 120 bpm, 4/4

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    def _read_vocal_only(path, always_2d=True, dtype="float32"):
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    # Tempo ramp 4 s → 10 s, no temp pitch. Same BPM end-to-end so the
    # mocked timemap_stretch is a passthrough. Vocal rides in with the 1-bar
    # crossfade and stays present — the ramp does not mute it.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 10.0, "start_bpm": 120.0, "end_bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with (
        patch("soundfile.read", side_effect=_read_vocal_only),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)

    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Mid-ramp (~7 s): vocal is present, NOT muted (the regression we fixed).
    during = np.mean(np.abs(out[int(7.0 * SR): int(7.1 * SR)]))
    assert during > 0.4, f"vocal should ride through the tempo ramp, got {during}"
    # Just before settle (~9 s): still present.
    just_before = np.mean(np.abs(out[int(9.0 * SR): int(9.1 * SR)]))
    assert just_before > 0.4, f"vocal should still be present at 9s, got {just_before}"
    # Past the ramp (~15 s): still at full level.
    present = np.mean(np.abs(out[int(15.0 * SR): int(15.1 * SR)]))
    assert present > 0.4, f"vocal should be present once tempo settles, got {present}"


def test_render_uses_original_master_for_a_head():
    """A's body before the seam comes from the untouched master, not the
    stem sum. A master reads 0.3; the 4-stem sum reads 0.4."""
    a = SongRenderInputs(
        stem_paths=_stems_dict("A"), analysis=_bundle(),
        original_audio_path="orig/A_master.wav",
    )
    b = _inputs(prefix="B")  # no original → B stays stem-based

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.3 * np.ones((N, 2), dtype=np.float32), SR
        sig = 0.2 * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR  # 4-stem sum = 0.2-0.2+0.2+0.2 = 0.4

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Pre-seam (~0.5 s, before the 2 s seam): the untouched master (0.3),
    # not the stem sum (0.4).
    head = np.mean(np.abs(out[int(0.5 * SR): int(0.6 * SR)]))
    assert abs(head - 0.3) < 0.02, f"A head should be master 0.3, got {head}"


def test_render_splices_original_master_into_settled_b_tail():
    """Once B has fully settled (tempo+pitch+vocal native), its tail is the
    untouched master, not the stem reconstruction. Same 16 s temp-pitch
    setup as the mute test, but B carries an original master reading 0.9."""
    long_dur = 16.0
    long_n = int(SR * long_dur)
    sec_per_bar = 2.0

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * sec_per_bar for i in range(int(long_dur / sec_per_bar))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(
        stem_paths=_stems_dict("B"), analysis=_bundle_long(),
        original_audio_path="orig/B_master.wav",
    )

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.9 * np.ones((long_n, 2), dtype=np.float32), SR
        amp = 0.5 if "vocals" in path else 0.0
        return amp * np.ones((long_n, 2), dtype=np.float32), SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0, "duration_bars": 1},
        {"tool": "set_tempo_ramp", "song": "B",
         "start_time": 4.0, "end_time": 8.0, "start_bpm": 120.0, "end_bpm": 120.0},
        {"tool": "temporary_pitch_shift", "song": "B", "start_time": 4.0,
         "semitones": -4, "fade_in_bars": 0, "hold_bars": 0, "fade_out_bars": 2},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with (
        patch("soundfile.read", side_effect=_read),
        patch("pyrubberband.pyrb.pitch_shift", side_effect=lambda y, sr, n: y),
        patch("pyrubberband.pyrb.timemap_stretch", side_effect=lambda y, sr, tm: y),
    ):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Settled tail (~14 s): the master (0.9), not the stem vocal level (0.5).
    tail = np.mean(np.abs(out[int(14.0 * SR): int(14.1 * SR)]))
    assert abs(tail - 0.9) < 0.05, f"settled tail should be master 0.9, got {tail}"
    # During the pitch return (~6 s): still the stem path, master NOT yet
    # spliced (vocal muted → ~0).
    during = np.max(np.abs(out[int(6.0 * SR): int(6.1 * SR)]))
    assert during < 0.05, f"master must not appear during transition, got {during}"


def test_render_precondition_rejects_wrong_sample_rate():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [{"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 1}]

    def bad_sr_read(path, always_2d=True, dtype="float32"):
        return np.zeros((N, 2), dtype=np.float32), 22050

    with patch("soundfile.read", side_effect=bad_sr_read):
        with pytest.raises(MixerPreconditionError, match="sample rate"):
            render(plan, a, b)


def test_render_precondition_rejects_mono():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [{"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 1}]

    def mono_read(path, always_2d=True, dtype="float32"):
        return np.zeros((N, 1), dtype=np.float32), SR

    with patch("soundfile.read", side_effect=mono_read):
        with pytest.raises(MixerPreconditionError, match="channels"):
            render(plan, a, b)


def _kick_train(emphasis_beat: int, n_beats: int, spb_beat: int) -> np.ndarray:
    """Stereo low-freq (55 Hz) kick pattern with a distinctive per-bar shape:
    a strong hit on the bar's `emphasis_beat` (the downbeat) and a weaker one
    at the half-bar, silence on the off-beats. That gives the low-band onset
    envelope a clear bar phase for the aligner to lock onto (unlike a
    four-on-the-floor train, where every beat looks the same and the aligner
    should — and does — refuse to shift)."""
    spbar = spb_beat * 4
    out = np.zeros((n_beats * spb_beat + spbar, 2), dtype=np.float32)
    n = int(0.1 * SR)
    decay = np.exp(-np.linspace(0.0, 6.0, n)).astype(np.float32)
    tone = np.sin(2 * np.pi * 55.0 * np.arange(n) / SR).astype(np.float32)
    for i in range(n_beats):
        phase = (i - emphasis_beat) % 4
        amp = 1.0 if phase == 0 else (0.5 if phase == 2 else 0.0)
        if amp == 0.0:
            continue
        s = i * spb_beat
        out[s : s + n] += (amp * decay * tone)[:, None]
    return out


def test_align_pair_phase_shifts_when_b_bar_is_offset():
    """B's bar emphasis is one beat late vs A → the aligner enters B one beat
    later so the bars lock in phase."""
    from app.services.mixer.executor import _align_pair_phase

    spb_beat = int(round(60.0 / 120.0 * SR))  # samples per beat @120bpm
    spbar = float(spb_beat * 4)
    a = _kick_train(emphasis_beat=0, n_beats=24, spb_beat=spb_beat)
    b = _kick_train(emphasis_beat=1, n_beats=24, spb_beat=spb_beat)

    out_seam = _align_pair_phase(a, b, 0, 0, spbar, 4, SR)
    assert out_seam == spb_beat  # shifted forward by exactly 1 beat


def test_align_pair_phase_no_shift_when_already_aligned():
    from app.services.mixer.executor import _align_pair_phase

    spb_beat = int(round(60.0 / 120.0 * SR))
    spbar = float(spb_beat * 4)
    a = _kick_train(emphasis_beat=0, n_beats=24, spb_beat=spb_beat)
    b = _kick_train(emphasis_beat=0, n_beats=24, spb_beat=spb_beat)

    assert _align_pair_phase(a, b, 0, 0, spbar, 4, SR) == 0


def test_align_pair_phase_no_shift_on_flat_audio():
    """Constant tone has no rhythmic pattern to disambiguate → leave the
    analysis-derived seam alone (the synthetic-tone executor tests rely on
    this no-op)."""
    from app.services.mixer.executor import _align_pair_phase

    flat = np.full((4 * SR, 2), 0.2, dtype=np.float32)
    spbar = float(int(round(60.0 / 120.0 * SR)) * 4)
    assert _align_pair_phase(flat, flat, 0, 0, spbar, 4, SR) == 0


def test_soft_knee_limit_passes_through_below_ceiling():
    """A buffer already at/under the ceiling is returned untouched — no
    global attenuation. This is the core regression vs the old normalize,
    which scaled the whole buffer whenever any sample crossed the ceiling."""
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal((1000, 2)).astype(np.float32) * 0.3).clip(-0.8, 0.8)
    out = _soft_knee_limit(audio, SOFT_KNEE_THRESHOLD, SOFT_CLIP_CEILING)
    assert out is audio  # untouched, not even copied


def test_soft_knee_limit_does_not_duck_quiet_samples():
    """One loud transient must not pull down the quiet body. The old
    normalize multiplied everything by ceiling/peak; the limiter leaves
    sub-knee samples exactly as they were and only shapes what's over."""
    audio = np.full((100, 2), 0.3, dtype=np.float32)
    audio[50] = 1.6  # single loud transient
    out = _soft_knee_limit(audio, SOFT_KNEE_THRESHOLD, SOFT_CLIP_CEILING)
    # Quiet body untouched (would be 0.3 * 0.999/1.6 ≈ 0.187 under the old normalize).
    assert out[0, 0] == pytest.approx(0.3, abs=1e-6)
    assert out[99, 0] == pytest.approx(0.3, abs=1e-6)
    # Transient tamed below the ceiling, sign preserved.
    assert 0.9 < out[50, 0] < SOFT_CLIP_CEILING


def test_soft_knee_limit_caps_and_preserves_sign():
    """Output never reaches the ceiling and keeps the input's sign."""
    audio = np.array([[5.0, -5.0], [0.95, -0.95]], dtype=np.float32)
    out = _soft_knee_limit(audio, SOFT_KNEE_THRESHOLD, SOFT_CLIP_CEILING)
    assert np.all(np.abs(out) < SOFT_CLIP_CEILING)
    assert out[0, 0] > 0 and out[0, 1] < 0


def test_render_soft_clips_loud_sum():
    """Sum two stems that exceed 1.0 peak; verify output is clamped to 0.999."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 1.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    def loud_read(path, always_2d=True, dtype="float32"):
        # Each stem returns 0.4 → 4-stem sum = 1.6 (clipping needed).
        # Use +0.4 on all 4 stems (no sign flip) so the sum is monotonic.
        return 0.4 * np.ones((N, 2), dtype=np.float32), SR

    with patch("soundfile.read", side_effect=loud_read):
        result = render(plan, a, b)

    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.max() <= 0.999 + 1e-5
    assert decoded.min() >= -0.999 - 1e-5


def test_render_clamps_crossfade_when_stems_shorter_than_plan(caplog):
    """Plan asks for a 1-bar crossfade (2s) starting at seam=2.0s, but
    the stem WAVs are only 2.5s long — there's room for 0.5s of crossfade,
    not 2s. Executor should clamp to what's available, log a WARN, and
    produce a valid WAV rather than raising MixerPreconditionError.
    Models the real-world drift between Song.duration_seconds (yt-dlp
    metadata) and the actual stem WAV length after Demucs + pyrubberband."""
    short_n = int(SR * 2.5)  # 2.5s — leaves 0.5s after the 2s seam

    def short_read(path, always_2d=True, dtype="float32"):
        return 0.2 * np.ones((short_n, 2), dtype=np.float32), SR

    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0,   # snaps to downbeat 2.0
         "to_song_time_start": 0.0,
         "duration_bars": 1},            # asks for 2s of crossfade
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=short_read), \
         caplog.at_level("WARNING"):
        result = render(plan, a, b)

    # Should have logged the clamp.
    assert any("clamping crossfade" in rec.message for rec in caplog.records)
    # And produced a non-empty WAV.
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_rejects_when_no_overlap_available():
    """A's downbeats extend past the actual stem length (analysis claims
    a 10s downbeat in a 5s stem — a real bug case, not just sample
    drift). The seam snaps to that beyond-end downbeat and max_crossfade_a
    becomes negative. Executor still raises rather than silently
    producing nonsense."""
    bad_analysis = AnalysisBundle(
        bpm=120.0,
        key="C",
        camelot_key="8B",
        time_signature=4,
        beat_grid=[i * 0.5 for i in range(40)],
        downbeats=[10.0, 12.0],   # past the 5s audio
        sections=[{"start": 0.0, "end": DUR, "label": "body"}],
        duration=DUR,
    )
    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=bad_analysis)
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 8.0,   # snaps to 10.0 downbeat (past EOF)
         "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(MixerPreconditionError, match="no overlap"):
            render(plan, a, b)


def test_render_equal_power_curve_keeps_loudness_flat_at_midpoint():
    """Equal-power crossfade: at t=0.5 both gains = cos(π/4) = sin(π/4) ≈
    0.707. Mid-crossfade output of two identical signals (both at amp 0.4)
    should be 0.4 × (0.707 + 0.707) = 0.566, NOT 0.4 (which linear would
    produce). This is what 'no midpoint dip' means in practice: when both
    inputs are at +0.4, the output is louder by sqrt(2) than either input
    alone, which is the correct sum-of-uncorrelated-powers behavior."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0

    def steady_read(path, always_2d=True, dtype="float32"):
        # Identical positive constant for both songs → A and B at +0.4
        # after the 4-stem sum (4 × 0.1).
        return 0.1 * np.ones((N, 2), dtype=np.float32), SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=steady_read):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    seam_sample = int(round(sec_per_bar * SR))
    crossfade_samples = int(round(sec_per_bar * SR))
    mid = seam_sample + crossfade_samples // 2
    # Equal-power midpoint: cos(π/4)*0.4 + sin(π/4)*0.4 ≈ 0.5657
    assert decoded[mid, 0] == pytest.approx(0.4 * np.sqrt(2.0), abs=1e-3)
    # Compare against the linear case at the same midpoint: 0.4 (no boost).
    plan_linear = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=steady_read):
        result_lin = render(plan_linear, a, b)
    decoded_lin, _ = sf.read(io.BytesIO(result_lin.wav_bytes), always_2d=True)
    assert decoded_lin[mid, 0] == pytest.approx(0.4, abs=1e-3)


def test_render_rejects_unsupported_curve():
    """s_curve and exponential are reserved for Phase 9; the executor
    refuses them rather than silently falling back to linear."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "s_curve"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(NotImplementedError, match="s_curve"):
            render(plan, a, b)


def test_render_per_stem_envelopes_sum_correctly():
    """Per-stem envelopes: vocals fade out earlier than drums/bass/other.

    A's stems carry distinct constants; B's stems are zero. So at any output
    sample inside the post-seam region we know exactly which stems should
    still be ringing (gain≈1.0 on A-side because B is silent, less the
    envelope fade applied during each stem's window). The sum at each
    probe point matches the analytic expectation."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 60.0 / 120.0 * 4  # = 2.0s; total stem N = 5s, seam at 0

    # Distinct A levels per stem so the post-seam sum tells us which
    # stems are still in. B silent so envelopes only fade A.
    levels = {"vocals": 0.1, "drums": 0.2, "bass": 0.3, "other": 0.05}

    def stems_read(path, always_2d=True, dtype="float32"):
        if "B/" in path or "B_" in path or "/B/" in path:
            return np.zeros((N, 2), dtype=np.float32), SR
        for s, v in levels.items():
            if f"/{s}." in path:
                return v * np.ones((N, 2), dtype=np.float32), SR
        return np.zeros((N, 2), dtype=np.float32), SR

    # Vocals fade out bars [0, 1) → seconds [0, 2.0); other stems hold
    # through bar 0 and fade in bars [1, 2) → seconds [2.0, 4.0).
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 2},
        {"tool": "crossfade_stem", "stem": "vocals", "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "drums", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "bass", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
        {"tool": "crossfade_stem", "stem": "other", "from_song": "A", "to_song": "B",
         "start_bar": 1, "duration_bars": 1, "curve": "linear"},
    ]
    with patch("soundfile.read", side_effect=stems_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # Probe at 0.25s into the vocal fade-out (t=0.125 in vocal window):
    # vocals weight = 1 - 0.125 = 0.875 → vocals contribute 0.0875.
    # drums/bass/other are still pre-window (pure A) → full levels.
    probe_a = int(0.25 * SR)
    expected_a = 0.875 * levels["vocals"] + levels["drums"] + levels["bass"] + levels["other"]
    assert out[probe_a, 0] == pytest.approx(expected_a, abs=1e-3)

    # Probe at 1.0s (bar 0.5 into vocals' window so vocal=0.5 gain,
    # still pre-window for other stems):
    probe_b = int(1.0 * SR)
    expected_b = 0.5 * levels["vocals"] + levels["drums"] + levels["bass"] + levels["other"]
    assert out[probe_b, 0] == pytest.approx(expected_b, abs=1e-3)

    # Probe at 3.0s (mid-fade for drums/bass/other; vocals fully out):
    # drums/bass/other gain = 1 - 0.5 = 0.5; vocals = 0.
    probe_c = int(3.0 * SR)
    expected_c = 0.5 * (levels["drums"] + levels["bass"] + levels["other"])
    assert out[probe_c, 0] == pytest.approx(expected_c, abs=1e-3)
def _baseline_stem_calls(curve: str = "equal_power") -> list[dict]:
    """Standard 4-stem crossfade tail every plan needs."""
    return [
        {"tool": "crossfade_stem", "stem": s,
         "from_song": "A", "to_song": "B",
         "start_bar": 0, "duration_bars": 1, "curve": curve}
        for s in ("vocals", "drums", "bass", "other")
    ]


def test_render_filter_sweep_lowpass_kills_high_frequencies():
    """A lowpass sweep from 20 kHz → 200 Hz over the post-window region
    should leave the high-frequency content noticeably attenuated by the
    end of the sweep. We feed in a high-frequency sine on A and read the
    energy out before vs after the sweep window."""
    # 4 second signal so the sweep window has room.
    long_dur = 4.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's stems are all a 5 kHz tone; B's are silent so we can isolate A.
    t_axis = np.arange(long_n) / SR
    tone = 0.1 * np.sin(2.0 * np.pi * 5000.0 * t_axis).astype(np.float32)
    tone_stereo = np.column_stack([tone, tone])

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/" in path:
            return tone_stereo.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 3.0s; A's body before the seam includes the sweep window
    # [0.5s, 2.5s]. After 2.5s the 5 kHz energy should be much lower.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 3.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "filter_sweep", "song": "A", "type": "lowpass",
         "start_time": 0.5, "end_time": 2.5,
         "start_cutoff_hz": 20000.0, "end_cutoff_hz": 200.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    pre_rms = float(np.sqrt(np.mean(out[int(0.3 * SR): int(0.5 * SR)] ** 2)))
    post_rms = float(np.sqrt(np.mean(out[int(2.6 * SR): int(2.9 * SR)] ** 2)))
    # 5 kHz survives a 20 kHz cutoff (start) but is heavily attenuated by
    # a 200 Hz cutoff (end). Expect at least a 10× drop.
    assert post_rms < pre_rms * 0.1, (
        f"lowpass sweep failed to kill highs: pre={pre_rms:.4f} post={post_rms:.4f}"
    )


def test_render_filter_sweep_clamps_zero_cutoff_silently():
    """A 0 Hz cutoff is illegal for iirfilter (division by zero in the
    bilinear transform). The executor clamps to 20 Hz silently rather
    than raising — the LLM prompt-side rules will steer away from this,
    but the DSP must not crash if it slips through."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "filter_sweep", "song": "A", "type": "highpass",
         "start_time": 0.5, "end_time": 1.5,
         "start_cutoff_hz": 0.0, "end_cutoff_hz": -100.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    # Just ensure we got a valid WAV back; the clamp shouldn't blow up.
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_echo_out_decaying_taps_after_cut():
    """echo_out at start_time T: dry signal is hard-cut at T (silence
    immediately after), and N taps of the pre-cut audio appear at one-beat
    intervals after T, each attenuated by feedback ** i."""
    long_dur = 6.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # Put a unit-amplitude pulse on A's vocal stem; everything else silent.
    # Pulse fits within the last beat before the cut so it's the tap source.
    bpm = 120.0
    delay_samp = int(SR * 60.0 / bpm)
    cut_samp = int(2.0 * SR)
    pulse_start = cut_samp - delay_samp + 100  # a few samples into the tap window
    pulse_len = 200

    a_vocal = np.zeros((long_n, 2), dtype=np.float32)
    a_vocal[pulse_start : pulse_start + pulse_len] = 0.5

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/vocals" in path:
            return a_vocal.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 5s so the entire echo region is observable in the output.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 5.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "echo_out", "song": "A",
         "start_time": 2.0, "beats": 3, "feedback": 0.5, "bpm": bpm},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # The original pulse pre-cut is preserved.
    assert out[pulse_start + 50, 0] == pytest.approx(0.5, abs=0.05)

    # Dry signal is hard-cut: NO energy at sample (cut_samp + 100), well
    # before the first echo lands.
    dry_after = float(np.max(np.abs(out[cut_samp + 50 : cut_samp + 200])))
    assert dry_after < 0.05, f"dry signal should be cut, got {dry_after}"

    # Each tap is the same pulse, attenuated by feedback ** i, placed at
    # cut_samp + i*delay_samp. The pulse arrives at the same offset within
    # each tap window as it was within the pre-cut tap source.
    pulse_offset_in_tap = pulse_start - (cut_samp - delay_samp)
    for i in (1, 2, 3):
        tap_peak_samp = cut_samp + i * delay_samp + pulse_offset_in_tap + 50
        expected = 0.5 * (0.5 ** i)
        # Output channels are stereo identical here.
        observed = float(out[tap_peak_samp, 0])
        assert observed == pytest.approx(expected, abs=0.05), (
            f"tap {i}: expected ~{expected:.3f}, got {observed:.3f}"
        )


def test_render_loop_section_repeats_section_length():
    """loop_section with repeats=2 and beats=1 should tile the chosen
    slice twice; the second copy is content-identical to the first.

    We mark the loop-source slice with a unique amplitude on A's vocal
    stem so we can detect it by reading the rendered output."""
    long_dur = 8.0
    long_n = int(SR * long_dur)
    bpm = 120.0
    beat_samp = int(SR * 60.0 / bpm)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=bpm, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's vocal: zero everywhere except a distinct stamp inside the loop slice.
    loop_start = int(1.0 * SR)
    stamp_offset = beat_samp // 4  # well inside the 1-beat loop window
    stamp_len = 100
    stamp_amp = 0.6

    a_vocal = np.zeros((long_n, 2), dtype=np.float32)
    a_vocal[loop_start + stamp_offset : loop_start + stamp_offset + stamp_len] = stamp_amp

    def _read(path, always_2d=True, dtype="float32"):
        if "/A/vocals" in path:
            return a_vocal.copy(), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    # Seam at 7s so the entire looped region is in the output.
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 7.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "loop_section", "song": "A",
         "start_time": 1.0, "beats": 1, "repeats": 2, "bpm": bpm},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # First copy: the original stamp is still where we put it.
    first_peak = float(out[loop_start + stamp_offset + 10, 0])
    assert first_peak == pytest.approx(stamp_amp, abs=0.05)

    # Second copy: same stamp shifted by exactly one beat (the loop length).
    second_idx = loop_start + beat_samp + stamp_offset + 10
    second_peak = float(out[second_idx, 0])
    assert second_peak == pytest.approx(stamp_amp, abs=0.05), (
        f"second loop copy missing: idx={second_idx} got {second_peak}"
    )


def test_render_loop_section_zero_beats_is_noop():
    """beats=0 or repeats=0 must NOT raise — silent no-op."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "loop_section", "song": "A",
         "start_time": 0.0, "beats": 0, "repeats": 4, "bpm": 120.0},
        {"tool": "loop_section", "song": "A",
         "start_time": 0.0, "beats": 2, "repeats": 0, "bpm": 120.0},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_swap_stem_replaces_output_tail_with_to_song_stem():
    """swap_stem at time T: every output sample after T equals
    to_song's stem buffer on that channel."""
    long_dur = 6.0
    long_n = int(SR * long_dur)

    def _bundle_long() -> AnalysisBundle:
        return AnalysisBundle(
            bpm=120.0, key="C", camelot_key="8B", time_signature=4,
            beat_grid=[i * 0.5 for i in range(int(long_dur / 0.5))],
            downbeats=[i * 2.0 for i in range(int(long_dur / 2.0))],
            sections=[{"start": 0.0, "end": long_dur, "label": "body"}],
            duration=long_dur,
        )

    a = SongRenderInputs(stem_paths=_stems_dict("A"), analysis=_bundle_long())
    b = SongRenderInputs(stem_paths=_stems_dict("B"), analysis=_bundle_long())

    # A's vocal stem is a distinctive +0.4 ramp; everything else 0.
    # We'll swap to A's vocal after the seam region — but easier to swap
    # to B's vocals using a sign-flipping signal so the assertion is
    # unambiguous.
    t_axis = np.arange(long_n) / SR
    b_vocal_signal = (0.3 * np.sin(2.0 * np.pi * 220.0 * t_axis)).astype(np.float32)
    b_vocal_stereo = np.column_stack([b_vocal_signal, b_vocal_signal])

    def _read(path, always_2d=True, dtype="float32"):
        if "/B/vocals" in path:
            return b_vocal_stereo.copy(), SR
        if "/A/" in path:
            return 0.1 * np.ones((long_n, 2), dtype=np.float32), SR
        return np.zeros((long_n, 2), dtype=np.float32), SR

    swap_time = 4.0
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *_baseline_stem_calls(),
        {"tool": "swap_stem", "from_song": "A", "to_song": "B",
         "stem": "vocals", "time": swap_time},
    ]
    # Same-BPM songs: rate = 1.0 so the stretch is a passthrough and the
    # B-side buffers stay in their original sample positions.
    with (
        patch("soundfile.read", side_effect=_read),
        patch("pyrubberband.pyrb.time_stretch", side_effect=lambda y, sr, r: y),
    ):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")

    # After the swap (allowing a few samples for the ZC snap), output
    # should match B's vocal stem on both channels.
    probe = int(swap_time * SR) + 200
    assert out[probe, 0] == pytest.approx(b_vocal_stereo[probe, 0], abs=1e-3)
    assert out[probe, 1] == pytest.approx(b_vocal_stereo[probe, 1], abs=1e-3)
    # Pre-swap region is unaffected.
    pre = int(swap_time * SR) - 1000
    assert out[pre, 0] != pytest.approx(b_vocal_stereo[pre, 0], abs=1e-3)


def test_render_swap_stem_past_output_length_is_noop():
    """time past output end → no-op (don't raise)."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        *_baseline_stem_calls(),
        {"tool": "swap_stem", "from_song": "A", "to_song": "B",
         "stem": "drums", "time": 9999.0},
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_unknown_tool_still_raises():
    """The dispatch's else-branch now raises 'unknown tool' (rather than
    the old Phase-7 message) — the wrap-up signal stays the same so the
    worker's failure path still distinguishes 'plan bug' from
    'precondition error'."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 0.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "make_coffee", "song": "A"},
        *_baseline_stem_calls(),
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        with pytest.raises(NotImplementedError, match="unknown tool"):
            render(plan, a, b)


def _ab_reader(a_amp: float, b_amp: float):
    """soundfile.read stub returning a constant tone whose amplitude
    depends on which song's stem path is requested ('/A/' vs '/B/')."""
    def _read(path, always_2d=True, dtype="float32"):
        amp = a_amp if "/A/" in path else b_amp
        return amp * np.ones((N, 2), dtype=np.float32), SR
    return _read


def test_render_a_fade_out_bars_cuts_a_early():
    """With B silent, a_fade_out_bars < duration_bars must drive A to
    silence by the half-way point while a coupled crossfade keeps A
    audible across the whole window."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    # 120bpm => 2s/bar = 88200 samples. seam at 0.0 (a downbeat, no snap);
    # the 2-bar window starts at output sample 0.
    bar = int(2.0 * SR)
    win_start = 0                 # start_bar=0, seam at 0.0
    win_half = win_start + bar    # a_fade ends here when a_fade_out_bars=1
    win_end = win_start + 2 * bar

    def _plan(a_fade):
        stems = []
        for s in ("vocals", "drums", "bass", "other"):
            call = {"tool": "crossfade_stem", "stem": s,
                    "from_song": "A", "to_song": "B",
                    "start_bar": 0, "duration_bars": 2, "curve": "equal_power"}
            if a_fade is not None:
                call["a_fade_out_bars"] = a_fade
            stems.append(call)
        return [
            {"tool": "set_transition_window",
             "from_song_time_start": 0.0, "to_song_time_start": 0.0,
             "duration_bars": 2},
            *stems,
        ]

    # A audible (0.2/stem), B silent.
    with patch("soundfile.read", side_effect=_ab_reader(0.2, 0.0)):
        coupled = render(_plan(None), a, b)       # default == full-window A fade
        decoupled = render(_plan(1), a, b)        # A gone at the half point

    cd, _ = sf.read(io.BytesIO(coupled.wav_bytes), always_2d=True)
    dd, _ = sf.read(io.BytesIO(decoupled.wav_bytes), always_2d=True)

    back_coupled = np.max(np.abs(cd[win_half:win_end]))
    back_decoupled = np.max(np.abs(dd[win_half:win_end]))
    front_decoupled = np.max(np.abs(dd[win_start:win_half]))

    assert back_coupled > 0.1          # coupled: A still ringing in 2nd half
    assert back_decoupled < 1e-3       # decoupled: A silent in 2nd half (B silent too)
    assert front_decoupled > 0.1       # A still present in the 1st half


def test_render_a_fade_out_bars_default_is_identical():
    """Omitting a_fade_out_bars must be byte-identical to setting it equal
    to duration_bars — i.e. the coupled crossfade is unchanged."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")

    def _plan(explicit):
        stems = []
        for s in ("vocals", "drums", "bass", "other"):
            call = {"tool": "crossfade_stem", "stem": s,
                    "from_song": "A", "to_song": "B",
                    "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            if explicit:
                call["a_fade_out_bars"] = 1
            stems.append(call)
        return [
            {"tool": "set_transition_window",
             "from_song_time_start": 1.0, "to_song_time_start": 0.0,
             "duration_bars": 1},
            *stems,
        ]

    with patch("soundfile.read", side_effect=_fake_sf_read()):
        omitted = render(_plan(False), a, b)
        explicit = render(_plan(True), a, b)

    assert omitted.wav_bytes == explicit.wav_bytes


@pytest.mark.parametrize(
    "fx",
    [
        {"tool": "apply_reverb", "song": "A", "start_time": 1.0,
         "tail_duration_bars": 1.0, "wet_level": 0.6, "bpm": 120.0},
        {"tool": "turntable_stop", "song": "A", "start_time": 2.0,
         "duration_bars": 1.0, "bpm": 120.0},
        {"tool": "volume_fade", "song": "A", "start_time": 0.0,
         "duration_bars": 1.0, "start_gain": 1.0, "end_gain": 0.0, "bpm": 120.0},
    ],
)
def test_render_applies_new_dsp_tools(fx):
    """apply_reverb / turntable_stop / volume_fade run through the executor's
    pre_fx path and produce a valid, full-length render."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 1.0,
         "duration_bars": 1},
        fx,
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, sr = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert sr == SR
    assert decoded.shape[0] > 0
    assert np.all(np.abs(decoded) < 1.0001)


def test_render_new_dsp_tool_missing_key_is_noop_not_crash():
    """A new-tool call missing required params degrades to a no-op instead of
    stranding the whole render (KeyError -> failed plan)."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 1.0, "to_song_time_start": 1.0,
         "duration_bars": 1},
        {"tool": "turntable_stop", "song": "A"},  # no bpm/duration
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)  # must not raise
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_volume_fade_targets_single_stem_when_stem_set():
    """volume_fade with `stem` kills only that stem (EQ-kill); the others are
    untouched. A's bass is faded to 0 before the seam; at the seam the bass
    contribution is gone but vocals/other still carry A's level."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 2.0  # 120bpm
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        # Kill A's bass over the first bar (done by the seam at 2.0s).
        {"tool": "volume_fade", "song": "A", "stem": "bass",
         "start_time": 0.0, "duration_bars": 1.0,
         "start_gain": 1.0, "end_gain": 0.0, "bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    # Pre-seam A sum is vocals+drums+bass+other = 0.2-0.2+0.2+0.2 = 0.4.
    # With bass killed by the seam, the just-before-seam A sum drops by bass's
    # 0.2 to ~0.2 (vocals 0.2 - drums 0.2 + bass 0.0 + other 0.2).
    seam = int(round(sec_per_bar * SR))
    just_before = decoded[seam - 200, 0]
    assert just_before == pytest.approx(0.2, abs=1e-3)


def test_filter_sweep_onset_has_no_transient_click():
    """The filter state must be scaled to the first sample. Unscaled
    sosfilt_zi is the steady state for a unit-step (DC 1.0) input, which
    injected a ringing transient ~4x the signal peak at the sweep onset —
    the click heard at the start of every washout transition."""
    x = (0.05 * np.sin(2 * np.pi * 220.0 * np.arange(SR) / SR)).astype(np.float32)
    audio = np.column_stack((x, x))
    call = {
        "tool": "filter_sweep", "type": "lowpass",
        "start_time": 0.25, "end_time": 0.75,
        "start_cutoff_hz": 20000.0, "end_cutoff_hz": 120.0,
    }
    out = _apply_filter_sweep(audio, call)
    assert float(np.max(np.abs(out))) < 0.08, (
        "filter sweep onset rang above the signal's own scale"
    )


def test_apply_reverb_ramps_wet_in_without_step():
    """The wet/dry mix must ramp in, not hard-switch: at wet_level 0.8 the
    old instant switch dropped the dry signal -14 dB in one sample."""
    n = SR * 2
    x = (0.5 * np.sin(2 * np.pi * 180.0 * np.arange(n) / SR)).astype(np.float32)
    audio = np.column_stack((x, x))
    call = {
        "tool": "apply_reverb", "song": "A", "start_time": 1.0,
        "tail_duration_bars": 2.0, "wet_level": 0.8, "bpm": 120.0,
    }
    out = _apply_reverb(audio, call)
    start = SR
    # At the onset the mix is still ~fully dry — no step.
    assert np.allclose(out[start : start + 64], audio[start : start + 64], atol=2e-3)
    # No sample-to-sample jump beyond what the dry waveform itself has.
    dry_step = float(np.max(np.abs(np.diff(x[start - 200 : start + 200]))))
    out_step = float(np.max(np.abs(np.diff(out[start - 200 : start + 200, 0]))))
    assert out_step <= dry_step * 1.5
    # The wash does arrive once the ramp has opened up.
    assert not np.allclose(out[int(1.9 * SR):], audio[int(1.9 * SR):], atol=1e-2)


def test_render_keeps_a_master_before_fx_onset():
    """An A-side effect (washout's reverb at the seam) must not swap A's
    whole pre-seam body to the stem reconstruction: the untouched master
    (0.3) plays until the effect's onset, not the stem sum (0.4)."""
    a = SongRenderInputs(
        stem_paths=_stems_dict("A"), analysis=_bundle(),
        original_audio_path="orig/A_master.wav",
    )
    b = _inputs(prefix="B")

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.3 * np.ones((N, 2), dtype=np.float32), SR
        sig = 0.2 * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR  # 4-stem sum = 0.4

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "apply_reverb", "song": "A", "start_time": 2.0,
         "tail_duration_bars": 2.0, "wet_level": 0.8, "bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    head = np.mean(np.abs(out[int(0.5 * SR): int(0.6 * SR)]))
    assert abs(head - 0.3) < 0.02, f"A head should be master 0.3, got {head}"


def test_render_a_fx_before_seam_switches_to_processed_at_onset():
    """A stem-scoped fade beginning before the seam (bass_kill) must still
    be audible pre-seam: master until the fade's onset, processed stem-sum
    after it."""
    a = SongRenderInputs(
        stem_paths=_stems_dict("A"), analysis=_bundle(),
        original_audio_path="orig/A_master.wav",
    )
    b = _inputs(prefix="B")

    def _read(path, always_2d=True, dtype="float32"):
        if "master" in path:
            return 0.3 * np.ones((N, 2), dtype=np.float32), SR
        sig = 0.2 * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        # Kill A's bass from 1.0 s, done by 1.5 s (0.25 bars @ 120bpm 4/4).
        {"tool": "volume_fade", "song": "A", "stem": "bass",
         "start_time": 1.0, "duration_bars": 0.25,
         "start_gain": 1.0, "end_gain": 0.0, "bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s, "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "equal_power"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    out, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True, dtype="float32")
    # Before the fade's onset: the master.
    head = np.mean(np.abs(out[int(0.5 * SR): int(0.9 * SR)]))
    assert abs(head - 0.3) < 0.02, f"pre-onset should be master 0.3, got {head}"
    # After the fade completed (bass gone): processed sum 0.2-0.2+0+0.2 = 0.2.
    killed = np.mean(np.abs(out[int(1.7 * SR): int(1.9 * SR)]))
    assert abs(killed - 0.2) < 0.02, f"post-onset should be 0.2, got {killed}"


def test_volume_fade_whole_song_when_no_stem():
    """Without `stem`, volume_fade fades the entire song (all four stems)."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 2.0
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "volume_fade", "song": "A",
         "start_time": 0.0, "duration_bars": 1.0,
         "start_gain": 1.0, "end_gain": 0.0, "bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    seam = int(round(sec_per_bar * SR))
    # Whole A faded to silence by the seam.
    assert decoded[seam - 200, 0] == pytest.approx(0.0, abs=1e-3)


def test_filter_sweep_targets_single_stem_when_stem_set():
    """filter_sweep with `stem` sweeps only that stem (EQ-style bass kill):
    A's bass is an 8 kHz tone that a 120 Hz lowpass silences; the other
    stems are constant and must come through untouched."""
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    sec_per_bar = 2.0  # 120bpm

    def _read(path, always_2d=True, dtype="float32"):
        if "A/bass" in path:
            t = np.arange(N) / SR
            x = (0.2 * np.sin(2 * np.pi * 8000.0 * t)).astype(np.float32)
            return np.column_stack((x, x)), SR
        sig = 0.2 * np.ones((N, 2), dtype=np.float32)
        if "drums" in path:
            sig = -sig
        return sig, SR

    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": sec_per_bar, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "filter_sweep", "song": "A", "stem": "bass",
         "type": "lowpass", "start_time": 0.0, "end_time": 1.0,
         "start_cutoff_hz": 20000.0, "end_cutoff_hz": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_read):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    # After the sweep has closed (1.5-1.9s, still pre-seam): the 8 kHz bass
    # is filtered to ~nothing, leaving vocals+drums+other = 0.2-0.2+0.2 = 0.2.
    region = decoded[int(1.5 * SR): int(1.9 * SR), 0]
    assert np.mean(np.abs(region)) == pytest.approx(0.2, abs=0.02)


def test_apply_backspin_reverses_and_silences():
    """The spin region plays the pre-cut audio backwards (a rising ramp
    becomes a falling one) and everything after the spin is silence."""
    from app.services.mixer.executor import _apply_backspin

    n = SR * 4
    ramp = np.linspace(0.0, 0.8, n, dtype=np.float32)
    audio = np.column_stack((ramp, ramp))
    call = {"tool": "backspin", "song": "A", "start_time": 2.0,
            "duration_beats": 4.0, "bpm": 120.0}  # 4 beats = 2.0 s source
    out = _apply_backspin(audio, call)

    start = 2 * SR
    # Untouched before the cut.
    assert np.array_equal(out[:start], audio[:start])
    # Spin region: starts at the last pre-cut value and falls (reversed).
    spin = out[start: start + int(0.5 * SR), 0]
    assert abs(spin[0] - audio[start - 1, 0]) < 1e-3
    assert spin[0] > spin[-1] > 0.0
    # Hard silence after the spin (~60% of the 2 s source).
    assert np.max(np.abs(out[start + int(1.3 * SR):])) == 0.0


def test_render_backspin_tool_runs_through_pre_fx():
    a = _inputs(prefix="A")
    b = _inputs(prefix="B")
    plan = [
        {"tool": "set_transition_window",
         "from_song_time_start": 2.0, "to_song_time_start": 0.0,
         "duration_bars": 1},
        {"tool": "backspin", "song": "A", "start_time": 2.0,
         "duration_beats": 4.0, "bpm": 120.0},
        *[
            {"tool": "crossfade_stem", "stem": s,
             "from_song": "A", "to_song": "B",
             "start_bar": 0, "duration_bars": 1, "curve": "linear"}
            for s in ("vocals", "drums", "bass", "other")
        ],
    ]
    with patch("soundfile.read", side_effect=_fake_sf_read()):
        result = render(plan, a, b)
    decoded, _ = sf.read(io.BytesIO(result.wav_bytes), always_2d=True)
    assert decoded.shape[0] > 0


def test_render_halftime_ratio_avoids_stretch():
    """tempo_ratio 2.0 on an 85/170 pair means B already interlocks —
    no time-stretch call at all. Without the ratio the same pair would
    demand a 2x stretch."""
    a = _inputs(bpm=85.0, prefix="A")
    b = _inputs(bpm=170.0, prefix="B")

    def _plan(ratio: float | None):
        window = {"tool": "set_transition_window",
                  "from_song_time_start": 0.0, "to_song_time_start": 0.0,
                  "duration_bars": 1}
        if ratio is not None:
            window["tempo_ratio"] = ratio
        return [
            window,
            *[
                {"tool": "crossfade_stem", "stem": s,
                 "from_song": "A", "to_song": "B",
                 "start_bar": 0, "duration_bars": 1, "curve": "linear"}
                for s in ("vocals", "drums", "bass", "other")
            ],
        ]

    with (
        patch("soundfile.read", side_effect=_fake_sf_read()),
        patch("pyrubberband.pyrb.time_stretch",
              side_effect=lambda y, sr, rate: y) as stretch,
    ):
        render(_plan(2.0), a, b)
        assert stretch.call_count == 0   # grids interlock natively

        render(_plan(None), a, b)
        assert stretch.call_count > 0    # 1:1 beatmatch needs the 2x stretch
        assert stretch.call_args[0][2] == pytest.approx(0.5)  # 85/170
