"""Shared types for the mixer service.

Tool-call dicts mirror the spec's Mix Plan Schema. Dataclasses are the
shape the executor's caller assembles from the DB (Song / Analysis /
Stems rows) — keeps the executor pure (no SQLAlchemy import).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

# Tool-call dicts as the LLM (Phase 9) and our hand-built plan generator
# (Phase 7) both emit. The mixer executor walks these in order.

SongRef = Literal["A", "B"]
StemName = Literal["vocals", "drums", "bass", "other"]
# Crossfade curve options. `equal_power` is the industry standard for
# crossfading uncorrelated signals (different songs) — it holds perceived
# loudness flat across the fade by using cos/sin gains, whose squares sum
# to 1.0 at every t. `linear` is correct only for *correlated* signals
# (same source with delay) where the two add coherently. `s_curve` and
# `exponential` are reserved for Phase 9 — the executor refuses them
# with NotImplementedError until the LLM has reason to emit them.
CrossfadeCurve = Literal["equal_power", "linear", "exponential", "s_curve"]
# Filter shape for `filter_sweep`. Gentle 2-pole Butterworth — DJ filters
# taste closer to "fade out the highs" than a resonant synth sweep, so we
# avoid steeper orders that would ring at the cutoff.
FilterType = Literal["lowpass", "highpass"]


class SetTransitionWindow(TypedDict):
    tool: Literal["set_transition_window"]
    from_song_time_start: float  # seconds in A (original time)
    to_song_time_start: float    # seconds in B (original time, pre-stretch)
    duration_bars: int


class PitchShift(TypedDict):
    """Static pitch shift applied to the entire song. Phase 9's LLM may
    emit this for harmonic matching when keys are incompatible. Phase 7's
    hand-built plan generator does NOT emit it — incompatible keys are
    left alone, since pyrubberband artifacts on shifts > 2 semitones
    can be worse than the dissonance they're trying to fix."""

    tool: Literal["pitch_shift"]
    song: SongRef
    semitones: float  # may be fractional; positive=up, negative=down


class TemporaryPitchShift(TypedDict):
    """Time-limited pitch shift that fades in, holds, and fades back to
    the song's original key. Lets the planner introduce a brief key
    excursion (e.g. shift B up 3 semitones at the seam, hold for 4 bars
    of the chorus, then glide back over 8 bars to land on B's true key).

    All bar counts are measured at the planner's target BPM (typically
    A.bpm post-stretch). Total duration of the effect on the audio is
    (fade_in_bars + hold_bars + fade_out_bars) bars at that tempo.

    NOT implemented in Phase 7's executor — Phase 9 territory."""

    tool: Literal["temporary_pitch_shift"]
    song: SongRef
    start_time: float          # seconds in original song time
    semitones: float           # peak shift; fractional allowed
    fade_in_bars: int          # bars to glide from 0 → semitones
    hold_bars: int             # bars to hold at full shift
    fade_out_bars: int         # bars to glide back from semitones → 0


class SetTempoRamp(TypedDict):
    """Gradual tempo change for one song over a time window. Lets the
    planner avoid a hard tempo lock at the seam — e.g. ramp B from its
    original BPM up to A's BPM over the last 8 bars of B's intro, so the
    listener doesn't hear an instantaneous tempo jump when B enters.

    The mixer interpolates the time-stretch rate linearly across the ramp
    window. Outside the window the song plays at the closer endpoint BPM
    (the planner is responsible for ensuring no audible jump at the
    boundaries — typically by anchoring start_bpm to the song's natural
    BPM at the start of the ramp).

    NOT implemented in Phase 7's executor — Phase 9 territory."""

    tool: Literal["set_tempo_ramp"]
    song: SongRef
    start_time: float          # seconds in original song time
    end_time: float            # seconds in original song time
    start_bpm: float           # BPM at start_time
    end_bpm: float             # BPM at end_time


class CrossfadeStem(TypedDict):
    tool: Literal["crossfade_stem"]
    stem: StemName
    from_song: SongRef
    to_song: SongRef
    start_bar: int
    duration_bars: int
    curve: CrossfadeCurve


class FilterSweep(TypedDict):
    """Cutoff-sweeping filter applied to all 4 stems of one song.

    Implements a 2-pole Butterworth that processes the audio in ~10 ms
    blocks; the cutoff frequency is log-interpolated (geomspace) between
    `start_cutoff_hz` and `end_cutoff_hz` across the sweep window. The
    block size is small enough to be perceptually continuous and the
    filter state is carried across blocks (sosfilt_zi) so there are no
    boundary clicks.

    Operates on the per-song stem buffer before the crossfade region is
    built, so the swept signal participates in the existing crossfade
    envelope rather than fighting it.
    """

    tool: Literal["filter_sweep"]
    song: SongRef
    type: FilterType
    start_time: float          # seconds in original song time
    end_time: float            # seconds in original song time
    start_cutoff_hz: float     # Hz; clamped to 20 Hz floor
    end_cutoff_hz: float       # Hz; clamped to 20 Hz floor


class EchoOut(TypedDict):
    """Hard-cut + tempo-locked echo tail on one song.

    At `start_time` the dry signal is silenced and `beats` echoes of the
    pre-cut signal are scheduled at one-beat intervals, each attenuated
    by `feedback ** i` (i = 1..beats). `bpm` is supplied by the caller
    so the executor doesn't need to consult song metadata.

    Operates per-stem and writes a fresh buffer (the dry post-cut signal
    is intentionally dropped — that's the "out" in echo_out).
    """

    tool: Literal["echo_out"]
    song: SongRef
    start_time: float          # seconds in original song time
    beats: int                 # number of echo taps (>= 0; 0 = no-op)
    feedback: float            # 0.0–0.9 per-tap gain
    bpm: float                 # caller-supplied tempo for delay computation


class LoopSection(TypedDict):
    """Beat-locked loop of a fixed slice on one song.

    Slices `beats` worth of audio starting at `start_time`, tiles it
    `repeats` times, and writes the tiled section back in place. A 5 ms
    equal-power crossfade joins each loop boundary to hide any phase
    discontinuity at the slice edges.

    `beats == 0` or `repeats == 0` is a silent no-op.
    """

    tool: Literal["loop_section"]
    song: SongRef
    start_time: float          # seconds in original song time
    beats: float               # length of the loop in beats (can be fractional for stutters)
    repeats: int               # number of times to play the loop
    bpm: float                 # caller-supplied tempo for slice length


class SwapStem(TypedDict):
    """Zero-crossing-aligned hot-swap of one stem between songs.

    At `time` (measured in OUTPUT timeline samples, not original-song
    coordinates) the executor searches ±5 ms in both source stems for
    the closest matched zero-crossing pair, then replaces all samples
    after that boundary with `to_song`'s stem. Used for "drop swaps"
    where a single element of the mix flips to the incoming track.

    If `time` lands past the rendered output length the call is a
    silent no-op.
    """

    tool: Literal["swap_stem"]
    from_song: SongRef
    to_song: SongRef
    stem: StemName
    time: float                # seconds in OUTPUT timeline


class ApplyReverb(TypedDict):
    """Wash-out reverb effect.
    
    Convolves the audio with an exponentially decaying noise burst
    to create a dense, expansive tail. Great for throwing vocals
    or synths into a huge space before a transition drop.
    """
    tool: Literal["apply_reverb"]
    song: SongRef
    start_time: float          # seconds in original song time
    tail_duration_bars: float  # how long the reverb tail takes to decay
    wet_level: float           # 0.0 to 1.0 mix level of the reverb signal
    bpm: float                 # tempo to compute duration


class TurntableStop(TypedDict):
    """Vinyl brake / turntable stop effect.
    
    Simulates a DJ hitting the stop button on a turntable. The playback
    speed (and pitch) decelerates to zero over the specified duration.
    Leaves the stem silent after the stop completes.
    """
    tool: Literal["turntable_stop"]
    song: SongRef
    start_time: float
    duration_bars: float
    bpm: float


class VolumeFade(TypedDict):
    """Standalone volume automation curve.
    
    Fades a song's volume in or out independently of the main A->B
    crossfade. By default it applies to the whole song; set `stem` to target
    a single stem (a true EQ-kill, e.g. drop A's bass 4 bars early so B's
    bass hits harder on the drop).
    """
    tool: Literal["volume_fade"]
    song: SongRef
    start_time: float
    duration_bars: float
    start_gain: float
    end_gain: float
    bpm: float
    # Optional: restrict the fade to one stem. Omit to fade the whole song.
    stem: NotRequired[StemName]


# Phase 9 vocabulary now includes the four DSP additions
# (filter_sweep, echo_out, loop_section, swap_stem). The executor's
# dispatch table covers all of them; only `set_reasoning` (the LLM's
# free-text scratchpad) is still purely informational and ignored.
ToolCall = (
    SetTransitionWindow
    | PitchShift
    | TemporaryPitchShift
    | SetTempoRamp
    | CrossfadeStem
    | FilterSweep
    | EchoOut
    | LoopSection
    | SwapStem
    | ApplyReverb
    | TurntableStop
    | VolumeFade
)
MixPlanJSON = list[dict]  # list[ToolCall] but JSONB persists as plain dicts


@dataclass(frozen=True)
class AnalysisBundle:
    """The slice of an Analysis row the plan generator needs.

    Kept explicit so the generator can be tested with literal fixtures
    instead of building a real Analysis ORM object.
    """

    bpm: float
    key: str            # e.g. "C", "F#m"
    camelot_key: str    # e.g. "8B", "11A"
    time_signature: int
    beat_grid: list[float]
    downbeats: list[float]
    sections: list[dict]   # each has at least {"start": float, "end": float}
    duration: float
    envelopes: dict | None = None
    transcription_segments: list[dict] | None = None
    tags: dict | None = None


@dataclass(frozen=True)
class SongRenderInputs:
    """Per-song inputs the executor needs to load and process audio.

    `stem_paths` map canonical stem names to StorageBackend keys. Keys
    are resolved to filesystem paths via `storage.path(...)`.

    `original_audio_path` is the untouched master WAV. When present, the
    executor uses it (instead of the stem sum) for the parts of the output
    that aren't being actively transitioned — A's body up to the seam, and
    B's body once its tempo/pitch/vocal have all settled back to native.
    """

    stem_paths: dict[str, str]
    analysis: AnalysisBundle
    original_audio_path: str | None = None


@dataclass(frozen=True)
class RenderedTransition:
    """Executor output. Caller persists via StorageBackend.write."""

    wav_bytes: bytes
    sample_rate: int
    duration_seconds: float
    pitch_shift_warning: bool  # true when |δ| > 2 was applied


class MixerPreconditionError(ValueError):
    """Raised when the inputs can't be mixed (wrong sample rate, missing
    stems on disk, etc.). The worker translates this into a `failed`
    MixPlan row with the message in `error_text`."""
