"""Mixer executor — walks a MixPlanJSON and produces a stereo WAV.

Pure-ish: reads stem WAVs from disk through the StorageBackend, calls
pyrubberband for time-stretch / pitch-shift, returns the rendered WAV as
bytes. No DB.

Coordinate convention: tool-call times are in *original-song* time
(seconds in A, seconds in B before any stretch). The executor handles
the original→post-stretch translation internally. This matches what
the LLM planner produces — it sees analyses in original time.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import pyrubberband.pyrb as pyrb
import soundfile as sf
from scipy.signal import fftconvolve, iirfilter, sosfilt, sosfilt_zi

from app.services.mixer.tempo_map import map_sample, ramp_time_map
from app.services.mixer.types import (
    AnalysisBundle,
    MixerPreconditionError,
    MixPlanJSON,
    RenderedTransition,
    SongRenderInputs,
)

logger = logging.getLogger(__name__)

REQUIRED_SAMPLE_RATE = 44100
REQUIRED_CHANNELS = 2
SOFT_CLIP_CEILING = 0.999
# Soft-knee limiter: |sample| <= SOFT_KNEE_THRESHOLD passes through
# untouched; the region above it is tanh-compressed into
# [threshold, SOFT_CLIP_CEILING]. Keeps song bodies (which sit near but
# under full scale) unchanged while taming the crossfade overlap, where
# two near-full-scale masters stack to ~+3 dB.
SOFT_KNEE_THRESHOLD = 0.9
LARGE_SHIFT_THRESHOLD = 2
# Temporary-pitch path: bars over which B's (unshifted) vocal fades back in
# at the tail of the ramp, so it arrives just as B reaches its native key.
VOCAL_FADE_BARS = 2

# filter_sweep: process the buffer in ~10 ms chunks with a log-interpolated
# cutoff. 10 ms is short enough to track a continuous-sounding sweep but
# long enough that the per-block overhead stays negligible.
FILTER_SWEEP_BLOCK_MS = 10.0
# 2-pole Butterworth — DJ-style filters favour gentle slopes over resonant
# peaks; raising the order makes the cutoff "snap" rather than "fade".
FILTER_SWEEP_ORDER = 2
# Audible-floor clamp for filter cutoffs. Below 20 Hz the design is
# effectively a DC block, and zero/negative values are illegal for iirfilter.
FILTER_CUTOFF_FLOOR_HZ = 20.0
# loop_section: hide the seam at each loop boundary with a 5 ms equal-power
# crossfade. Short enough that the rhythmic feel of the loop is preserved.
LOOP_XFADE_MS = 5.0
# swap_stem: search window for a matched zero-crossing on either side of
# the requested swap time, in milliseconds.
SWAP_ZC_SEARCH_MS = 5.0
# apply_reverb: the wet/dry mix ramps from 0 to wet_level over this many
# bars instead of switching instantly. A hard switch steps the dry signal
# down by (1 - wet_level) — -14 dB at wet_level 0.8 — in one sample: an
# audible click plus a sudden volume drop at the effect onset. One bar
# reads as a DJ bringing the reverb send up.
REVERB_WET_RAMP_BARS = 1.0

# Pair-phase alignment (render-time downbeat correction). Per-song downbeat
# detection can still mis-call one track's bar-1; aligning A's seam downbeat to
# B's then locks the two grids a beat or two out of phase (right tempo, wrong
# bar — the subtle "off" feel). At the seam we re-check by trying B at each
# whole-beat offset within the bar and keeping whichever makes B's low-band
# (kick/bass) onset pattern best match A's over a few bars — but only when the
# win over "don't shift" is clear, so a confident detection isn't second-guessed
# on noise.
PAIR_PHASE_WINDOW_BARS = 4
PAIR_PHASE_LOW_BAND_HZ = 250.0
# Required gain in normalized correlation for a non-zero offset to beat the
# no-shift baseline. ~0.12 keeps us from flipping a good alignment on noise.
PAIR_PHASE_MIN_MARGIN = 0.12


def _low_band_onset(seg: np.ndarray, sr: int) -> np.ndarray | None:
    """Mono low-band onset-strength envelope for a (samples, ch) segment.

    Returns None when the segment is too short or too flat (silence / constant
    tone) to carry a usable rhythmic pattern — callers treat that as "can't
    disambiguate, don't shift"."""
    if seg.ndim == 2:
        mono = seg.mean(axis=1).astype(np.float32)
    else:
        mono = seg.astype(np.float32)
    if mono.shape[0] < sr // 4:  # < 0.25 s
        return None
    if float(np.std(mono)) < 1e-5:
        return None
    import librosa

    mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_mels=64)
    freqs = librosa.mel_frequencies(n_mels=64, fmin=0.0, fmax=sr / 2.0)
    low = freqs <= PAIR_PHASE_LOW_BAND_HZ
    if not low.any():
        low[0] = True
    env = librosa.onset.onset_strength(
        S=librosa.power_to_db(mel[low], ref=np.max), sr=sr
    )
    if env.shape[0] < 4 or float(np.std(env)) < 1e-6:
        return None
    return env


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-lag normalized cross-correlation of two 1-D envelopes (-1..1)."""
    n = min(a.shape[0], b.shape[0])
    if n < 4:
        return -1.0
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return -1.0
    return float(np.dot(a, b) / (na * nb))


def _align_pair_phase(
    a_mix: np.ndarray,
    b_mix: np.ndarray,
    a_seam: int,
    b_seam: int,
    samples_per_bar: float,
    time_signature: int,
    sr: int,
) -> int:
    """Return a possibly-corrected ``b_seam`` so B's bar phase matches A's.

    Tries entering B at each whole-beat offset within the bar
    (0..time_signature-1), correlating low-band onset patterns over
    PAIR_PHASE_WINDOW_BARS bars at the seam, and keeps the offset that best
    matches A. Only overrides the analysis-derived seam when it beats the
    no-shift baseline by PAIR_PHASE_MIN_MARGIN — otherwise returns ``b_seam``
    unchanged. Both mixes are in output (post-stretch) time here, so one
    "beat" is ``samples_per_bar / time_signature`` for both songs."""
    if samples_per_bar <= 0 or time_signature <= 1:
        return b_seam
    spb_beat = samples_per_bar / time_signature
    win = int(round(PAIR_PHASE_WINDOW_BARS * samples_per_bar))
    a_env = _low_band_onset(a_mix[a_seam : a_seam + win], sr)
    if a_env is None:
        return b_seam

    base_corr: float | None = None
    best_k, best_corr = 0, -1.0
    for k in range(time_signature):
        bs = b_seam + int(round(k * spb_beat))
        b_env = _low_band_onset(b_mix[bs : bs + win], sr)
        if b_env is None:
            continue
        c = _norm_corr(a_env, b_env)
        if k == 0:
            base_corr = c
        if c > best_corr:
            best_corr, best_k = c, k

    if base_corr is None:
        return b_seam
    if best_k != 0 and (best_corr - base_corr) >= PAIR_PHASE_MIN_MARGIN:
        logger.info(
            "render: pair-phase shifted B by %d beat(s) at the seam "
            "(corr %.3f vs %.3f no-shift)",
            best_k, best_corr, base_corr,
        )
        return b_seam + int(round(best_k * spb_beat))
    return b_seam


def _load_and_sum_stems(inputs: SongRenderInputs) -> np.ndarray:
    """Read all 4 stems from local paths, decode to float32 stereo, sum.

    Returns (samples, 2) float32. Raises MixerPreconditionError if any
    stem disagrees on sample rate or channel count.
    """
    arrays = []
    for stem in ("vocals", "drums", "bass", "other"):
        path = inputs.stem_paths.get(stem)
        if path is None:
            raise MixerPreconditionError(f"missing stem path: {stem!r}")
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        if sr != REQUIRED_SAMPLE_RATE:
            raise MixerPreconditionError(
                f"stem {stem} sample rate {sr} != required {REQUIRED_SAMPLE_RATE}"
            )
        if data.shape[1] != REQUIRED_CHANNELS:
            raise MixerPreconditionError(
                f"stem {stem} channels {data.shape[1]} != required {REQUIRED_CHANNELS}"
            )
        arrays.append(data)
    # Pad to the max length in case stems happen to differ by a sample.
    n = max(a.shape[0] for a in arrays)
    summed = np.zeros((n, REQUIRED_CHANNELS), dtype=np.float32)
    for a in arrays:
        summed[: a.shape[0]] += a
    return summed


def _load_stems_dict(inputs: SongRenderInputs) -> dict[str, np.ndarray]:
    """Load all 4 stems individually, validate sr/channels, pad to a common
    length. Used by the per-stem rendering path so each stem can carry its
    own envelope. Returns ``{stem_name: (samples, 2) float32}``."""
    stems: dict[str, np.ndarray] = {}
    for stem in ("vocals", "drums", "bass", "other"):
        path = inputs.stem_paths.get(stem)
        if path is None:
            raise MixerPreconditionError(f"missing stem path: {stem!r}")
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        if sr != REQUIRED_SAMPLE_RATE:
            raise MixerPreconditionError(
                f"stem {stem} sample rate {sr} != required {REQUIRED_SAMPLE_RATE}"
            )
        if data.shape[1] != REQUIRED_CHANNELS:
            raise MixerPreconditionError(
                f"stem {stem} channels {data.shape[1]} != required {REQUIRED_CHANNELS}"
            )
        stems[stem] = data

    n = max(d.shape[0] for d in stems.values())
    padded: dict[str, np.ndarray] = {}
    for name, arr in stems.items():
        if arr.shape[0] == n:
            padded[name] = arr
        else:
            p = np.zeros((n, REQUIRED_CHANNELS), dtype=np.float32)
            p[: arr.shape[0]] = arr
            padded[name] = p
    return padded


def _stems_sum(stems: dict[str, np.ndarray]) -> np.ndarray:
    """Sum the 4-stem dict into a (samples, 2) array. Assumes stems are
    already pad-equal-length (which `_load_stems_dict` guarantees)."""
    n = max(arr.shape[0] for arr in stems.values())
    out = np.zeros((n, REQUIRED_CHANNELS), dtype=np.float32)
    for arr in stems.values():
        out[: arr.shape[0]] += arr
    return out


def _load_audio(path: str) -> np.ndarray:
    """Read a single (untouched master) WAV to (samples, 2) float32, with the
    same sr/channel validation as the stem loader."""
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    if sr != REQUIRED_SAMPLE_RATE:
        raise MixerPreconditionError(
            f"original audio sample rate {sr} != required {REQUIRED_SAMPLE_RATE}"
        )
    if data.shape[1] != REQUIRED_CHANNELS:
        raise MixerPreconditionError(
            f"original audio channels {data.shape[1]} != required {REQUIRED_CHANNELS}"
        )
    return np.asarray(data, dtype=np.float32)


def _snap_downbeat(t: float, downbeats: list[float]) -> float:
    if not downbeats:
        return t
    for d in downbeats:
        if d >= t:
            return d
    return downbeats[-1]


def _curve_envelopes(curve: str, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (a_gain, b_gain) envelopes for the given crossfade curve.

    `t` is in [0, 1). `a_gain` ramps down (1 → 0); `b_gain` ramps up (0 → 1).

    Curves:
    - "equal_power": cos(π/2·t) for A, sin(π/2·t) for B. Gains-squared sum
      to 1.0 at every t → perceived loudness flat. Industry standard for
      crossfading uncorrelated tracks. **Default for Phase 7.**
    - "linear": (1-t) for A, t for B. Gains sum to 1.0 but gains-squared
      sum to 0.5 at midpoint → audible -3 dB dip. Correct for correlated
      signals only.
    - "exponential", "s_curve": reserved for Phase 9 — raise
      NotImplementedError until the LLM has reason to emit them.
    """
    if curve == "equal_power":
        a_gain = np.cos(t * (np.pi / 2.0)).astype(np.float32)
        b_gain = np.sin(t * (np.pi / 2.0)).astype(np.float32)
        return a_gain, b_gain
    if curve == "linear":
        return (1.0 - t).astype(np.float32), t.astype(np.float32)
    raise NotImplementedError(
        f"crossfade curve {curve!r} not supported in Phase 7 "
        f"(supported: equal_power, linear)"
    )


def _apply_filter_sweep(audio: np.ndarray, call: dict) -> np.ndarray:
    """Sweep a 2-pole Butterworth lowpass/highpass across a window.

    Cutoff is log-interpolated (geomspace) from start_cutoff_hz to
    end_cutoff_hz across the window, in ~10 ms blocks. SOS state is
    carried across blocks so the boundaries don't click. The filter is
    applied from start_time through end-of-buffer (no snap-back).

    `start_cutoff_hz`/`end_cutoff_hz` are clamped to FILTER_CUTOFF_FLOOR_HZ
    (20 Hz) so the iirfilter call is always valid even if the LLM emits 0.
    """
    btype = call["type"]
    start_samp = max(0, int(float(call["start_time"]) * REQUIRED_SAMPLE_RATE))
    sweep_end_samp = min(audio.shape[0], int(float(call["end_time"]) * REQUIRED_SAMPLE_RATE))
    if sweep_end_samp <= start_samp:
        return audio
    start_hz = max(FILTER_CUTOFF_FLOOR_HZ, float(call["start_cutoff_hz"]))
    end_hz = max(FILTER_CUTOFF_FLOOR_HZ, float(call["end_cutoff_hz"]))

    block_samples = max(1, int(FILTER_SWEEP_BLOCK_MS / 1000.0 * REQUIRED_SAMPLE_RATE))
    nyquist = REQUIRED_SAMPLE_RATE / 2.0

    process_end = audio.shape[0]
    window = audio[start_samp:process_end]
    n = window.shape[0]
    sweep_len = sweep_end_samp - start_samp
    sweep_blocks = max(1, (sweep_len + block_samples - 1) // block_samples)
    total_blocks = max(1, (n + block_samples - 1) // block_samples)
    sweep_cutoffs = np.geomspace(start_hz, end_hz, sweep_blocks)
    cutoffs = np.concatenate([
        sweep_cutoffs,
        np.full(max(0, total_blocks - sweep_blocks), end_hz, dtype=float),
    ])

    out = audio.copy()
    zi_per_channel: list[np.ndarray] | None = None
    for i in range(total_blocks):
        wn = float(np.clip(cutoffs[i] / nyquist, 1e-6, 0.999999))
        sos = iirfilter(
            FILTER_SWEEP_ORDER, wn, btype=btype, ftype="butter", output="sos"
        )
        if zi_per_channel is None:
            # sosfilt_zi returns the steady state for a UNIT STEP input;
            # scale it to the actual first sample or the filter starts as
            # if the signal had been pinned at DC 1.0 — an audible click
            # at the sweep onset.
            zi_per_channel = [
                sosfilt_zi(sos) * window[0, ch] for ch in range(window.shape[1])
            ]
        block_start = i * block_samples
        block_end = min(block_start + block_samples, n)
        chunk = window[block_start:block_end]
        filtered = np.empty_like(chunk)
        for ch in range(chunk.shape[1]):
            filtered[:, ch], zi_per_channel[ch] = sosfilt(
                sos, chunk[:, ch], zi=zi_per_channel[ch]
            )
        out[start_samp + block_start : start_samp + block_end] = filtered
    return out


def _apply_echo_out(audio: np.ndarray, call: dict) -> np.ndarray:
    """Hard-cut the dry signal at start_time, schedule N decaying echoes
    of the last beat of audio at one-beat intervals.

    Each tap copies the `delay_samp` samples leading up to the cut,
    attenuated by feedback ** i for i in 1..beats. The dry post-cut
    signal is intentionally not copied through — that's the "out".
    """
    bpm = float(call["bpm"])
    beats = int(call["beats"])
    feedback = float(call["feedback"])
    if beats <= 0 or bpm <= 0:
        return audio
    start_samp = max(0, min(int(float(call["start_time"]) * REQUIRED_SAMPLE_RATE),
                            audio.shape[0]))
    delay_samp = max(1, int(REQUIRED_SAMPLE_RATE * 60.0 / bpm))
    if start_samp < delay_samp:
        return audio.copy()

    out = np.zeros_like(audio)
    out[:start_samp] = audio[:start_samp]
    tap_src = audio[start_samp - delay_samp : start_samp]
    for i in range(1, beats + 1):
        gain = feedback ** i
        tap_start = start_samp + i * delay_samp
        tap_end = min(tap_start + delay_samp, out.shape[0])
        if tap_start >= out.shape[0]:
            break
        usable = tap_end - tap_start
        out[tap_start:tap_end] += (gain * tap_src[:usable]).astype(out.dtype)
    return out
    return out


def _apply_reverb(audio: np.ndarray, call: dict) -> np.ndarray:
    """Wash-out reverb using FFT convolution with decaying noise.

    The wet/dry mix ramps from fully dry to `wet_level` over
    REVERB_WET_RAMP_BARS so the onset is a swell, not a step (see the
    constant's comment)."""
    bpm = float(call.get("bpm", 0.0))
    tail_bars = float(call.get("tail_duration_bars", 0.0))
    wet_level = float(call.get("wet_level", 0.0))
    if tail_bars <= 0 or bpm <= 0 or wet_level <= 0:
        return audio

    start_samp = max(0, min(int(float(call.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE), audio.shape[0]))
    
    decay_sec = tail_bars * (60.0 / bpm) * 4.0
    ir_len = int(decay_sec * REQUIRED_SAMPLE_RATE)
    if ir_len <= 0 or start_samp >= audio.shape[0]:
        return audio
        
    t = np.linspace(0, decay_sec, ir_len, dtype=np.float32)
    tau = decay_sec / 6.0 
    env = np.exp(-t / tau).astype(np.float32)[:, None]
    
    noise = np.random.randn(ir_len, REQUIRED_CHANNELS).astype(np.float32)
    ir = noise * env
    ir /= (np.linalg.norm(ir) + 1e-6)
    
    dry = audio[start_samp:]
    if dry.shape[0] == 0:
        return audio
        
    wet = np.empty_like(dry)
    for ch in range(REQUIRED_CHANNELS):
        wet[:, ch] = fftconvolve(dry[:, ch], ir[:, ch], mode='full')[:dry.shape[0]]

    ramp_len = min(
        int(REVERB_WET_RAMP_BARS * (60.0 / bpm) * 4.0 * REQUIRED_SAMPLE_RATE),
        dry.shape[0],
    )
    mix = np.full(dry.shape[0], wet_level, dtype=np.float32)
    if ramp_len > 0:
        mix[:ramp_len] = np.linspace(
            0.0, wet_level, ramp_len, endpoint=False, dtype=np.float32
        )
    mix = mix[:, None]

    out = audio.copy()
    out[start_samp:] = dry * (1.0 - mix) + wet * mix
    return out


# vocal_tease: edge fades on the teased hook and duck depth for A's
# melodic stem while the foreign vocal rides.
TEASE_EDGE_FADE_S = 0.05
TEASE_DUCK_GAIN = 0.65


def _apply_vocal_tease(
    a_stems: dict[str, np.ndarray], b_vocal: np.ndarray, call: dict
) -> None:
    """Drop a slice of B's vocal hook onto A's timeline (in place).

    The hook is time-stretched to A's tempo (bpm_from -> bpm_to), pitch-
    shifted into A's key when requested, edge-faded, and ADDED to A's
    vocal stem at start_time; A's "other" stem ducks underneath so two
    melodies never fight the riding line. All positions are A-original
    time (pre any meet-in-the-middle stretch), so the tease moves with
    A's map like every other pre-fx.
    """
    start = int(float(call.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE)
    h0 = int(float(call.get("hook_start", 0.0)) * REQUIRED_SAMPLE_RATE)
    h1 = int(float(call.get("hook_end", 0.0)) * REQUIRED_SAMPLE_RATE)
    h0 = max(0, min(h0, b_vocal.shape[0]))
    h1 = max(h0, min(h1, b_vocal.shape[0]))
    a_vocals = a_stems["vocals"]
    if h1 <= h0 or start < 0 or start >= a_vocals.shape[0]:
        return

    clip = np.array(b_vocal[h0:h1], dtype=np.float32)
    bpm_from = float(call.get("bpm_from") or 0.0)
    bpm_to = float(call.get("bpm_to") or 0.0)
    if bpm_from > 0 and bpm_to > 0 and abs(bpm_to - bpm_from) / bpm_from > 1e-3:
        clip = np.asarray(
            pyrb.time_stretch(clip, REQUIRED_SAMPLE_RATE, bpm_to / bpm_from),
            dtype=np.float32,
        )
    semitones = int(call.get("semitones") or 0)
    if semitones:
        clip = np.asarray(
            pyrb.pitch_shift(clip, REQUIRED_SAMPLE_RATE, semitones),
            dtype=np.float32,
        )

    gain = float(call.get("gain", 0.9))
    fade = min(int(TEASE_EDGE_FADE_S * REQUIRED_SAMPLE_RATE), clip.shape[0] // 2)
    if fade > 0:
        clip[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
        clip[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]

    end = min(start + clip.shape[0], a_vocals.shape[0])
    usable = end - start
    if usable <= 0:
        return
    a_vocals[start:end] += gain * clip[:usable]

    # Duck A's melodic stem under the riding hook, with soft edges.
    other = a_stems["other"]
    duck_env = np.full(usable, TEASE_DUCK_GAIN, dtype=np.float32)
    edge = min(fade, usable // 2)
    if edge > 0:
        duck_env[:edge] = np.linspace(1.0, TEASE_DUCK_GAIN, edge, dtype=np.float32)
        duck_env[-edge:] = np.linspace(TEASE_DUCK_GAIN, 1.0, edge, dtype=np.float32)
    other[start:end] *= duck_env[:, None]


def _apply_backspin(audio: np.ndarray, call: dict) -> np.ndarray:
    """The rewind: the audio just before start_time plays REVERSED with
    accelerating speed (like yanking a record backwards), then hard
    silence — the incoming song drops clean at the cut.

    The reversed source is `duration_beats` of audio leading up to
    start_time; the spin occupies ~60% of that (acceleration eats the
    rest), ending in a 5 ms fade so the cut never clicks.
    """
    bpm = float(call.get("bpm", 0.0))
    beats = float(call.get("duration_beats", 0.0))
    if bpm <= 0 or beats <= 0:
        return audio
    start_samp = max(0, min(int(float(call.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE),
                            audio.shape[0]))
    src_len = int(beats * (60.0 / bpm) * REQUIRED_SAMPLE_RATE)
    if src_len <= 0 or start_samp < src_len or start_samp >= audio.shape[0]:
        return audio

    src = audio[start_samp - src_len: start_samp][::-1]
    out_len = min(max(1, int(src_len * 0.6)), audio.shape[0] - start_samp)

    # Quadratic phase: rate starts at 1.0 and accelerates so the whole
    # reversed slice is consumed in out_len samples.
    t = np.arange(out_len, dtype=np.float64)
    alpha = max(0.0, (src_len - out_len) / max(1.0, float(out_len) ** 2))
    phase = t * (1.0 + alpha * t)
    phase = np.clip(phase, 0, src_len - 1)

    out = audio.copy()
    spun = np.empty((out_len, audio.shape[1]), dtype=audio.dtype)
    src_idx = np.arange(src_len, dtype=np.float64)
    for ch in range(audio.shape[1]):
        spun[:, ch] = np.interp(phase, src_idx, src[:, ch].astype(np.float64))
    fade = min(int(0.005 * REQUIRED_SAMPLE_RATE), out_len)
    if fade > 0:
        spun[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)[:, None]
    out[start_samp: start_samp + out_len] = spun
    out[start_samp + out_len:] = 0.0
    return out


def _apply_turntable_stop(audio: np.ndarray, call: dict) -> np.ndarray:
    """Simulates hitting stop on a turntable."""
    duration_bars = float(call.get("duration_bars", 0.0))
    bpm = float(call.get("bpm", 0.0))
    start_time = float(call.get("start_time", 0.0))
    if duration_bars <= 0 or bpm <= 0:
        return audio

    start_samp = max(0, min(int(start_time * REQUIRED_SAMPLE_RATE), audio.shape[0]))
    duration_sec = duration_bars * (60.0 / bpm) * 4.0
    stop_len = int(duration_sec * REQUIRED_SAMPLE_RATE)
    
    if stop_len <= 0 or start_samp >= audio.shape[0]:
        return audio
        
    stop_end = min(start_samp + stop_len, audio.shape[0])
    actual_len = stop_end - start_samp
    
    out = audio.copy()
    out[stop_end:] = 0.0
    
    t = np.arange(actual_len, dtype=np.float64)
    phase = t - (t**2) / (2.0 * actual_len)
    
    for ch in range(REQUIRED_CHANNELS):
        out[start_samp:stop_end, ch] = np.interp(
            phase, 
            np.arange(actual_len, dtype=np.float64), 
            audio[start_samp:stop_end, ch]
        ).astype(np.float32)
        
    return out


def _apply_volume_fade(audio: np.ndarray, call: dict) -> np.ndarray:
    """Standalone volume automation curve."""
    start_gain = float(call.get("start_gain", 1.0))
    end_gain = float(call.get("end_gain", 1.0))
    duration_bars = float(call.get("duration_bars", 0.0))
    bpm = float(call.get("bpm", 0.0))
    start_time = float(call.get("start_time", 0.0))
    if duration_bars <= 0 or bpm <= 0:
        return audio

    start_samp = max(0, min(int(start_time * REQUIRED_SAMPLE_RATE), audio.shape[0]))
    duration_sec = duration_bars * (60.0 / bpm) * 4.0
    fade_len = int(duration_sec * REQUIRED_SAMPLE_RATE)
    
    if fade_len <= 0 or start_samp >= audio.shape[0]:
        return audio
        
    fade_end = min(start_samp + fade_len, audio.shape[0])
    actual_len = fade_end - start_samp
    
    out = audio.copy()
    t = np.linspace(start_gain, end_gain, actual_len, dtype=np.float32)[:, None]
    out[start_samp:fade_end] *= t
    if fade_end < out.shape[0]:
        out[fade_end:] *= end_gain
        
    return out


def _apply_loop_section(audio: np.ndarray, call: dict) -> np.ndarray:
    """Tile a beats-long slice `repeats` times in place.

    Equal-power crossfades the seam between successive loop copies over
    LOOP_XFADE_MS to hide phase discontinuity at slice edges.
    """
    beats = float(call["beats"])
    repeats = int(call["repeats"])
    bpm = float(call["bpm"])
    if beats <= 0 or repeats <= 0 or bpm <= 0:
        return audio
    start_samp = max(0, int(float(call["start_time"]) * REQUIRED_SAMPLE_RATE))
    loop_len = int(beats * (60.0 / bpm) * REQUIRED_SAMPLE_RATE)
    if loop_len <= 0 or start_samp + loop_len > audio.shape[0]:
        return audio
    loop = audio[start_samp : start_samp + loop_len]

    total_len = loop_len * repeats
    write_end = min(start_samp + total_len, audio.shape[0])
    write_len = write_end - start_samp

    xfade = min(int(LOOP_XFADE_MS / 1000.0 * REQUIRED_SAMPLE_RATE), loop_len // 2)
    out = audio.copy()
    tiled = np.zeros((write_len, audio.shape[1]), dtype=audio.dtype)
    for r in range(repeats):
        seg_start = r * loop_len
        seg_end = min(seg_start + loop_len, write_len)
        usable = seg_end - seg_start
        if usable <= 0:
            break
        if r == 0 or xfade == 0:
            tiled[seg_start:seg_end] = loop[:usable]
            continue
        tiled[seg_start:seg_end] = loop[:usable]
        head_xfade = min(xfade, usable)
        t = np.linspace(0.0, 1.0, head_xfade, endpoint=False, dtype=np.float32)
        prev_tail = tiled[seg_start - head_xfade : seg_start].copy()
        cur_head = loop[:head_xfade].copy()
        g_prev = np.cos(t * (np.pi / 2.0))[:, None].astype(audio.dtype)
        g_cur = np.sin(t * (np.pi / 2.0))[:, None].astype(audio.dtype)
        tiled[seg_start - head_xfade : seg_start] = (
            g_prev * prev_tail + g_cur * cur_head
        )

    out[start_samp:write_end] = tiled
    return out


def _find_zero_crossing(audio: np.ndarray, center: int, search_samples: int) -> int:
    """Return the sample index nearest `center` (±search_samples) where
    the mono-summed signal changes sign. Falls back to `center` if none."""
    if audio.shape[0] == 0:
        return center
    lo = max(1, center - search_samples)
    hi = min(audio.shape[0] - 1, center + search_samples)
    if hi <= lo:
        return min(max(center, 0), audio.shape[0] - 1)
    mono = audio[lo - 1 : hi + 1].mean(axis=1)
    signs = np.sign(mono)
    flips = np.where(signs[1:] * signs[:-1] < 0)[0]
    if flips.size == 0:
        return min(max(center, 0), audio.shape[0] - 1)
    absolute = lo + flips
    return int(absolute[np.argmin(np.abs(absolute - center))])


def _soft_knee_limit(
    audio: np.ndarray, threshold: float, ceiling: float
) -> np.ndarray:
    """Stateless soft-knee limiter.

    Samples with ``|x| <= threshold`` pass through unchanged; the region
    above the knee is mapped ``[threshold, inf) -> [threshold, ceiling)``
    via ``tanh``. Because ``tanh'(0) == 1`` the curve joins the unity line
    with a continuous slope at the knee (no audible kink), and the output
    can never reach ``ceiling``. When the whole buffer already sits at or
    under ``ceiling`` there's nothing to clip, so we return it untouched —
    the common case stays bit-exact and, crucially, a single overlap
    transient no longer attenuates the entire track the way the old
    whole-buffer normalize did.
    """
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling:
        return audio
    span = ceiling - threshold
    mag = np.abs(audio)
    over = mag > threshold
    if not np.any(over):
        return audio
    shaped = audio.copy()
    limited = threshold + span * np.tanh((mag[over] - threshold) / span)
    shaped[over] = np.sign(audio[over]) * limited
    return shaped


def render(
    plan: MixPlanJSON,
    a: SongRenderInputs,
    b: SongRenderInputs,
) -> RenderedTransition:
    """Walk `plan` in order, produce a 44.1k stereo WAV.

    Supported tools (Phase 7): set_transition_window, pitch_shift,
    crossfade_stem. Any other tool raises NotImplementedError — Phase 9
    will grow the dispatch table.
    """
    # 1. Load audio per song. A is never pitched/stretched, so when the
    #    untouched master is available we use it directly (no Demucs
    #    reconstruction coloration). Otherwise fall back to the stem sum.
    #    Per-stem A is also loaded so each crossfade_stem call can carry
    #    its own envelope on the A side.
    a_stems = _load_stems_dict(a)
    a_mix = _load_audio(a.original_audio_path) if a.original_audio_path else _stems_sum(a_stems)
    # B is loaded per-stem too: the temporary-pitch path needs the vocal
    # split from the instrumental, AND per-stem envelopes need each stem
    # available independently. b_mix (post-processed sum) is preserved for
    # the original-B-splice tail logic.
    b_stems_raw = _load_stems_dict(b)
    b_vocal = b_stems_raw["vocals"]
    n_b = max(arr.shape[0] for arr in b_stems_raw.values())
    b_instr = np.zeros((n_b, REQUIRED_CHANNELS), dtype=np.float32)
    for stem in ("drums", "bass", "other"):
        b_instr += b_stems_raw[stem]
    b_mix = b_vocal + b_instr
    # B's untouched master, for splicing back in once B has fully settled.
    b_original = _load_audio(b.original_audio_path) if b.original_audio_path else None

    # 2. Walk the plan to extract tools
    window: dict | None = None
    stem_calls: list[dict] = []
    perm_pitch = None
    temp_pitch = None
    tempo_ramp = None       # B-side: post-crossfade glide to native tempo
    a_tempo_ramp = None     # A-side: pre-seam meet-in-the-middle ramp
    # Per-song effects in original time; applied before the stretch / crossfade.
    pre_fx: list[dict] = []
    # Output-time effects (swap_stem); applied after the crossfade is laid down.
    post_fx: list[dict] = []

    for call in plan:
        tool = call["tool"]
        if tool == "set_transition_window":
            window = call
        elif tool == "pitch_shift":
            perm_pitch = call
        elif tool == "temporary_pitch_shift":
            temp_pitch = call
        elif tool == "set_tempo_ramp":
            if call.get("song") == "A":
                a_tempo_ramp = call
            else:
                tempo_ramp = call
        elif tool == "crossfade_stem":
            stem_calls.append(call)
        elif tool in ("filter_sweep", "echo_out", "loop_section", "apply_reverb", "turntable_stop", "backspin", "volume_fade", "vocal_tease"):
            pre_fx.append(call)
        elif tool == "swap_stem":
            post_fx.append(call)
        else:
            raise NotImplementedError(f"unknown tool {tool!r}")

    if window is None:
        raise MixerPreconditionError("plan is missing set_transition_window")

    # 2b. Apply pre-stretch song-local effects to the per-stem buffers, then
    #     rebuild the summed mixes. filter_sweep / echo_out / loop_section
    #     are linear or near-linear; applying per-stem and resumming gives
    #     the same result as applying to the sum, but keeps the per-stem
    #     paths (a_stems / b_stems_raw) available for the crossfade region.
    a_fx_start: int | None = None  # earliest A-side fx onset, in samples
    b_had_fx = False
    for fx in pre_fx:
        song = fx["song"]
        tool = fx["tool"]
        if tool == "vocal_tease":
            # Cross-song effect: B's vocal hook rides on A's timeline.
            _apply_vocal_tease(a_stems, b_stems_raw["vocals"], fx)
            onset = max(0, int(float(fx.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE))
            a_fx_start = onset if a_fx_start is None else min(a_fx_start, onset)
            continue
        target = a_stems if song == "A" else b_stems_raw
        # volume_fade and filter_sweep may target a single stem (true
        # EQ-kill / EQ-sweep, e.g. drop A's bass early or high-pass just
        # the bass out); every other pre-fx applies to the whole song. An
        # invalid/absent stem falls back to all four.
        fx_stem = fx.get("stem")
        names = (
            (fx_stem,)
            if tool in ("volume_fade", "filter_sweep")
            and fx_stem in ("vocals", "drums", "bass", "other")
            else ("vocals", "drums", "bass", "other")
        )
        for name in names:
            if tool == "filter_sweep":
                target[name] = _apply_filter_sweep(target[name], fx)
            elif tool == "echo_out":
                target[name] = _apply_echo_out(target[name], fx)
            elif tool == "loop_section":
                target[name] = _apply_loop_section(target[name], fx)
            elif tool == "apply_reverb":
                target[name] = _apply_reverb(target[name], fx)
            elif tool == "turntable_stop":
                target[name] = _apply_turntable_stop(target[name], fx)
            elif tool == "backspin":
                target[name] = _apply_backspin(target[name], fx)
            elif tool == "volume_fade":
                target[name] = _apply_volume_fade(target[name], fx)
        if song == "A":
            onset = max(0, int(float(fx.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE))
            a_fx_start = onset if a_fx_start is None else min(a_fx_start, onset)
        else:
            b_had_fx = True

    if a_fx_start is not None:
        # A's summed mix is only read before the seam (plus the pair-phase
        # window); the crossfade region reads a_stems directly. So keep the
        # untouched master up to the first effect's onset and splice the
        # processed stem-sum in over a short equal-power fade — replacing
        # the whole buffer would serve the Demucs reconstruction for all of
        # A's pre-seam body just because the transition has an effect.
        processed = _stems_sum(a_stems)
        blended = a_mix.copy()
        n_blend = min(blended.shape[0], processed.shape[0])
        switch = min(a_fx_start, n_blend)
        xf = min(int(0.020 * REQUIRED_SAMPLE_RATE), switch)
        if xf > 0:
            # Ends AT the onset, so the effect starts on a fully-processed
            # path; before the onset the two signals carry the same content
            # (master vs stem reconstruction), so a 20 ms fade is inaudible.
            t = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)[:, None]
            blended[switch - xf : switch] = (
                np.cos(t * (np.pi / 2.0)) * blended[switch - xf : switch]
                + np.sin(t * (np.pi / 2.0)) * processed[switch - xf : switch]
            )
        blended[switch:n_blend] = processed[switch:n_blend]
        a_mix = blended
    if b_had_fx:
        b_vocal = b_stems_raw["vocals"]
        b_instr = b_stems_raw["drums"] + b_stems_raw["bass"] + b_stems_raw["other"]
        b_mix = b_vocal + b_instr

    # 2c. Tempo meet-in-the-middle: an A-side ramp means A glides to the
    #     meet tempo over its last bars before the seam, so the crossfade
    #     runs at mid-tempo and B only stretches the other half of the
    #     gap. A's stems and master all go through the same time map;
    #     A-timeline positions at/after the ramp must be translated via
    #     _a_out_sample() from here on.
    a_meet_bpm = a.analysis.bpm
    a_time_pairs: list[tuple[int, int]] | None = None
    if a_tempo_ramp and a.analysis.bpm:
        end_bpm = float(a_tempo_ramp.get("end_bpm") or a.analysis.bpm)
        a_rate_after = end_bpm / a.analysis.bpm
        if abs(1.0 - a_rate_after) > 1e-6:
            a_meet_bpm = end_bpm
            ramp_start = int(float(a_tempo_ramp.get("start_time", 0.0)) * REQUIRED_SAMPLE_RATE)
            ramp_end = int(float(a_tempo_ramp.get("end_time", 0.0)) * REQUIRED_SAMPLE_RATE)
            a_time_pairs = ramp_time_map(
                a_mix.shape[0], ramp_start, ramp_end, 1.0, a_rate_after
            )

            def _stretch_a(audio: np.ndarray) -> np.ndarray:
                pairs = ramp_time_map(
                    audio.shape[0], ramp_start, ramp_end, 1.0, a_rate_after
                )
                return np.asarray(
                    pyrb.timemap_stretch(audio, REQUIRED_SAMPLE_RATE, pairs),
                    dtype=np.float32,
                )

            a_stems = {name: _stretch_a(arr) for name, arr in a_stems.items()}
            a_mix = _stretch_a(a_mix)

    def _a_out_sample(src_samp: float) -> int:
        """A original-timeline sample -> output sample (identity unless
        the meet-in-the-middle ramp stretched A)."""
        if a_time_pairs is None:
            return int(round(src_samp))
        return map_sample(a_time_pairs, src_samp)

    # 3. Build B's time-stretch once, apply on demand. The map depends only
    #    on sample positions (not content), so the same stretch applies to
    #    the full mix or to the vocal / instrumental split.
    #    tempo_ratio (window, default 1.0) is the half/double-time trick:
    #    B is beatmatched toward a.bpm * ratio so 85 <-> 170 BPM pairs
    #    interlock at 2:1 instead of suffering a 2x stretch. With a
    #    meet-in-the-middle ramp the crossfade tempo is a_meet_bpm.
    tempo_ratio = float(window.get("tempo_ratio") or 1.0)
    rate_A = (a_meet_bpm * tempo_ratio) / b.analysis.bpm
    stretch_factor = 1.0 / rate_A             # how much longer B becomes
    b_total = b_mix.shape[0]

    time_map: list[tuple[int, int]] | None = None
    # Post-ramp, B plays at native rate (1.0), so output and original-B
    # samples advance 1:1 with a constant offset: orig = output - offset.
    # Captured here for splicing B's untouched master back into the tail.
    b_tail_offset: int | None = None
    b_native_out: int | None = None  # output sample where B reaches native tempo
    if tempo_ramp:
        start_samp = int(tempo_ramp["start_time"] * REQUIRED_SAMPLE_RATE)
        end_samp = int(tempo_ramp["end_time"] * REQUIRED_SAMPLE_RATE)
        time_map = [(0, 0)]
        if start_samp > 0:
            time_map.append((start_samp, int(start_samp / rate_A)))

        ramp_len = end_samp - start_samp
        if ramp_len > 0:
            num_points = 10
            t_source = np.linspace(0, ramp_len, num_points)
            rates = np.linspace(rate_A, 1.0, num_points)
            t_target = np.zeros_like(t_source)
            for i in range(1, num_points):
                dt = t_source[i] - t_source[i - 1]
                avg_rate = (rates[i] + rates[i - 1]) / 2.0
                t_target[i] = t_target[i - 1] + dt / avg_rate

            for s, t in zip(t_source[1:], t_target[1:]):
                time_map.append((int(start_samp + s), int(start_samp / rate_A + t)))
            ramp_end_target = int(start_samp / rate_A + t_target[-1])
        else:
            ramp_end_target = int(start_samp / rate_A)
        b_tail_offset = ramp_end_target - end_samp
        b_native_out = ramp_end_target

        last_source = time_map[-1][0]
        last_target = time_map[-1][1]
        remaining = b_total - last_source
        if remaining > 0:
            time_map.append((b_total, int(last_target + remaining)))

    def _stretch(audio: np.ndarray) -> np.ndarray:
        if time_map is not None:
            return np.asarray(
                pyrb.timemap_stretch(audio, REQUIRED_SAMPLE_RATE, time_map),
                dtype=np.float32,
            )
        if abs(1.0 - stretch_factor) > 1e-6:
            return np.asarray(
                pyrb.time_stretch(audio, REQUIRED_SAMPLE_RATE, rate_A),
                dtype=np.float32,
            )
        return audio

    # 4. Pitch shift B
    pitch_shift_warning = False
    # Output sample at which B is fully back to native (tempo, pitch, and
    # vocal) — i.e. where the untouched master can be spliced in. None means
    # "don't splice" (permanent pitch shift, or no tempo ramp to align to).
    b_settle_out: int | None = None

    def apply_pitch(audio, semitones):
        nonlocal pitch_shift_warning
        if abs(semitones) > LARGE_SHIFT_THRESHOLD:
            logger.warning(
                "render: large pitch shift applied (n_steps=%d); expect artifacts",
                semitones,
            )
            pitch_shift_warning = True
        return np.asarray(
            pyrb.pitch_shift(audio, REQUIRED_SAMPLE_RATE, semitones),
            dtype=np.float32,
        )

    # b_stems_processed holds the per-stem post-processed audio that the
    # per-stem crossfade region will read from. The summed b_mix is kept in
    # parallel so the original-master splice (which lives on the summed
    # output) can still work and so legacy/no-envelope-variation behavior
    # stays bit-identical when all 4 stems share an envelope.
    b_stems_processed: dict[str, np.ndarray] = {}
    if perm_pitch and int(perm_pitch["semitones"]) != 0:
        # Permanent shift: vocal is shifted with the rest and stays present.
        # No vocal muting (the song now lives in the new key for good).
        b_mix = apply_pitch(_stretch(b_mix), int(perm_pitch["semitones"]))
        for stem in ("vocals", "drums", "bass", "other"):
            b_stems_processed[stem] = apply_pitch(
                _stretch(b_stems_raw[stem]), int(perm_pitch["semitones"])
            )
    else:
        # A temporary *pitch* shift mutes B's vocal until the key settles —
        # a vocal sung in the wrong key is the ugliest artifact of a
        # transition, and our vocal source is the Demucs stem (colored
        # already), so we hide it until B is back in its own key. A tempo
        # ramp ALONE does NOT mute the vocal: the ramp now lands after the
        # crossfade (system prompt rule 7), and B is a clean constant-tempo
        # stretch through the crossfade, so the vocal rides in with the
        # blend instead of being held out for tens of seconds until the ramp
        # settles (which made B's vocal arrive jarringly late).
        instr_native_per_stem = {
            s: _stretch(b_stems_raw[s]) for s in ("drums", "bass", "other")
        }
        instr_native = sum(instr_native_per_stem.values())  # type: ignore[assignment]
        vocal_native = _stretch(b_vocal)
        sec_per_bar_b = (60.0 / b.analysis.bpm) * b.analysis.time_signature

        # `vocal_resume_out` is the output sample where the vocal starts
        # fading back in. It's the latest moment a temp transition is still
        # active: max of the pitch-return end and the tempo-ramp end.
        vocal_resume_out = 0

        instr = instr_native
        instr_per_stem = dict(instr_native_per_stem)
        if temp_pitch and int(temp_pitch["semitones"]) != 0:
            # Pitch the instrumental, then crossfade it back to native key
            # over fade_out_bars. Vocal is left unshifted (never sung in
            # the wrong key) and stays muted until the key has settled.
            semitones = int(temp_pitch["semitones"])
            return_bars = float(temp_pitch.get("fade_out_bars") or VOCAL_FADE_BARS)
            return_len = int(return_bars * sec_per_bar_b * REQUIRED_SAMPLE_RATE)
            fade_start = max(
                0, min(int(temp_pitch["start_time"] * REQUIRED_SAMPLE_RATE / rate_A),
                       instr_native.shape[0])
            )
            fade_end = min(fade_start + return_len, instr_native.shape[0])
            xlen = fade_end - fade_start
            t_x = (
                np.linspace(0.0, 1.0, xlen, endpoint=False, dtype=np.float32)
                if xlen > 0 else None
            )

            def _shift_and_blend(native: np.ndarray) -> np.ndarray:
                shifted = apply_pitch(native, semitones)
                out = np.copy(shifted)
                if t_x is not None:
                    out[fade_start:fade_end] = (
                        np.cos(t_x * (np.pi / 2.0))[:, None] * shifted[fade_start:fade_end]
                        + np.sin(t_x * (np.pi / 2.0))[:, None] * native[fade_start:fade_end]
                    )
                out[fade_end:] = native[fade_end:]
                return out

            instr_per_stem = {
                s: _shift_and_blend(instr_native_per_stem[s])
                for s in ("drums", "bass", "other")
            }
            instr = sum(instr_per_stem.values())  # type: ignore[assignment]
            vocal_resume_out = max(vocal_resume_out, fade_end)

        # NOTE: a tempo ramp on its own intentionally does NOT push
        # vocal_resume_out — see the block comment above. Only a temp pitch
        # shift gates the vocal. The master splice below still waits for the
        # ramp to land (b_settle_out is clamped to b_native_out).

        n = min(instr.shape[0], vocal_native.shape[0])
        if vocal_resume_out > 0:
            vocal_fade_len = int(VOCAL_FADE_BARS * sec_per_bar_b * REQUIRED_SAMPLE_RATE)
            vocal_gain = np.zeros(n, dtype=np.float32)
            v_start = min(vocal_resume_out, n)
            v_end = min(v_start + vocal_fade_len, n)
            if v_end > v_start:
                vt = np.linspace(0.0, 1.0, v_end - v_start, endpoint=False, dtype=np.float32)
                vocal_gain[v_start:v_end] = np.sin(vt * (np.pi / 2.0))
            vocal_gain[v_end:] = 1.0
            b_mix = instr[:n] + vocal_native[:n] * vocal_gain[:, None]
            for s in ("drums", "bass", "other"):
                b_stems_processed[s] = instr_per_stem[s][:n]
            b_stems_processed["vocals"] = vocal_native[:n] * vocal_gain[:, None]
            # Fully settled once the vocal has finished fading in — and, if a
            # tempo ramp is running, not before B reaches native tempo (the
            # spliced master is native-rate, so it can't replace the mix
            # mid-ramp).
            b_settle_out = v_end
            if tempo_ramp and b_native_out is not None:
                b_settle_out = max(b_settle_out, b_native_out)
        else:
            # No temporary transition: vocal stays present from the start.
            b_mix = instr[:n] + vocal_native[:n]
            for s in ("drums", "bass", "other"):
                b_stems_processed[s] = instr_per_stem[s][:n]
            b_stems_processed["vocals"] = vocal_native[:n]
            if perm_pitch is None:
                b_settle_out = b_native_out

    # 4b. Splice B's untouched master back in once it's fully settled, so the
    #     body isn't the Demucs stem-sum reconstruction. Only when we have the
    #     original AND a native-rate tail to align it to (orig = output -
    #     offset). A short equal-power crossfade hides any sub-frame drift.
    if b_original is not None and b_settle_out is not None and b_tail_offset is not None:
        splice_out = max(0, min(b_settle_out, b_mix.shape[0]))
        splice_orig = splice_out - b_tail_offset
        tail_len = b_mix.shape[0] - splice_out
        if 0 <= splice_orig < b_original.shape[0] and tail_len > 0:
            orig_tail = b_original[splice_orig : splice_orig + tail_len]
            L = orig_tail.shape[0]
            if L > 0:
                b_mix = b_mix.copy()
                xf = min(int(0.050 * REQUIRED_SAMPLE_RATE), L)
                if xf > 0:
                    t = np.linspace(0.0, 1.0, xf, endpoint=False, dtype=np.float32)
                    g_stem = np.cos(t * (np.pi / 2.0))[:, None]
                    g_orig = np.sin(t * (np.pi / 2.0))[:, None]
                    b_mix[splice_out : splice_out + xf] = (
                        g_stem * b_mix[splice_out : splice_out + xf]
                        + g_orig * orig_tail[:xf]
                    )
                b_mix[splice_out + xf : splice_out + L] = orig_tail[xf:L]

    if len(stem_calls) != 4:
        raise MixerPreconditionError(
            f"plan must have 4 crossfade_stem calls, got {len(stem_calls)}"
        )
    stems_seen = {c["stem"] for c in stem_calls}
    if stems_seen != {"vocals", "drums", "bass", "other"}:
        raise MixerPreconditionError(
            "crossfade_stem calls must cover vocals/drums/bass/other"
        )

    # 4. Resolve seam downbeats. A's downbeats are unchanged (A isn't
    #    stretched); B's seam in original time → post-stretch by 1/rate.
    a_seam = _snap_downbeat(
        float(window["from_song_time_start"]), a.analysis.downbeats
    )
    b_seam_orig = _snap_downbeat(
        float(window["to_song_time_start"]), b.analysis.downbeats
    )
    b_seam_post = b_seam_orig * stretch_factor

    # The seam in OUTPUT samples: identity unless the meet-in-the-middle
    # ramp stretched A's tail (then the map translates it).
    a_seam_sample = _a_out_sample(a_seam * REQUIRED_SAMPLE_RATE)
    b_seam_sample = int(round(b_seam_post * REQUIRED_SAMPLE_RATE))

    # Bars inside/after the window run at the crossfade tempo — A's
    # native bpm normally, the meet tempo when an A-ramp is present.
    sec_per_bar_a = (60.0 / a_meet_bpm) * a.analysis.time_signature
    samples_per_bar_a = sec_per_bar_a * REQUIRED_SAMPLE_RATE

    # 4c. Pair-phase correction. A's and B's downbeats are aligned at the seam,
    #     but if either song's bar-1 was mis-detected the grids lock a beat or
    #     two out of phase. Re-check against A's actual low-band onsets at the
    #     seam and nudge B onto the matching bar phase (no-op unless the win is
    #     clear). Both mixes are in output time, so beats are aligned at the
    #     same tempo already — this only fixes the bar-level phase.
    b_seam_sample = _align_pair_phase(
        a_mix,
        b_mix,
        a_seam_sample,
        b_seam_sample,
        samples_per_bar_a,
        a.analysis.time_signature,
        REQUIRED_SAMPLE_RATE,
    )

    # 5. Per-stem crossfade windows. Each call places its own envelope at
    #    [seam + start_bar*bar, seam + (start_bar + duration_bars)*bar]
    #    in output coords. Stems can start and end at different times to
    #    support stem-swap moves (e.g. vocal fades out before drums).
    #
    #    Bounds: the plan generator clamps based on Song.duration_seconds
    #    (from yt-dlp metadata), but the *actual* stem WAV length can drift
    #    by a few hundred ms — Demucs pads/trims, pyrubberband's stretch
    #    adds/removes a handful of samples, yt-dlp rounds. Clamp each
    #    stem's window gracefully rather than refusing the render. Only
    #    raise when there's no overlap at all to work with (real bug:
    #    plan picked a seam past end-of-audio).
    if a_mix.shape[0] - a_seam_sample <= 0 or b_mix.shape[0] - b_seam_sample <= 0:
        raise MixerPreconditionError(
            f"no overlap available for crossfade "
            f"(a_seam={a_seam_sample}/len_a={a_mix.shape[0]}, "
            f"b_seam={b_seam_sample}/len_b={b_mix.shape[0]})"
        )

    # Use the latest end across all stems to size the output (so a stem
    # whose envelope finishes later still has somewhere to write).
    requested_ends = [
        int(round((int(c["start_bar"]) + int(c["duration_bars"])) * samples_per_bar_a))
        for c in stem_calls
    ]
    max_end_offset = max(requested_ends)
    # Clamp to available room across both songs.
    available_xf = min(a_mix.shape[0] - a_seam_sample, b_mix.shape[0] - b_seam_sample)
    if max_end_offset > available_xf:
        logger.warning(
            "render: clamping crossfade end from %d to %d samples "
            "(%.3fs -> %.3fs) — stem WAV length drifted from "
            "Song.duration_seconds by ~%d samples",
            max_end_offset,
            available_xf,
            max_end_offset / REQUIRED_SAMPLE_RATE,
            available_xf / REQUIRED_SAMPLE_RATE,
            max_end_offset - available_xf,
        )
        max_end_offset = available_xf

    # 6. Build output buffer.
    out_len = a_seam_sample + (b_mix.shape[0] - b_seam_sample)
    out = np.zeros((out_len, REQUIRED_CHANNELS), dtype=np.float32)

    # Pre-seam: pure A (uses the original master when available; otherwise
    # the 4-stem sum).
    out[:a_seam_sample] = a_mix[:a_seam_sample]

    # Per-stem crossfade. For each stem S with (start_bar, duration_bars,
    # curve):
    #   - In [seam, seam + start_bar*bar): pure A's stem S (B not yet in).
    #   - In [seam + start_bar*bar, seam + (start_bar+dur)*bar): the cross-
    #     fade — a_gain * a_stem_S + b_gain * b_stem_S.
    #   - In [seam + (start_bar+dur)*bar, latest_end): pure B's stem S
    #     (A out, B in).
    # Past `latest_end`: handled in one shot below using b_mix so the
    # original-master splice keeps applying.
    #
    # endpoint=False is load-bearing for each stem's t array (same reason
    # as the previous shared-envelope path): keeps the last crossfade
    # sample slightly A-weighted so it joins the pure-B section after it
    # without a 1-frame discontinuity.
    for call in stem_calls:
        stem = call["stem"]
        start_off = int(round(int(call["start_bar"]) * samples_per_bar_a))
        end_off = int(round(
            (int(call["start_bar"]) + int(call["duration_bars"])) * samples_per_bar_a
        ))
        # Clamp this stem's window to what's actually available, but
        # leave room for at least one sample of crossfade so the math
        # doesn't degenerate.
        start_off = max(0, min(start_off, max_end_offset))
        end_off = max(start_off, min(end_off, max_end_offset))
        xf_len = end_off - start_off
        curve = call["curve"]

        # Optional decoupled A fade-out. By default A fades out over the
        # full B fade-in window (a classic coupled crossfade). When
        # `a_fade_out_bars` < `duration_bars`, A reaches silence earlier
        # and B keeps swelling on its own for the remainder — i.e. cut A
        # out without shortening B's entrance. Defaults to duration_bars
        # so omitting it is bit-identical to the coupled crossfade.
        afb = call.get("a_fade_out_bars")
        a_fade_bars = int(afb) if afb is not None else int(call["duration_bars"])
        a_fade_off = int(round(
            (int(call["start_bar"]) + a_fade_bars) * samples_per_bar_a
        ))
        a_fade_off = max(start_off, min(a_fade_off, end_off))

        out_xf_start = a_seam_sample + start_off
        out_xf_end = a_seam_sample + end_off
        out_post_end = a_seam_sample + max_end_offset

        a_stem = a_stems[stem]
        b_stem = b_stems_processed[stem]

        # Pre-window (within this stem's post-seam region): pure A stem.
        if start_off > 0:
            pre_a = a_stem[a_seam_sample : a_seam_sample + start_off]
            out[a_seam_sample : a_seam_sample + start_off] += pre_a
        # Crossfade window.
        if xf_len > 0:
            t = np.linspace(0.0, 1.0, xf_len, endpoint=False, dtype=np.float32)
            # B fades in over the full window. A fades out over its own
            # (possibly shorter) sub-window, then stays silent so B keeps
            # rising alone.
            _, b_gain = _curve_envelopes(curve, t)
            a_gain = np.zeros(xf_len, dtype=np.float32)
            a_fade_len = max(0, min(a_fade_off - start_off, xf_len))
            if a_fade_len > 0:
                t_a = np.linspace(0.0, 1.0, a_fade_len, endpoint=False, dtype=np.float32)
                a_env, _ = _curve_envelopes(curve, t_a)
                a_gain[:a_fade_len] = a_env
            a_region = a_stem[a_seam_sample + start_off : a_seam_sample + end_off]
            b_region = b_stem[b_seam_sample + start_off : b_seam_sample + end_off]
            # a_region/b_region may be shorter than xf_len if the source
            # array ran out — trim to the shorter side and apply.
            L = min(a_region.shape[0], b_region.shape[0], xf_len)
            if L > 0:
                out[out_xf_start : out_xf_start + L] += (
                    a_gain[:L, None] * a_region[:L]
                    + b_gain[:L, None] * b_region[:L]
                )
        # Post-window (still inside [seam, seam+max_end_offset)): pure B
        # stem. Stems that ended earlier contribute B audio here; stems
        # still mid-crossfade have already covered this range above.
        post_len = out_post_end - out_xf_end
        if post_len > 0:
            b_post = b_stem[b_seam_sample + end_off : b_seam_sample + end_off + post_len]
            out[out_xf_end : out_xf_end + b_post.shape[0]] += b_post

    # After every stem has finished its crossfade, the output is pure B —
    # but we want to use the SUMMED b_mix here (not the per-stem sum) so
    # the original-master splice (if any) still applies.
    tail_out_start = a_seam_sample + max_end_offset
    tail_b_start = b_seam_sample + max_end_offset
    if tail_out_start < out_len:
        tail_len = min(out_len - tail_out_start, b_mix.shape[0] - tail_b_start)
        if tail_len > 0:
            out[tail_out_start : tail_out_start + tail_len] = (
                b_mix[tail_b_start : tail_b_start + tail_len]
            )

    # 6b. swap_stem: replace the output tail after `time` with one specific
    #     stem of `to_song`. Times are in OUTPUT-timeline samples. The
    #     boundary is snapped to the nearest matched zero-crossing in both
    #     source stems (±5 ms) to keep the splice click-free.
    for fx in post_fx:
        target_samp = int(float(fx["time"]) * REQUIRED_SAMPLE_RATE)
        if target_samp <= 0 or target_samp >= out.shape[0]:
            continue
        stem_name = fx["stem"]
        to_song = fx["to_song"]
        from_song = fx["from_song"]

        def _load_stem(song: str, stem: str) -> np.ndarray:
            inp = a if song == "A" else b
            data, sr = sf.read(inp.stem_paths[stem], always_2d=True, dtype="float32")
            if sr != REQUIRED_SAMPLE_RATE or data.shape[1] != REQUIRED_CHANNELS:
                raise MixerPreconditionError(
                    f"swap_stem source {song}/{stem} sr/channels mismatch"
                )
            return np.asarray(data, dtype=np.float32)

        from_stem = _load_stem(from_song, stem_name)
        to_stem = _load_stem(to_song, stem_name)
        # B stems live in original time and must be advanced through the same
        # stretch the rest of B went through.
        if to_song == "B":
            to_stem = _stretch(to_stem)
        if from_song == "B":
            from_stem = _stretch(from_stem)

        search = int(SWAP_ZC_SEARCH_MS / 1000.0 * REQUIRED_SAMPLE_RATE)
        from_zc = _find_zero_crossing(from_stem, target_samp, search)
        to_zc = _find_zero_crossing(to_stem, target_samp, search)
        boundary = from_zc if abs(from_zc - target_samp) <= abs(to_zc - target_samp) else to_zc
        boundary = max(0, min(boundary, out.shape[0]))
        tail_len = min(out.shape[0] - boundary, to_stem.shape[0] - boundary)
        if tail_len > 0:
            out[boundary : boundary + tail_len] = to_stem[boundary : boundary + tail_len]
            if boundary + tail_len < out.shape[0]:
                out[boundary + tail_len :] = 0.0

    # 7. Soft-knee limit to SOFT_CLIP_CEILING. Unlike the old whole-buffer
    #    normalize (which ducked the entire track — and so every transition
    #    by a different amount — whenever one overlap transient peaked),
    #    this leaves song bodies untouched and only tames samples above the
    #    knee, keeping loudness consistent across the stitched queue.
    peak = float(np.max(np.abs(out)))
    if peak > SOFT_CLIP_CEILING:
        logger.info(
            "render: soft-knee limiting output (peak=%.4f, knee=%.3f, ceiling=%.3f)",
            peak, SOFT_KNEE_THRESHOLD, SOFT_CLIP_CEILING,
        )
        out = _soft_knee_limit(out, SOFT_KNEE_THRESHOLD, SOFT_CLIP_CEILING)

    # 8. Encode to 16-bit PCM WAV in memory.
    buf = io.BytesIO()
    sf.write(buf, out, REQUIRED_SAMPLE_RATE, format="WAV", subtype="PCM_16")
    duration_seconds = out.shape[0] / REQUIRED_SAMPLE_RATE
    return RenderedTransition(
        wav_bytes=buf.getvalue(),
        sample_rate=REQUIRED_SAMPLE_RATE,
        duration_seconds=duration_seconds,
        pitch_shift_warning=pitch_shift_warning,
    )
