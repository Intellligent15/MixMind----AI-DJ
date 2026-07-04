"""Shared tempo-ramp time-map math.

One ramp, three consumers that must agree to the sample:

  * the executor builds pyrubberband ``timemap_stretch`` maps from it;
  * the stitcher maps mid-song times through it to place junctions;
  * render QA maps the seam through it to find the transition in the
    rendered output.

A "rate" is pyrubberband's convention: playback-speed multiplier
(rate 2.0 = twice as fast = half as long). A ramp plays at
``rate_before`` up to ``start``, glides linearly (sampled at
``NUM_POINTS`` anchors, trapezoid-integrated — exactly the executor's
historical B-ramp math) to ``rate_after`` across ``[start, end]``, and
holds ``rate_after`` after. pyrubberband interpolates linearly between
anchor pairs, so ``map_sample``'s ``np.interp`` over the same pairs is
exact with respect to the rendered audio.
"""

from __future__ import annotations

import numpy as np

NUM_POINTS = 10


def ramp_time_map(
    total: int,
    start: int,
    end: int,
    rate_before: float,
    rate_after: float,
) -> list[tuple[int, int]]:
    """(source_sample, target_sample) anchor pairs for one tempo ramp.

    ``total`` is the source buffer length; ``start``/``end`` are the ramp
    bounds in source samples. Degenerate ramps (end <= start) collapse to
    a rate step at ``start``.
    """
    start = max(0, min(start, total))
    end = max(start, min(end, total))

    pairs: list[tuple[int, int]] = [(0, 0)]
    if start > 0:
        pairs.append((start, int(round(start / rate_before))))
    base_target = start / rate_before

    ramp_len = end - start
    if ramp_len > 0:
        t_source = np.linspace(0, ramp_len, NUM_POINTS)
        rates = np.linspace(rate_before, rate_after, NUM_POINTS)
        t_target = np.zeros_like(t_source)
        for i in range(1, NUM_POINTS):
            dt = t_source[i] - t_source[i - 1]
            avg_rate = (rates[i] + rates[i - 1]) / 2.0
            t_target[i] = t_target[i - 1] + dt / avg_rate
        for s, t in zip(t_source[1:], t_target[1:]):
            pairs.append((int(round(start + s)), int(round(base_target + t))))
        end_target = base_target + t_target[-1]
    else:
        end_target = base_target

    remaining = total - end
    if remaining > 0:
        pairs.append((total, int(round(end_target + remaining / rate_after))))

    # De-duplicate monotonically (rounding can collide adjacent anchors).
    deduped: list[tuple[int, int]] = []
    for s, t in pairs:
        if deduped and s <= deduped[-1][0]:
            continue
        deduped.append((s, t))
    return deduped


def map_sample(pairs: list[tuple[int, int]], source_sample: float) -> int:
    """Source→target through the anchor pairs (linear between anchors,
    matching pyrubberband's interpolation). Clamps outside the map."""
    if not pairs:
        return int(round(source_sample))
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    return int(round(float(np.interp(source_sample, xs, ys))))


def a_ramp_from_plan(plan: list[dict]) -> dict | None:
    """The A-side tempo ramp call, if the plan carries one (tempo
    meet-in-the-middle). B-side ramps are handled separately."""
    return next(
        (c for c in plan
         if c.get("tool") == "set_tempo_ramp" and c.get("song") == "A"),
        None,
    )


def a_output_sample(
    plan: list[dict], a_native_bpm: float | None, t_seconds: float, sr: int
) -> int:
    """Map a time in A's ORIGINAL timeline to the render's output sample,
    accounting for an A-side meet-in-the-middle ramp when present.

    Without an A ramp this is the identity (A is never stretched)."""
    ramp = a_ramp_from_plan(plan)
    src = t_seconds * sr
    if ramp is None or not a_native_bpm:
        return int(round(src))
    end_bpm = float(ramp.get("end_bpm") or a_native_bpm)
    rate_after = end_bpm / a_native_bpm
    start = int(float(ramp.get("start_time", 0.0)) * sr)
    end = int(float(ramp.get("end_time", 0.0)) * sr)
    # `total` only bounds the map; anything comfortably past `end` works
    # for mapping points at/before the ramp end.
    total = max(end, int(src)) + sr
    pairs = ramp_time_map(total, start, end, 1.0, rate_after)
    return map_sample(pairs, src)
