"""Occasion presets + arc templates — the context a human DJ books the
gig with.

Occasions bias the set planner's arc and the per-pair style choices (all
as plain-language prompt context, never hard rules); arc templates are
named energy shapes sampled per pair. Both are optional — an unset queue
plans exactly as before.

Pure data module: no DB, no prompts built here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Occasion:
    id: str
    label: str
    arc_hint: str
    style_bias: str
    energy_floor: float   # 0..1 — ordering/arc guidance, not a hard rule
    energy_ceiling: float
    host_persona: str     # F11 default voice personality
    default_arc: str | None = None


OCCASIONS: dict[str, Occasion] = {
    o.id: o
    for o in (
        Occasion(
            id="house_party", label="House party",
            arc_hint="build steadily to a big peak in the back half, end on "
                     "a singalong high",
            style_bias="favor drop_swap, drum_bridge and acapella moments; "
                       "save double_drop for the peak",
            energy_floor=0.5, energy_ceiling=1.0,
            host_persona="hype", default_arc="peak_late",
        ),
        Occasion(
            id="gym", label="Gym",
            arc_hint="sustained high energy throughout — no long breathers, "
                     "no ambient valleys",
            style_bias="favor drop_swap, stutter_buildup and double_drop; "
                       "avoid wash_out and breakdown_blend",
            energy_floor=0.7, energy_ceiling=1.0,
            host_persona="hype", default_arc="steady_high",
        ),
        Occasion(
            id="dinner", label="Dinner",
            arc_hint="warm, unhurried, conversation-friendly — energy stays "
                     "mid-low and moves gently",
            style_bias="favor smooth_blend and breakdown_blend with long "
                       "windows; no vinyl_stop, backspin or stutter_buildup",
            energy_floor=0.0, energy_ceiling=0.6,
            host_persona="minimal", default_arc="steady_low",
        ),
        Occasion(
            id="study", label="Study / focus",
            arc_hint="flat and unobtrusive; transitions should be invisible",
            style_bias="breakdown_blend and long smooth_blend only; no "
                       "theatrical styles at all",
            energy_floor=0.0, energy_ceiling=0.5,
            host_persona="minimal", default_arc="steady_low",
        ),
        Occasion(
            id="radio", label="Radio show",
            arc_hint="variety over arc — showcase each track, keep momentum",
            style_bias="shorter blends; acapella moments and echo tails read "
                       "great on air",
            energy_floor=0.3, energy_ceiling=0.9,
            host_persona="radio", default_arc="wave",
        ),
        Occasion(
            id="wedding", label="Wedding",
            arc_hint="open warm, peak with crowd-pleasers, land gently",
            style_bias="favor smooth_blend, acapella_out on iconic hooks; "
                       "one theatrical moment max",
            energy_floor=0.4, energy_ceiling=0.95,
            host_persona="radio", default_arc="peak_late",
        ),
        Occasion(
            id="afters", label="Afters / late night",
            arc_hint="deep and hypnotic; long blends, few interruptions, "
                     "slow overall descent",
            style_bias="long smooth_blend, wash_out and breakdown_blend; "
                       "keep vocals sparse",
            energy_floor=0.2, energy_ceiling=0.7,
            host_persona="latenight", default_arc="descend",
        ),
    )
}


@dataclass(frozen=True)
class ArcTemplate:
    id: str
    label: str
    description: str
    # Normalized target energies, resampled to the queue's pair count.
    energy_targets: tuple[float, ...]


ARC_TEMPLATES: dict[str, ArcTemplate] = {
    a.id: a
    for a in (
        ArcTemplate(
            "slow_burn", "Slow burn",
            "start low, one long continuous climb to a final peak",
            (0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
        ),
        ArcTemplate(
            "peak_late", "Peak late",
            "build steadily, peak around three-quarters, ease out",
            (0.4, 0.55, 0.7, 0.9, 1.0, 0.7),
        ),
        ArcTemplate(
            "peak_early", "Peak early",
            "come out swinging, then settle into a groove",
            (0.9, 1.0, 0.8, 0.6, 0.55, 0.5),
        ),
        ArcTemplate(
            "wave", "Wave",
            "alternate lifts and breathers so the floor never tires",
            (0.5, 0.8, 0.55, 0.85, 0.6, 0.95),
        ),
        ArcTemplate(
            "steady_high", "Steady high",
            "pin the energy high and hold it",
            (0.85, 0.9, 0.9, 0.95, 0.9, 0.95),
        ),
        ArcTemplate(
            "steady_low", "Steady low",
            "calm throughout; motion without excitement",
            (0.35, 0.4, 0.35, 0.4, 0.35, 0.4),
        ),
        ArcTemplate(
            "descend", "Wind down",
            "start at the night's energy and glide down to a soft landing",
            (0.8, 0.7, 0.55, 0.45, 0.35, 0.25),
        ),
    )
}


def arc_targets_for_pairs(template_id: str, n_pairs: int) -> list[float] | None:
    """Resample the template's curve to one target per adjacent pair."""
    template = ARC_TEMPLATES.get(template_id)
    if template is None or n_pairs <= 0:
        return None
    src = template.energy_targets
    if n_pairs == 1:
        return [src[len(src) // 2]]
    out = []
    for k in range(n_pairs):
        pos = k * (len(src) - 1) / (n_pairs - 1)
        lo = int(pos)
        hi = min(lo + 1, len(src) - 1)
        frac = pos - lo
        out.append(round(src[lo] * (1 - frac) + src[hi] * frac, 2))
    return out


def occasion_context(
    occasion_id: str | None, vibe_note: str | None
) -> dict[str, str]:
    """Prompt-context lines for the per-pair decision planner."""
    ctx: dict[str, str] = {}
    occ = OCCASIONS.get(occasion_id or "")
    if occ is not None:
        ctx["occasion"] = (
            f"This set is a {occ.label.lower()}: {occ.arc_hint}. "
            f"Style guidance: {occ.style_bias}."
        )
    if vibe_note:
        ctx["vibe_note"] = vibe_note[:300]
    return ctx
