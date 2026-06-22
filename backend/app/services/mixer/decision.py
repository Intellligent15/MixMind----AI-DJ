"""The transition *decision* — planner v2's LLM output contract.

Instead of emitting raw tool calls with hand-computed timestamps (v1),
the model emits this small, discrete decision: which pre-validated seam
candidates to use, which archetype, and a few bounded knobs. The
deterministic expander in `archetypes.py` turns it into an exact,
invariant-satisfying tool-call list.

Pydantic does the heavy lifting on validation, so a malformed model
response raises a clean ValidationError (→ repair / fallback) instead
of producing a broken render.
"""

from __future__ import annotations

import enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TransitionStyle(str, enum.Enum):
    """The archetype library. Keep ids stable — they're persisted on
    MixPlan rows, used in the prompt, and exposed through the API."""

    smooth_blend = "smooth_blend"
    drop_swap = "drop_swap"
    drum_bridge = "drum_bridge"
    wash_out = "wash_out"
    stutter_buildup = "stutter_buildup"
    vinyl_stop = "vinyl_stop"
    acapella_out = "acapella_out"
    acapella_in = "acapella_in"


class TransitionExtra(str, enum.Enum):
    """Optional garnishes an archetype may layer on."""

    bass_kill = "bass_kill"          # kill A's bass early so B's bass hits harder
    filter_sweep_out = "filter_sweep_out"  # lowpass A's tail to nothing
    echo_tail = "echo_tail"          # A cuts with trailing beat echoes
    reverb_tail = "reverb_tail"      # A's last moment washes into reverb


# Per-style allowed crossfade lengths (bars). Short styles are short on
# purpose — a 16-bar drop swap isn't a drop swap.
STYLE_DURATION_CHOICES: dict[TransitionStyle, tuple[int, ...]] = {
    TransitionStyle.smooth_blend: (8, 12, 16),
    TransitionStyle.drop_swap: (2, 4),
    TransitionStyle.drum_bridge: (8, 12, 16),
    TransitionStyle.wash_out: (8, 12, 16),
    TransitionStyle.stutter_buildup: (4, 8),
    TransitionStyle.vinyl_stop: (2, 4),
    TransitionStyle.acapella_out: (8, 12, 16),
    TransitionStyle.acapella_in: (8, 12, 16),
}

STYLE_DESCRIPTIONS: dict[TransitionStyle, str] = {
    TransitionStyle.smooth_blend: (
        "Classic beatmatched blend — all stems crossfade together. The safe, "
        "musical default for two similar-energy tracks."
    ),
    TransitionStyle.drop_swap: (
        "B enters on a drop/chorus: a snappy 2-4 bar swap landing exactly on "
        "the downbeat. Needs a high-energy, vocal-safe IN point on B."
    ),
    TransitionStyle.drum_bridge: (
        "B's drums sneak in early and bridge the two grooves before the rest "
        "of B arrives. Great when both tracks are drum-driven."
    ),
    TransitionStyle.wash_out: (
        "A's tail dissolves into reverb / a closing lowpass filter while B "
        "fades in clean underneath. Great when A is hot and B starts calm, "
        "or for big genre/mood jumps."
    ),
    TransitionStyle.stutter_buildup: (
        "A's last beat stutters (rapid fractional loops) to build tension, "
        "then B drops. Needs a vocal-safe OUT point on A. High-energy move."
    ),
    TransitionStyle.vinyl_stop: (
        "A grinds to a halt like a turntable being stopped, then B starts "
        "fresh. The escape hatch for incompatible tempos or total vibe "
        "changes — use sparingly, it's theatrical."
    ),
    TransitionStyle.acapella_out: (
        "B's instrumental takes over fast at the seam while A's VOCALS keep "
        "riding on top of B's new beat, then hand over to B's vocals at a "
        "phrase boundary. Spine-tingling when A has an iconic vocal and B "
        "is groove-led. REQUIRES compatible keys — never pick on a clash."
    ),
    TransitionStyle.acapella_in: (
        "B's VOCALS arrive immediately over A's still-playing backing track "
        "(a teaser), then A's instrumental swaps to B's later. Great when "
        "B's hook is instantly recognizable. REQUIRES compatible keys — "
        "never pick on a clash — and a vocal-safe OUT point on A."
    ),
}


class TransitionDecision(BaseModel):
    """What the LLM actually returns. Everything else is computed."""

    model_config = ConfigDict(extra="ignore")

    out: str = Field(description="id of the chosen OUT candidate in A, e.g. 'A2'")
    in_: str = Field(alias="in", description="id of the chosen IN candidate in B")
    style: TransitionStyle
    duration_bars: int = Field(ge=2, le=16)
    a_fade_out_bars: int | None = Field(
        default=None, ge=1, le=16,
        description="bars over which A fades to silence; <= duration_bars. "
        "Omit for a fully coupled crossfade.",
    )
    extras: list[TransitionExtra] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=600)

    @field_validator("extras")
    @classmethod
    def _cap_extras(cls, v: list[TransitionExtra]) -> list[TransitionExtra]:
        # One garnish is tasteful; three is mud. Keep the first two.
        return v[:2]

    def normalized_duration(self) -> int:
        """Snap duration_bars to the nearest allowed choice for the style."""
        choices = STYLE_DURATION_CHOICES[self.style]
        return min(choices, key=lambda c: abs(c - self.duration_bars))

    def normalized_a_fade(self, duration_bars: int) -> int:
        if self.a_fade_out_bars is None:
            return duration_bars
        return max(1, min(self.a_fade_out_bars, duration_bars))
