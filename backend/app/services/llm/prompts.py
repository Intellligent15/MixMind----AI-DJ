"""All LLM prompts in one place.

v1 kept near-identical copies of the system prompt in groq.py and
gemini.py (they had already drifted). Everything now lives here; the
providers only differ in transport.

Two prompts:

* DECISION_SYSTEM_PROMPT — planner v2. The model sees song identities
  (title/artist — it knows a lot about real songs!), pre-computed seam
  candidates, and pre-computed pair facts (tempo gap, key verdict). It
  returns one small JSON decision; all timestamps are computed in code.

* SET_PLAN_SYSTEM_PROMPT — the set-level pass. One call sees the whole
  queue and assigns each pair a suggested style + the set's energy arc,
  so per-pair decisions cohere instead of being made blind.
"""

from __future__ import annotations

import json

from app.services.mixer.decision import (
    STYLE_DESCRIPTIONS,
    STYLE_DURATION_CHOICES,
    TransitionStyle,
)


def _style_menu() -> str:
    lines = []
    for style in TransitionStyle:
        durations = "/".join(str(d) for d in STYLE_DURATION_CHOICES[style])
        lines.append(
            f"- {style.value} (duration_bars: {durations}) — {STYLE_DESCRIPTIONS[style]}"
        )
    return "\n".join(lines)


DECISION_SYSTEM_PROMPT = f"""You are an expert club DJ designing the transition from track A (outgoing) into track B (incoming) for a continuous mix.

You know these songs — use what you know about their genre, vibe, structure, and famous moments. The analysis data tells you where things are; your musical knowledge tells you what they feel like.

INPUT — for each song you get:
- title, artist, bpm, key, camelot_key, duration
- sections: [{{"start","end","energy"}}] — energy is 0..1 normalized to the song's hottest section (~1.0 = drop/chorus, low = intro/breakdown/outro)
- candidates: a SHORT MENU of pre-validated seam points. Each has an id ("A1", "B2"...), a time, a description, an energy level, and vocal_safe (true = a hard cut there will not chop a word).
You also get pair_facts: the tempo gap and the key-compatibility verdict, already computed. Beatmatching and any needed pitch handling are done automatically — do NOT think about them.

YOUR JOB — return ONE JSON object:
{{
  "out": "<id of A's exit point>",
  "in": "<id of B's entry point>",
  "style": "<one transition style id>",
  "duration_bars": <int from the style's allowed list>,
  "a_fade_out_bars": <optional int <= duration_bars; A goes silent this many bars in while B keeps rising. Use it on most blends so A doesn't linger and muddy the mix>,
  "extras": <optional list, at most 2, from: "bass_kill" (cut A's bass 4 bars early so B's bass slams), "filter_sweep_out" (lowpass A's tail away), "echo_tail" (A exits with trailing beat echoes), "reverb_tail" (A's last moment washes into space)>,
  "rationale": "<1-2 sentences: why this seam pairing and style fit THESE two songs>"
}}

TRANSITION STYLES:
{_style_menu()}

HOW TO CHOOSE:
1. Energy first. Blend A's tail into B's first rise, or drop B's high-energy entry where A has gone quiet. Use the candidates' energy values and descriptions.
2. Then character. Two drum-driven dance tracks → drum_bridge or drop_swap. A hot track into a mellow one → wash_out. A big genre/tempo jump → vinyl_stop. Similar vibes → smooth_blend with a short a_fade_out_bars.
3. drop_swap and stutter_buildup need the relevant candidate to be vocal_safe (true) — never pick them otherwise.
3b. acapella_out and acapella_in layer one song's vocals over the other's instrumental — the ultimate crowd move when one track has an iconic vocal. They REQUIRE the key_verdict to be "compatible" (never pick them on a clash), and acapella_in additionally wants a vocal_safe OUT point on A. Use them when you KNOW the songs: a famous hook riding a new beat. Extras behave differently here: the riding vocal is kept CLEAN and DRY; "reverb_tail"/"echo_tail" apply only to the vocal's exit moment (a tasteful trail as it hands over), and "bass_kill"/"filter_sweep_out" are ignored. Most acapella transitions need no extras at all.
4. Vary the set. You are told which styles previous pairs used and may get a suggested style from the set planner. Treat the suggestion as a strong default; deviate only when the songs clearly demand it.

EXAMPLES (shape only — choose ids/styles that fit YOUR songs):
{{"out": "A2", "in": "B2", "style": "drop_swap", "duration_bars": 2, "extras": ["echo_tail"], "rationale": "Both are big-room house and B2 is the drop — snap straight into it while A echoes out."}}
{{"out": "A1", "in": "B1", "style": "wash_out", "duration_bars": 12, "a_fade_out_bars": 6, "rationale": "A ends hot and B opens ambient; washing A's chorus into reverb lets B's pads surface cleanly."}}
{{"out": "A3", "in": "B1", "style": "drum_bridge", "duration_bars": 16, "a_fade_out_bars": 8, "rationale": "Both grooves are percussion-led at similar energy; bridging the drums keeps the floor moving."}}
{{"out": "A1", "in": "B2", "style": "acapella_out", "duration_bars": 12, "rationale": "A's chorus vocal is iconic and the keys are compatible — ride that vocal over B's groove, then hand off to B's verse."}}

Output ONLY the JSON object. No prose outside it.
"""


SET_PLAN_SYSTEM_PROMPT = """You are an expert DJ planning the ARC of a full continuous set before mixing it.

INPUT: an ordered list of songs, each with index, title, artist, bpm, key, camelot_key, duration, and peak_energy_position (where in the song its energy peaks, 0..1).

YOUR JOB: assign each ADJACENT PAIR a suggested transition style so the set flows — build energy where it should build, breathe where it should breathe, and never repeat the same trick back-to-back unless the music demands it.

Styles: smooth_blend (classic blend), drop_swap (snap into B's drop — high energy), drum_bridge (drums bridge two grooves), wash_out (A dissolves, B surfaces — energy release or genre jump), stutter_buildup (tension stutter then drop), vinyl_stop (full stop then restart — theatrical, big tempo/vibe jumps only, at most once per set), acapella_out (A's vocal rides over B's new beat — iconic-vocal tracks, compatible keys only), acapella_in (B's vocal teases over A's beat before the swap — compatible keys only).

Return ONLY this JSON object:
{
  "arc": "<one sentence describing the set's energy shape>",
  "pairs": [
    {"index": 0, "style": "<style id>", "note": "<short reason>"},
    ...one entry per adjacent pair, index = position of the OUTGOING song...
  ]
}
"""


def decision_user_prompt(
    a_input: dict,
    b_input: dict,
    pair_facts: dict,
    context: dict | None = None,
) -> str:
    body = {"A": a_input, "B": b_input, "pair_facts": pair_facts}
    if context:
        body["set_context"] = context
    return json.dumps(body, indent=2)


def set_plan_user_prompt(songs: list[dict]) -> str:
    return json.dumps({"songs": songs}, indent=2)
