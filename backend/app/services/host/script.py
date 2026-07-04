"""Host script generation — what the voice actually says.

One LLM call per stitch produces short segments for specific slots:
"intro" (over song 1's opening), integer pair indexes (right after that
transition lands, over B's entry), and "outro" (the last seconds of the
mix). Personas set the register; hard rules in the prompt keep segments
short and stop the model inventing facts about the songs.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

HOST_PERSONAS: dict[str, str] = {
    "hype": (
        "high-energy club MC: punchy, loud-on-the-page, crowd-first "
        "('let's GO', 'make some noise'), never more than one exclamation "
        "per segment"
    ),
    "latenight": (
        "2 a.m. late-night radio voice: warm, unhurried, intimate, a "
        "little wry; speaks quietly like the room is half-lit"
    ),
    "radio": (
        "classic drivetime radio host: professional, friendly, tight "
        "transitions, always names artists cleanly"
    ),
    "minimal": (
        "almost silent: a few words at most, cool and dry, lets the "
        "music do the talking"
    ),
}

# How many mid-set drops each frequency allows (besides intro/outro).
FREQUENCY_BUDGET = {
    "off": (False, 0, False),        # (intro, max mid-set, outro)
    "intro_only": (True, 0, False),
    "sparse": (True, 2, False),
    "chatty": (True, 4, True),
}


def _slots_for(frequency: str, n_pairs: int) -> tuple[bool, int, bool]:
    intro, mids, outro = FREQUENCY_BUDGET.get(
        frequency, FREQUENCY_BUDGET["intro_only"]
    )
    return intro, min(mids, max(0, n_pairs - 1)), outro


def build_script_user_prompt(
    songs: list[dict],
    occasion: str | None,
    vibe_note: str | None,
    persona: str,
    frequency: str,
) -> dict:
    """The JSON body handed to the LLM (kept as a dict for testability)."""
    intro, mids, outro = _slots_for(frequency, max(0, len(songs) - 1))
    return {
        "persona": HOST_PERSONAS.get(persona, HOST_PERSONAS["radio"]),
        "occasion": occasion,
        "vibe_note": vibe_note,
        "songs": [
            {"index": i, "title": s.get("title"), "artist": s.get("artist")}
            for i, s in enumerate(songs)
        ],
        "slots": {
            "intro": intro,
            "max_mid_set_drops": mids,
            "outro": outro,
        },
    }


def parse_script_response(obj, n_pairs: int, frequency: str) -> list[dict]:
    """Validate the LLM's segments; drop anything malformed or over
    budget rather than failing the stitch."""
    if not isinstance(obj, dict) or not isinstance(obj.get("segments"), list):
        return []
    intro_ok, mid_budget, outro_ok = _slots_for(frequency, n_pairs)
    out: list[dict] = []
    mid_used = 0
    for seg in obj["segments"]:
        if not isinstance(seg, dict):
            continue
        slot = seg.get("slot")
        text = seg.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = " ".join(text.split())[:400]
        if slot == "intro" and intro_ok and not any(
            s["slot"] == "intro" for s in out
        ):
            out.append({"slot": "intro", "text": text})
        elif slot == "outro" and outro_ok and not any(
            s["slot"] == "outro" for s in out
        ):
            out.append({"slot": "outro", "text": text})
        elif isinstance(slot, int) and 0 <= slot < n_pairs \
                and mid_used < mid_budget:
            if any(s["slot"] == slot for s in out):
                continue
            out.append({"slot": slot, "text": text})
            mid_used += 1
    return out


def generate_script(
    songs: list[dict],
    occasion: str | None,
    vibe_note: str | None,
    persona: str,
    frequency: str,
) -> list[dict]:
    """[{slot, text}] via one LLM call; [] on any failure."""
    import json

    from app.services.llm import get_llm_provider
    from app.services.llm.prompts import HOST_SCRIPT_SYSTEM_PROMPT

    n_pairs = max(0, len(songs) - 1)
    intro, mids, outro = _slots_for(frequency, n_pairs)
    if not intro and mids == 0 and not outro:
        return []
    try:
        provider = get_llm_provider()
        obj = asyncio.run(
            provider.complete_json(
                system=HOST_SCRIPT_SYSTEM_PROMPT,
                user=json.dumps(
                    build_script_user_prompt(
                        songs, occasion, vibe_note, persona, frequency
                    ),
                    indent=2,
                ),
                cache_namespace="host_script_logs",
            )
        )
        return parse_script_response(obj, n_pairs, frequency)
    except Exception:
        logger.exception("host script: generation failed")
        return []
