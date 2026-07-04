"""TTS host (F11): script budgets/parsing, placement + ducking math, the
overlay's failure tolerance, and clip caching. All TTS/LLM mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from app.core.config import settings
from app.services.host.mixdown import (
    DUCK_GAIN,
    duck_and_add,
    place_clips,
)
from app.services.host.overlay import (
    apply_host_overlay,
    pick_intro_start,
    resolve_host_settings,
)
from app.services.host.script import (
    build_script_user_prompt,
    parse_script_response,
)
from app.services.host.tts import clip_cache_key

SR = 44100


# -------------------------------------------------------------------- script

def test_script_budgets_by_frequency():
    songs = [{"title": f"S{i}", "artist": "X"} for i in range(6)]
    p = build_script_user_prompt(songs, "gym", None, "hype", "intro_only")
    assert p["slots"] == {"intro": True, "max_mid_set_drops": 0, "outro": False}
    p = build_script_user_prompt(songs, None, None, "radio", "chatty")
    assert p["slots"]["max_mid_set_drops"] == 4
    assert p["slots"]["outro"] is True
    assert "drivetime" in p["persona"]


def test_parse_script_enforces_budget_and_shape():
    obj = {"segments": [
        {"slot": "intro", "text": "  welcome   in  "},
        {"slot": "intro", "text": "duplicate intro"},
        {"slot": 0, "text": "first drop"},
        {"slot": 1, "text": "second drop"},
        {"slot": 99, "text": "out of range"},
        {"slot": "outro", "text": "goodnight"},
        {"slot": 2, "text": ""},
        "garbage",
    ]}
    out = parse_script_response(obj, n_pairs=5, frequency="sparse")
    slots = [s["slot"] for s in out]
    assert slots == ["intro", 0, 1]         # sparse: no outro, 2 mids max
    assert out[0]["text"] == "welcome in"   # whitespace collapsed
    assert parse_script_response({"nope": 1}, 5, "sparse") == []
    assert parse_script_response(obj, 5, "intro_only") == [
        {"slot": "intro", "text": "welcome in"}
    ]


# ------------------------------------------------------------------- mixdown

def test_duck_and_add_levels():
    mix = np.full((SR * 10, 2), 0.5, dtype=np.float32)
    clip = np.full((SR * 2, 2), 0.1, dtype=np.float32)
    duck_and_add(mix, SR, clip, start_sample=4 * SR)
    mid = int(5.0 * SR)
    # Music ducked to 0.5*DUCK_GAIN with the voice on top.
    assert mix[mid, 0] == pytest.approx(0.5 * DUCK_GAIN + 0.1, abs=1e-3)
    # Far outside: untouched.
    assert mix[SR, 0] == pytest.approx(0.5)
    assert mix[int(8 * SR), 0] == pytest.approx(0.5)


def test_place_clips_slot_rules():
    mix = np.zeros((SR * 120, 2), dtype=np.float32)
    transitions = [
        {"index": 0, "start": 40.0, "end": 50.0},
        {"index": 1, "start": 100.0, "end": 110.0},
    ]
    clip3s = np.full((SR * 3, 2), 0.1, dtype=np.float32)
    clips = [
        {"slot": "intro", "text": "hi", "audio": clip3s},
        {"slot": 0, "text": "that was...", "audio": clip3s},
        # 60s clip can never clear the next transition -> skipped.
        {"slot": 1, "text": "too long",
         "audio": np.full((SR * 60, 2), 0.1, dtype=np.float32)},
    ]
    events = place_clips(mix, SR, clips, transitions, intro_start_s=2.0)
    assert [e["slot"] for e in events] == ["intro", 0]
    assert events[0]["start"] == 2.0
    assert events[1]["start"] == pytest.approx(50.5)   # transition end + 0.5
    # The audio landed where the events say.
    assert abs(mix[int(3.0 * SR), 0]) > 0
    assert abs(mix[int(52.0 * SR), 0]) > 0


def test_place_clips_skips_intro_that_hits_first_transition():
    mix = np.zeros((SR * 60, 2), dtype=np.float32)
    transitions = [{"index": 0, "start": 6.0, "end": 12.0}]
    clips = [{
        "slot": "intro", "text": "long hello",
        "audio": np.full((SR * 10, 2), 0.1, dtype=np.float32),
    }]
    events = place_clips(mix, SR, clips, transitions, intro_start_s=2.0)
    assert events == []
    assert np.max(np.abs(mix)) == 0.0


def test_pick_intro_start():
    regions = [
        {"start": 0.0, "end": 3.0, "safe": True},     # too short for 4s clip
        {"start": 10.0, "end": 30.0, "safe": True},
        {"start": 5.0, "end": 50.0, "safe": False},   # vocals — ignored
    ]
    assert pick_intro_start(regions, clip_seconds=4.0) == 10.0
    assert pick_intro_start(None, 4.0) == 2.0
    assert pick_intro_start([], 4.0) == 2.0


# ------------------------------------------------------------------- overlay

def _songs():
    return [{"title": "Alpha", "artist": "A"}, {"title": "Beta", "artist": "B"}]


def test_overlay_off_switches():
    mix = np.zeros((SR * 30, 2), dtype=np.float32)
    assert apply_host_overlay(
        mix, SR, {"transitions": []}, object(),
        songs=_songs(), occasion=None, vibe_note=None,
        queue_frequency="off", queue_persona=None, song1_safe_regions=None,
    ) == []
    # Global provider off wins even when the queue asks for a host.
    assert settings.tts_provider == "off"  # conftest default
    assert apply_host_overlay(
        mix, SR, {"transitions": []}, object(),
        songs=_songs(), occasion=None, vibe_note=None,
        queue_frequency="chatty", queue_persona=None, song1_safe_regions=None,
    ) == []
    assert np.max(np.abs(mix)) == 0.0


def test_overlay_places_intro_with_mocked_pipeline():
    mix = np.zeros((SR * 60, 2), dtype=np.float32)
    timeline = {"transitions": [
        {"index": 0, "start": 40.0, "end": 50.0,
         "from_song_id": "x", "to_song_id": "y"},
    ]}
    clip = np.full((SR * 3, 2), 0.1, dtype=np.float32)
    with (
        patch.object(settings, "tts_provider", "kokoro"),
        patch("app.services.host.overlay.generate_script",
              return_value=[{"slot": "intro", "text": "locked in"}]),
        patch("app.services.host.overlay.synthesize_cached",
              new=AsyncMock(return_value=clip)),
    ):
        events = apply_host_overlay(
            mix, SR, timeline, object(),
            songs=_songs(), occasion="house_party", vibe_note=None,
            queue_frequency="intro_only", queue_persona=None,
            song1_safe_regions=[{"start": 1.0, "end": 20.0, "safe": True}],
        )
    assert len(events) == 1
    assert events[0]["slot"] == "intro"
    assert events[0]["text"] == "locked in"
    assert events[0]["start"] == 1.0        # safe-span start, not fallback
    assert np.max(np.abs(mix)) > 0


def test_overlay_survives_tts_failure():
    mix = np.zeros((SR * 60, 2), dtype=np.float32)
    with (
        patch.object(settings, "tts_provider", "kokoro"),
        patch("app.services.host.overlay.generate_script",
              return_value=[{"slot": "intro", "text": "hello"}]),
        patch("app.services.host.overlay.synthesize_cached",
              new=AsyncMock(side_effect=RuntimeError("tts down"))),
    ):
        events = apply_host_overlay(
            mix, SR, {"transitions": []}, object(),
            songs=_songs(), occasion=None, vibe_note=None,
            queue_frequency="intro_only", queue_persona=None,
            song1_safe_regions=None,
        )
    assert events == []                     # voiceless, not broken
    assert np.max(np.abs(mix)) == 0.0


def test_resolve_host_settings_defaults():
    freq, persona = resolve_host_settings(None, None, "afters")
    assert freq == settings.host_frequency
    assert persona == "latenight"           # afters occasion default
    freq, persona = resolve_host_settings("chatty", "minimal", "afters")
    assert (freq, persona) == ("chatty", "minimal")
    _, persona = resolve_host_settings(None, None, None)
    assert persona == "radio"


def test_clip_cache_key_stable_and_scoped():
    k1 = clip_cache_key("hello", "af_heart", "kokoro")
    assert k1 == clip_cache_key("hello", "af_heart", "kokoro")
    assert k1.startswith("host_clips/")
    assert k1 != clip_cache_key("hello", "af_heart", "openai")
    assert k1 != clip_cache_key("hello", "am_puck", "kokoro")
