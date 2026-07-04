"""Phrase-grid tests: seam candidates land on 8/16-bar phrase boundaries.

120 BPM, 4/4 → 2 s per bar. Downbeats every 2 s. Sections re-anchor the
phrase counter, so grid points are section starts plus every Nth downbeat
within the section.
"""

from __future__ import annotations

from app.services.mixer.candidates import (
    build_in_candidates,
    build_out_candidates,
    max_seam_time,
    phrase_grid,
    _snap_to_phrase,
    _snap_to_phrase_at_or_before,
)
from app.services.mixer.types import AnalysisBundle

SPB = 2.0  # seconds per bar at 120 BPM 4/4
DUR = 240.0
DOWNBEATS = [i * SPB for i in range(int(DUR / SPB))]
SECTIONS = [
    {"start": 0.0, "end": 30.0, "label": "intro"},
    {"start": 30.0, "end": 150.0, "label": "body"},
    {"start": 150.0, "end": 240.0, "label": "outro"},
]


def _bundle(duration: float = DUR, sections=None) -> AnalysisBundle:
    sections = SECTIONS if sections is None else sections
    downbeats = [i * SPB for i in range(int(duration / SPB))]
    return AnalysisBundle(
        bpm=120.0, key="C", camelot_key="8B", time_signature=4,
        beat_grid=[i * 0.5 for i in range(int(duration / 0.5))],
        downbeats=downbeats, sections=sections, duration=duration,
    )


def test_phrase_grid_anchors_at_sections_16_bars():
    grid = phrase_grid(DOWNBEATS, SECTIONS)
    # Section anchors: 0, 30, 150. 16 bars = 32 s between boundaries.
    assert 0.0 in grid and 30.0 in grid and 150.0 in grid
    assert 62.0 in grid and 94.0 in grid and 126.0 in grid  # body phrases
    assert 182.0 in grid and 214.0 in grid                   # outro phrases
    assert 32.0 not in grid  # intro's next 16-bar step is past the section


def test_phrase_grid_falls_back_to_8_bars_on_short_songs():
    short_downbeats = [i * SPB for i in range(30)]  # 60 s song
    grid = phrase_grid(short_downbeats, [{"start": 0.0, "end": 60.0}])
    # A 16-bar grid would only have 0 and 32 (2 boundaries) — too sparse.
    assert grid == [0.0, 16.0, 32.0, 48.0]


def test_phrase_grid_without_data():
    assert phrase_grid([], SECTIONS) == []
    no_sections = phrase_grid(DOWNBEATS[:40], [])
    assert no_sections[0] == 0.0
    assert all(t % (16 * SPB) == 0 for t in no_sections)


def test_snap_helpers():
    grid = phrase_grid(DOWNBEATS, SECTIONS)
    assert _snap_to_phrase(30.0, grid, DOWNBEATS) == 30.0     # exact
    assert _snap_to_phrase(31.0, grid, DOWNBEATS) == 62.0     # next boundary
    # Ceiling blocks the boundary -> plain downbeat snap.
    assert _snap_to_phrase(31.0, grid, DOWNBEATS, ceiling=40.0) == 32.0
    assert _snap_to_phrase_at_or_before(63.0, grid, DOWNBEATS) == 62.0
    # Empty grid -> downbeat behavior.
    assert _snap_to_phrase(31.0, [], DOWNBEATS) == 32.0


def test_out_candidates_land_on_phrase_grid():
    b = _bundle()
    grid = phrase_grid(b.downbeats, b.sections)
    ceiling = max_seam_time(b.duration, b.bpm, b.time_signature)
    cands = build_out_candidates(b, [0.5] * int(DUR), [])
    assert cands
    for c in cands:
        assert c.time <= ceiling
        assert c.time in grid or c.time in b.downbeats
    # At least one candidate advertises its phrase alignment to the LLM.
    assert any("phrase-aligned" in c.description for c in cands)


def test_in_candidates_land_on_phrase_grid():
    b = _bundle()
    grid = phrase_grid(b.downbeats, b.sections)
    energy = [0.2] * 30 + [1.0] * 120 + [0.3] * 90  # body = the drop
    cands = build_in_candidates(b, energy, [])
    assert cands
    for c in cands:
        assert c.time in grid or c.time in b.downbeats
    # The intro-end candidate snapped onto the section-30 phrase anchor.
    assert any(c.time == 30.0 for c in cands)
