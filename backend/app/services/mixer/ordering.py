"""Auto track ordering — a DJ sorts the crate before mixing.

Scores every adjacent pair on the four things that make a transition
easy or hard (key relationship, tempo gap, energy continuity, genre
similarity) and searches for the cheapest path through all songs:
exact Held-Karp DP up to HELD_KARP_MAX songs, greedy nearest-neighbour
plus 2-opt polish beyond that.

Pure module: no DB, no LLM. The API layer assembles PairInputs from
Analysis rows and (optionally) lets an LLM pick between the top
orderings for narrative — never invent one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from app.services.mixer.archetypes import camelot_compatible, halftime_ratio
from app.services.mixer.plan import compute_pitch_shift

HELD_KARP_MAX = 12
# Edge-cost weights. Key and tempo dominate: a clash or an unstretchable
# gap forces a theatrical transition, while energy/genre friction just
# needs a smarter style choice.
W_KEY = 1.5
W_TEMPO = 1.5
W_ENERGY = 0.5
W_GENRE = 0.5
# Window (seconds) for the energy-continuity ends.
ENERGY_EDGE_SECONDS = 30
# Grade thresholds on total edge cost.
GRADE_A_MAX = 0.5
GRADE_B_MAX = 1.5


@dataclass(frozen=True)
class PairInput:
    """Per-song facts the scorer needs."""

    song_id: str
    title: str
    bpm: float | None
    key: str | None
    camelot_key: str | None
    energy_curve: list[float] = field(default_factory=list)
    genres: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    from_song_id: str
    to_song_id: str
    cost: float
    grade: str          # "A" | "B" | "C"
    reason: str


def _norm_energy_edges(curve: list[float]) -> tuple[float, float]:
    """(start_energy, end_energy), each 0..1 normalized to the song's own
    peak, averaged over the first/last ENERGY_EDGE_SECONDS."""
    if not curve:
        return 0.5, 0.5
    peak = max(curve) or 1.0
    head = curve[:ENERGY_EDGE_SECONDS]
    tail = curve[-ENERGY_EDGE_SECONDS:]
    return (
        (sum(head) / len(head)) / peak,
        (sum(tail) / len(tail)) / peak,
    )


def _key_cost(a: PairInput, b: PairInput) -> tuple[float, str]:
    if not a.camelot_key or not b.camelot_key:
        return 0.5, "keys unknown"
    if a.camelot_key == b.camelot_key:
        return 0.0, "same key"
    if camelot_compatible(a.camelot_key, b.camelot_key):
        try:
            same_number = int(a.camelot_key[:-1]) == int(b.camelot_key[:-1])
        except ValueError:
            same_number = False
        return (0.0, "relative keys") if same_number else (0.5, "neighbour keys")
    try:
        semis = abs(compute_pitch_shift(a.key or "", b.key or ""))
    except ValueError:
        semis = 3
    return 2.0 + min(semis, 6) * 0.25, "key clash"


def _tempo_cost(a: PairInput, b: PairInput) -> tuple[float, str]:
    if not a.bpm or not b.bpm:
        return 0.5, "tempo unknown"
    # Half/double-time pairs interlock natively — score the best of the
    # 1:1 and 2:1 beatmatch options.
    best_gap, best_ratio = None, 1.0
    for ratio in (1.0, halftime_ratio(a.bpm, b.bpm)):
        target = a.bpm * ratio
        gap = abs(target - b.bpm) / target
        if best_gap is None or gap < best_gap:
            best_gap, best_ratio = gap, ratio
    if best_gap <= 0.04:
        note = "tight tempo" if best_ratio == 1.0 else "half-time interlock"
        return 0.0, note
    if best_gap <= 0.12:
        return (best_gap - 0.04) / 0.08, "stretchable tempo gap"
    return min(3.0, 1.0 + (best_gap - 0.12) * 10.0), "big tempo jump"


def _energy_cost(a: PairInput, b: PairInput) -> float:
    _, a_end = _norm_energy_edges(a.energy_curve)
    b_start, _ = _norm_energy_edges(b.energy_curve)
    return abs(a_end - b_start)


def _genre_cost(a: PairInput, b: PairInput) -> float:
    if not a.genres or not b.genres:
        return 0.5
    sa, sb = set(a.genres), set(b.genres)
    union = sa | sb
    return 1.0 - (len(sa & sb) / len(union)) if union else 0.5


def pair_cost(a: PairInput, b: PairInput) -> float:
    key_c, _ = _key_cost(a, b)
    tempo_c, _ = _tempo_cost(a, b)
    return (
        W_KEY * key_c
        + W_TEMPO * tempo_c
        + W_ENERGY * _energy_cost(a, b)
        + W_GENRE * _genre_cost(a, b)
    )


def edge_info(a: PairInput, b: PairInput) -> Edge:
    key_c, key_note = _key_cost(a, b)
    tempo_c, tempo_note = _tempo_cost(a, b)
    cost = pair_cost(a, b)
    grade = "A" if cost <= GRADE_A_MAX else "B" if cost <= GRADE_B_MAX else "C"
    notes = [key_note, tempo_note]
    if _energy_cost(a, b) > 0.5:
        notes.append("energy jump")
    return Edge(
        from_song_id=a.song_id, to_song_id=b.song_id,
        cost=round(cost, 3), grade=grade, reason=", ".join(notes),
    )


def order_cost(songs: list[PairInput], order: list[int]) -> float:
    return sum(
        pair_cost(songs[order[i]], songs[order[i + 1]])
        for i in range(len(order) - 1)
    )


def _held_karp(cost: list[list[float]], starts: list[int]) -> list[int]:
    """Exact cheapest Hamiltonian path over all nodes, restricted to the
    given start nodes. O(2^n * n^2)."""
    n = len(cost)
    full = (1 << n) - 1
    # dp[(mask, last)] = (cost, prev_last)
    dp: dict[tuple[int, int], tuple[float, int | None]] = {}
    for s in starts:
        dp[(1 << s, s)] = (0.0, None)
    for mask in range(1, full + 1):
        for last in range(n):
            if not mask & (1 << last) or (mask, last) not in dp:
                continue
            base, _ = dp[(mask, last)]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                nm = mask | (1 << nxt)
                cand = base + cost[last][nxt]
                if (nm, nxt) not in dp or cand < dp[(nm, nxt)][0]:
                    dp[(nm, nxt)] = (cand, last)
    best_last = min(
        (last for last in range(n) if (full, last) in dp),
        key=lambda last: dp[(full, last)][0],
    )
    order = [best_last]
    mask, last = full, best_last
    while True:
        _, prev = dp[(mask, last)]
        if prev is None:
            break
        mask ^= 1 << last
        order.append(prev)
        last = prev
    order.reverse()
    return order


def _greedy(cost: list[list[float]], start: int) -> list[int]:
    n = len(cost)
    order, seen = [start], {start}
    while len(order) < n:
        last = order[-1]
        nxt = min(
            (j for j in range(n) if j not in seen),
            key=lambda j: cost[last][j],
        )
        order.append(nxt)
        seen.add(nxt)
    return order


def _two_opt(cost: list[list[float]], order: list[int], pin_first: bool) -> list[int]:
    def path_cost(o: list[int]) -> float:
        return sum(cost[o[i]][o[i + 1]] for i in range(len(o) - 1))

    best = order[:]
    best_cost = path_cost(best)
    improved = True
    lo = 1 if pin_first else 0
    while improved:
        improved = False
        for i, j in combinations(range(lo, len(best)), 2):
            cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
            c = path_cost(cand)
            if c + 1e-9 < best_cost:
                best, best_cost = cand, c
                improved = True
    return best


def suggest_order(
    songs: list[PairInput], pin_first: bool = True
) -> tuple[list[int], list[Edge]]:
    """Best ordering (as indices into `songs`) plus per-adjacency edges.

    `pin_first` keeps the user's opener in place — people usually know
    how they want to start.
    """
    n = len(songs)
    if n < 3:
        order = list(range(n))
    else:
        cost = [[pair_cost(songs[i], songs[j]) for j in range(n)] for i in range(n)]
        starts = [0] if pin_first else list(range(n))
        if n <= HELD_KARP_MAX:
            order = _held_karp(cost, starts)
        else:
            candidates = [
                _two_opt(cost, _greedy(cost, s), pin_first)
                for s in (starts if len(starts) <= 4 else starts[:4])
            ]
            order = min(
                candidates,
                key=lambda o: sum(cost[o[i]][o[i + 1]] for i in range(n - 1)),
            )
    edges = [
        edge_info(songs[order[i]], songs[order[i + 1]])
        for i in range(len(order) - 1)
    ]
    return order, edges
