"""Listener-feedback aggregation → plain-language planner context.

No ML, no scores — just honest counts the LLM can weigh: "wash_out +4/-0;
stutter_buildup +1/-3, 2 skips". Interpretable, debuggable, and the
prompt tells the model to treat it as taste calibration, not law.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ListenerEvent, ListenerEventKind

# Only the most recent events shape the summary — taste drifts, and an
# unbounded history would let January outvote yesterday.
RECENT_EVENT_LIMIT = 200
# Styles with no reactions at all are omitted (no signal != bad signal).
MAX_STYLES_IN_SUMMARY = 6


@dataclass
class _Tally:
    up: int = 0
    down: int = 0
    skips: int = 0
    genre_pairs: dict[tuple[str, str], int] = field(
        default_factory=lambda: defaultdict(int)
    )

    @property
    def signal(self) -> int:
        return self.up + self.down + self.skips


def _fetch_recent(db: Session) -> list[ListenerEvent]:
    return list(
        db.scalars(
            select(ListenerEvent)
            .where(ListenerEvent.style.is_not(None))
            .order_by(ListenerEvent.created_at.desc())
            .limit(RECENT_EVENT_LIMIT)
        )
    )


def style_tallies(db: Session) -> dict[str, _Tally]:
    tallies: dict[str, _Tally] = defaultdict(_Tally)
    for ev in _fetch_recent(db):
        t = tallies[ev.style]
        if ev.kind == ListenerEventKind.thumbs_up:
            t.up += 1
        elif ev.kind == ListenerEventKind.thumbs_down:
            t.down += 1
        elif ev.kind == ListenerEventKind.skip:
            t.skips += 1
        if ev.from_genre and ev.to_genre:
            t.genre_pairs[(ev.from_genre, ev.to_genre)] += (
                1 if ev.kind == ListenerEventKind.thumbs_up else -1
            )
    return dict(tallies)


def feedback_context(
    db: Session,
    from_genre: str | None = None,
    to_genre: str | None = None,
) -> str | None:
    """One prompt line summarizing the listener's reaction history, or
    None when there is no signal yet."""
    tallies = style_tallies(db)
    if not tallies:
        return None
    ranked = sorted(
        tallies.items(), key=lambda kv: kv[1].signal, reverse=True
    )[:MAX_STYLES_IN_SUMMARY]
    parts = []
    for style, t in ranked:
        if t.signal == 0:
            continue
        bit = f"{style} +{t.up}/-{t.down}"
        if t.skips:
            bit += f" ({t.skips} skips)"
        parts.append(bit)
    if not parts:
        return None
    line = (
        "Listener reaction history (recent, treat as taste calibration, "
        "not law): " + "; ".join(parts) + "."
    )
    if from_genre and to_genre:
        pair_bits = [
            f"{style} net {t.genre_pairs[(from_genre, to_genre)]:+d}"
            for style, t in ranked
            if (from_genre, to_genre) in t.genre_pairs
        ]
        if pair_bits:
            line += (
                f" For {from_genre}→{to_genre} pairs specifically: "
                + "; ".join(pair_bits) + "."
            )
    return line
