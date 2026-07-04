# MixMind — "Human DJ" feature designs

13 designs, each independently approvable. Format: goal → user-visible behavior →
technical design (data model / backend / prompts / frontend) → tests → effort →
dependencies → open questions. File paths refer to the current codebase.

**Status: IMPLEMENTED 2026-07-04** (all 12 features; F12 dropped by decision).
Decisions: F11 both TTS providers behind a config switch, free-to-use (Kokoro
local) as the default; F12 radio-clean DROPPED; F8 pins the first track; F4 all
four archetypes; F6 keeps the 90 s lead time and adds an instant
skip-to-next-transition button.

Implementation waves (dependency-ordered):

- **Wave 1 (silent quality)**: F13 render QA, F1 phrase seams, F2 EQ bass swap
- **Wave 2 (new moves)**: F4 archetypes (all), F3 tempo meet-in-middle
- **Wave 3 (context)**: F8 auto ordering, F7 occasion presets, F10 arc templates
- **Wave 4 (headliners)**: F5 reaction signals, F6 energy dial + skip, F9 hook teasing, F11 TTS host

---

## F1. Phrase-aligned seams

**Goal.** Human DJs mix on 8/16-bar phrase boundaries. Today `candidates.py` snaps
seam candidates to the *next downbeat* after a section start, which can land mid-phrase
(bar 3 of a 8-bar phrase) — technically on-grid, musically "off".

**Behavior.** No UI change. Every transition starts/lands on a phrase boundary; renders
simply sound more intentional.

**Design.**
- New pure function in `services/mixer/candidates.py`:
  `phrase_grid(downbeats, sections, bars_per_phrase=8) -> list[float]` — for each
  section, anchor a phrase counter at the section start's nearest downbeat, then emit
  every `bars_per_phrase`-th downbeat until the section ends. Sections re-anchor the
  counter (song structure resets phrasing). Songs with no sections anchor at downbeat 0.
- New snapping helpers `_snap_to_phrase(t, grid, downbeats)` (next phrase boundary ≥ t,
  falling back to plain downbeat snap when the grid is empty or the next boundary
  overshoots the headroom ceiling by > 1 phrase) and the `_at_or_before` mirror.
- `build_out_candidates` / `build_in_candidates` use phrase snapping instead of
  `_snap_to_downbeat`; candidate `description` gains “(phrase-aligned)” so the LLM
  knows. The "latest possible out point" candidate snaps *at-or-before* the ceiling
  on the phrase grid first, downbeat grid second.
- Executor unchanged (`_snap_downbeat` stays as a safety net — phrase boundaries are
  downbeats by construction).
- 16-bar preference: try 16-bar boundaries first for OUT candidates on songs ≥ 3 min
  (typical EDM phrasing), fall back to 8. Constant `PHRASE_BARS_CHOICES = (16, 8)`.

**Tests** (`test_planner_v2.py` or new `test_candidates_phrasing.py`): synthetic
120 BPM grid + 2 sections → grid boundaries every 16 s; candidate times ∈ grid;
ceiling respected; empty-downbeats fallback; degenerate 20 s song falls back cleanly.

**Effort:** small. **Depends on:** nothing. **Risk:** low — pure functions.

---

## F2. EQ-style bass swap on every blend

**Goal.** Two basslines overlapping = mud. Humans swap bass with EQ, never blend it.
Acapella styles already quick-swap the bass stem; long blends (smooth_blend,
drum_bridge, wash_out) still crossfade bass A↔B over the full window.

**Behavior.** No UI change. During any blend ≥ 8 bars, A's bass exits (fast fade +
gentle high-pass) just before the window midpoint and B's bass takes over alone.

**Design.**
- `archetypes.py::_crossfade_calls` gains a `bass_swap: bool` param (True for
  smooth_blend / drum_bridge / wash_out with `duration_bars >= 8`; False otherwise
  so short styles stay bit-identical). When on, the bass stem call becomes:
  A-bass fades out over bars `[mid-2, mid]` (via `a_fade_out_bars` on a bass call
  whose window is `start_bar=0, duration_bars=mid`), B-bass fades in over
  `[mid, mid+1]` (second bass call? — no: the executor requires exactly 4
  crossfade_stem calls). Implementation that fits the existing contract:
  bass call gets `start_bar = mid - 1, duration_bars = 2, a_fade_out_bars = 1` —
  pure-A bass before the window, 2-bar handover at the midpoint, pure-B bass after.
  The executor's per-stem windowing already renders exactly that (pre-window = A stem,
  post-window = B stem).
- Optional garnish rides along: a stem-scoped `filter_sweep` (highpass 20 Hz → 250 Hz,
  1 bar) on A's bass ending at the handover. Requires executor support for
  `filter_sweep.stem` — mirror the existing `volume_fade` stem-scoping in the pre-fx
  loop (`executor.py` step 2b): scope `names` to the one stem when
  `fx.get("stem")` is canonical, for `filter_sweep` too.
- New `TransitionExtra.bass_swap`? No — it's default behavior, not an extra. Add a
  settings kill-switch `bass_swap: bool = True` in `core/config.py`, threaded through
  `expand()` like `loudness_match`.
- Prompt: one line in `DECISION_SYSTEM_PROMPT` rule 4: basses are auto-swapped at the
  window midpoint on long blends; the model doesn't need to manage bass overlap.

**Tests:** expanded smooth_blend(12) has bass call `start_bar=5, duration_bars=2`;
drop_swap unchanged; executor render with constant stems → summed bass level never
exceeds a single stem's level during the window (no doubling); stem-scoped
filter_sweep only touches the bass buffer.

**Effort:** small-medium. **Depends on:** nothing (F1 compatible). **Risk:** low.

---

## F3. Bidirectional tempo meet-in-the-middle

**Goal.** Today B is stretched 100% of the way to A's tempo through the crossfade, then
ramps to native. On a 10% gap that's a very audible stretch on B while A sounds native —
one-sided and artificial. Humans nudge *both* decks toward the middle.

**Behavior.** For tempo gaps between 4% and 12%: A accelerates/decelerates over its last
16 bars to the midpoint BPM, the crossfade happens at mid-tempo, then B glides from mid
to native afterward. Each song carries half the artifact. Gaps < 4% keep today's
behavior (inaudible); gaps > 12% are the planner's job to route around (vinyl_stop /
F4 half-time).

**Design.**
- **Plan format:** `set_tempo_ramp` with `song: "A"` is already legal
  (`validation.py::SONG_FIELDS_BY_TOOL`) but ignored by the executor. Archetypes emit,
  when `0.04 < gap <= 0.12`:
  - `set_tempo_ramp(song=A, start_time=seam_a - 16*spb_a, end_time=seam_a,
    start_bpm=a.bpm, end_bpm=mid_bpm)` where `mid_bpm = (a.bpm + b.bpm) / 2`
  - B's stretch target becomes `mid_bpm` (the existing post-crossfade B ramp then runs
    `mid_bpm → b.bpm` instead of `a.bpm → b.bpm`).
- **Executor:** build an A-side `timemap_stretch` mirroring the existing B map:
  identity up to ramp start, 10-point linear rate ramp `1.0 → a.bpm/mid_bpm`, constant
  after. Applied to `a_stems` (per stem) and `a_mix` (master — the F-fix from the
  washout work keeps the master alive; the stretch applies to the whole blended
  buffer). `a_seam_sample` is then computed *through the map* (ramp is entirely
  pre-seam, so `a_seam_out = map(seam_a)`); `samples_per_bar_a` inside the window uses
  `mid_bpm`. B's `rate_A` becomes `mid_bpm / b.bpm`.
- **Stitcher:** `stitch_queue.py::_get_mix0_sample` currently assumes
  `a_seam_samp = a_seam_orig * sr`. Extend it to read plan0's A-ramp (if present) and
  apply the same 10-point map — ~20 lines mirroring the B-ramp branch. Same for
  `_build_timeline`'s `a_seam_sample`.
- **Pre-fx interaction:** pre-fx apply in original A time *before* the stretch —
  already the executor's order of operations; no change. Pair-phase alignment runs on
  post-stretch buffers as today.
- Settings: `tempo_meet_in_middle: bool = True`, gap thresholds as constants in
  `archetypes.py`.

**Tests:** unit — A-map monotonic, `map(seam_a)` lands where 16 bars at ramped tempo
predict; render — output length matches analytic expectation, seam content is A's
(recognizable amplitude) at mapped position; stitcher — `_get_mix0_sample` with an
A-ramp plan returns the shifted junction (pure-function test, no audio); e2e pair test
with mocked pyrubberband asserting both A and B stretch calls happen.

**Effort:** large (the one genuinely invasive executor change). **Depends on:**
nothing, but land *after* F13 (QA) so regressions get caught by metrics.
**Risk:** medium-high — timeline math in three places (executor, stitcher junction,
stitcher timeline) must agree. Mitigate with shared helper
`services/mixer/tempo_map.py` used by all three.

---

## F4. New archetypes: double_drop, backspin, breakdown_blend, half-time trick

Four independent sub-features; approve individually.

### F4a. double_drop
- **What:** both songs' drops hit the same downbeat — the crowd-scream move.
- **Candidates:** OUT menu today only offers late-song section starts. Add "last
  high-energy section start (drop)" OUT candidate when a section has energy ≥ 0.8 and
  fits the headroom ceiling (`build_out_candidates`).
- **Decision:** new `TransitionStyle.double_drop`, durations (8,), gated in
  `planner_v2` like acapella: requires `camelot_compatible`, tempo gap ≤ 6%, both
  chosen candidates energy ≥ 0.8 (downgrade to drop_swap otherwise).
- **Expansion:** seam = both drops; all stems: B enters full over 1 bar
  (`duration_bars=1` fade-in), A holds at full for 4 bars (`a_fade_out_bars` applied to
  a window shaped `start_bar=0, duration=8, a_fade=4` — B rises fast, A leaves after
  4 bars); bass: A-bass killed at bar 0 via existing `volume_fade` (B's bass owns the
  drop — non-negotiable, two drop basslines will pump the limiter).
- **Prompt:** style menu entry + rule: only when both tracks have a real drop and the
  keys/tempos are close; at most once per set (like vinyl_stop).
- **Tests:** expansion shape; downgrade on clash/gap; soft-knee limiter engages but
  output peak ≤ ceiling.

### F4b. backspin
- **What:** A's last beat spins backward (accelerating reverse) into a hard cut; B
  drops clean. Sibling of vinyl_stop.
- **Executor:** new tool `backspin {song, start_time, duration_beats, bpm}` — take the
  `duration_beats` of audio *before* start_time, reverse it, resample through an
  accelerating phase map (mirror of `_apply_turntable_stop`'s quadratic, reversed),
  write it over `[start_time, start_time + ~0.6 * duration]`, silence after. Add to
  `LEGAL_TOOLS`, `SONG_FIELDS_BY_TOOL`, executor pre-fx dispatch.
- **Style:** `TransitionStyle.backspin`, durations (2, 4), a_fade forced ≤ 1,
  vocal-safe OUT preferred. Expansion mirrors vinyl_stop but with the new tool.
- **Prompt:** menu entry: theatrical, hip-hop/open-format flavor, at most once per set.
- **Tests:** tool output is time-reversed content (correlate against reversed source),
  ends in silence, no click at the cut (ends at zero-crossing or 5 ms fade).

### F4c. breakdown_blend
- **What:** blend during *both* songs' thin sections (A's final breakdown → B's intro/
  breakdown) — the "nobody notices the swap" move.
- **Candidates:** add OUT candidate "start of final low-energy section" (energy ≤ 0.4,
  late-song) and keep existing IN intro candidates (already low-energy).
- **Style:** `TransitionStyle.breakdown_blend`, durations (16,), expansion = smooth
  blend mechanics + `a_fade_out_bars = 12` + loudness match on (quiet sections
  mismatch more). No new executor work.
- **Prompt:** menu entry + choose-rule: prefer when both songs have clear quiet
  sections; the invisible transition.

### F4d. Half-time / double-time trick
- **What:** 85 BPM hip-hop into 170 BPM DnB without a 2× stretch: beat grids align at
  a 2:1 ratio; B is stretched only to `2 * a.bpm` (or `a.bpm / 2`), drums interlock.
- **Plan format:** `set_transition_window` gains optional `tempo_ratio: 2.0 | 0.5`
  (default 1.0). Executor: `rate_A = a.bpm / (b.bpm * ratio)`... i.e. effective B bpm
  for all stretch/bar math is `b.bpm * ratio`. One variable substitution at the top of
  `render()`; bar lengths on the A side unchanged.
- **Planner:** `_pair_facts` detects `abs(a.bpm*2 - b.bpm)/b.bpm <= 0.04` (or inverse)
  and reports "half-time compatible — grids interlock at 2:1; treat the tempo gap as
  small". Archetypes compute the ratio and stamp it on the window for any style when
  the raw gap > 12% but the ratio gap ≤ 4%. Pitch/camelot logic unchanged.
- **Stitcher:** `_get_mix0_sample`'s `rate_A` must read the ratio from plan0's window —
  one-line fix alongside F3's changes.
- **Tests:** ratio detection; executor stretch called with ratio-adjusted rate;
  drum_bridge at 85/170 renders with 2:1 alignment (beat positions of B land on every
  other A beat in output).

**Effort:** F4a small-medium, F4b medium, F4c small, F4d medium.
**Depends on:** F4d shares stitcher math with F3 — implement together.

---

## F5. Reaction signals (the feedback loop)

**Goal.** The system never learns. Capture explicit (👍/👎 per transition) and implicit
(skip/replay) signals and feed them back into planning as plain-language context.

**Behavior.** During mix playback the transition indicator (Player.tsx already tracks
`activeTransition` from the timeline) shows 👍/👎. Skipping ahead during/shortly after a
transition logs a skip against it. Future queue plans see: *"listener history:
wash_out 4👍 0👎; stutter_buildup 1👍 3👎, 2 skips — avoid unless clearly right."*

**Design.**
- **Model** (`models/feedback.py`, one table + migration): `listener_events`:
  `id UUID PK`, `queue_id FK`, `mix_plan_id FK nullable`, `kind`
  enum(`thumbs_up,thumbs_down,skip,replay`), `style String nullable` (denormalized so
  history survives plan deletion), `from_genre/to_genre String nullable` (first Essentia
  genre tag of each song, denormalized), `position_seconds Float nullable`,
  `created_at`.
- **API** (`api/feedback.py`): `POST /api/feedback` `{mix_plan_id, kind}` (server fills
  style/genres); `POST /api/feedback/playback` `{queue_id, kind, position_seconds}` —
  server maps position → transition via `QueueRender.timeline` (a skip is attributed to
  a transition if position is inside it or ≤ 30 s after its end). `GET /api/feedback/summary`
  for the UI.
- **Aggregation** (`services/feedback/summary.py`): pure function → per-style counts,
  overall and for the current pair's genre pair; render as ≤ 3 prompt lines. Cap the
  window to the last 200 events.
- **Planner wiring:** `render_transition._build_plan` fetches the summary (one indexed
  query) and passes `context["listener_feedback"]` through `build_plan_v2` (already has
  a `context` dict); `plan_set` gets the same block. Prompt addition: one rule — treat
  feedback as taste calibration, not law.
- **Frontend:** thumbs in the transition indicator chip (Player.tsx); skip detection in
  the existing seek/next handlers; optimistic POST, no UI beyond a small "noted ✓".
- Single-user assumption (no auth in the app) — no user column; add one later if needed.

**Tests:** API round-trip; position→transition attribution (timeline fixture); summary
rendering (counts, empty history → no context line); planner passes context through
(StubProvider captures the user prompt and asserts the block is present).

**Effort:** medium. **Depends on:** nothing. **Risk:** low; purely additive.

---

## F6. Live energy dial

**Goal.** "Take it up / hold / cool down" mid-set — a DJ reads the room and bends the
rest of the set; we re-plan the not-yet-played remainder.

**Behavior.** Three-button control in the Player during mix playback. Pressing one
re-plans transitions that start ≥ ~90 s ahead of the playhead (re-render takes time),
re-stitches, and the player hot-swaps to the new file at the same output position
(timestamps before the first changed transition are identical, so the position maps 1:1
— the stitcher only changes content downstream of the first re-rolled pair).

**Design.**
- **Model:** `MixPlan.energy_bias String nullable` (`"up" | "down"`) + migration.
  `Queue.energy_bias` for pairs not yet planned.
- **API:** `POST /api/queues/{id}/energy {direction, position_seconds}` →
  using `QueueRender.timeline`, select MixPlans whose transition `start >
  position + LEAD_SECONDS (90)`; for each: set `energy_bias`, bump `reroll_nonce`,
  reset status to `pending`, clear `rendered_audio_path`; reset `QueueRender` to
  pending; dispatch the existing render chord + `auto_stitch` flow (same path the
  reroll endpoint uses — reuse `mix_plans.py::reroll` internals, extracted to a
  shared helper). Response: list of affected pair indices + ETA hint.
- **Planner:** `energy_bias` → prompt context: *"the listener just asked to RAISE the
  energy: prefer high-energy IN candidates, drop_swap/stutter_buildup/double_drop,
  shorter blends"* (mirror for down: breakdown_blend/wash_out, earlier mellow
  entries). Also nudges `default_decision`'s IN-candidate pick (energy ≥ 0.8 rule
  already exists — reuse).
- **Frontend:** Player gains the dial (only while playing a mix, only if ≥ 2
  transitions remain); poll `GET /queues/{id}/mix` for the new render; on ready, swap
  `<audio>` src and restore `currentTime` (content before the first changed pair is
  byte-different — AAC re-encode — but time-identical; restoring position is safe).
  Show a subtle "re-reading the room… 2 transitions updating" toast.
- **Instant skip (DECIDED — included):** a "next transition" button beside the dial:
  seeks the `<audio>` element to `next_transition.start - 2s` using the existing
  timeline data. Pure frontend (no API), instant, and logs a `skip` listener event
  (F5) attributed to the skipped-over span so the feedback loop learns from it.

**Tests:** endpoint selects only future pairs (timeline fixture with playhead
mid-song 2 of 5); nonce bumped, statuses reset exactly once; planner receives bias
context; stitch re-dispatch happens; no-op when < 2 future transitions.

**Effort:** medium. **Depends on:** F5's prompt-context plumbing pattern (not the
table); reuses reroll + auto_stitch. **Risk:** medium — mid-playback swap UX needs the
position-mapping guarantee stated above (holds because pair renders upstream of the
first change are reused byte-identical pre-encode).

---

## F7. Occasion presets

**Goal.** A wedding, a gym session, and a 2 a.m. afters want different DJs. Give the
system the context a human books the gig with.

**Behavior.** When building/locking a queue, pick an occasion chip (house party / gym /
dinner / study / radio show / wedding / afters / none) and optionally type a free-text
vibe note ("90s hip-hop birthday, keep it fun"). All downstream planning reads it.

**Design.**
- **Model:** `Queue.occasion String nullable`, `Queue.vibe_note String(300) nullable`
  + migration. Set via new `PATCH /api/queues/{id}` fields (queues API) any time
  before lock.
- **Presets** (`services/mixer/occasions.py`, code not DB):
  `OCCASIONS: dict[str, Occasion]` with fields:
  `arc_hint` (e.g. gym: "sustained high energy, no long breathers"),
  `style_bias` ("prefer drop_swap/drum_bridge; avoid vinyl_stop" as prose),
  `energy_floor/ceiling` (0..1, used by F8 ordering + F10 arcs),
  `host_persona_default` (F11).
- **Wiring:** `plan_set` injects `occasion` + `arc_hint` + `vibe_note` into
  `set_plan_user_prompt`; `render_transition` adds the style_bias line to per-pair
  `context`; F8/F10/F11/F12 read their fields when those features land. Everything is
  optional-context — no occasion, no change.
- **Frontend:** chip row + note input in `QueueBuilder.tsx`; shown as a badge on the
  Player render card.

**Tests:** prompt payload includes occasion fields; PATCH validation (unknown occasion
→ 422); presets table sanity (all referenced styles legal).

**Effort:** small. **Depends on:** nothing (other features plug into it later).

---

## F8. Auto track ordering

**Goal.** Order the crate before mixing: minimize key/tempo/energy friction across the
whole set instead of accepting the user's add-order.

**Behavior.** "Suggest order" button in QueueBuilder (unlocked queues). Shows the
proposed order with a per-adjacency compatibility grade (A/B/C) and a one-line reason;
Apply rewrites positions via the existing reorder endpoint logic. Never automatic.

**Design.**
- **Scorer** (`services/mixer/ordering.py`, pure):
  `pair_cost(a: PairInput, b: PairInput) -> float` where PairInput carries bpm, camelot,
  energy_curve summary (mean of last 30 s for A-side, first 30 s for B-side), tags.
  Cost = weighted sum:
  - key: 0 same/relative, 0.5 wheel-neighbor, else `2 + semitone_distance(compute_pitch_shift)`
    (reuses `camelot_compatible` + `plan.py::compute_pitch_shift`),
  - tempo: 0 if gap ≤ 4% (or half-time ratio gap ≤ 4% — F4d aware), linear to 12%,
    step penalty beyond,
  - energy continuity: `|a_end_energy - b_start_energy|`,
  - genre: `1 - jaccard(genres_a, genres_b)` when both tagged, else 0.5 neutral.
- **Search:** exact Held-Karp DP for n ≤ 12 (queue cap is small; 2^12·12² ≈ 600k ops),
  greedy nearest-neighbor + 2-opt polish for larger. Option `pin_first=True` default
  (people usually know their opener).
- **Optional LLM garnish** (behind `use_llm_planner`): send the top-3 orderings with
  costs to the LLM to pick one for *narrative* (title/artist knowledge) — never to
  invent a new order. Skipped on any failure.
- **API:** `POST /api/queues/{id}/suggest_order {pin_first: bool}` → `{order:
  [song_id...], edges: [{from,to,grade,reason}], applied: false}`; caller applies via
  the existing reorder PATCH. Requires all songs analyzed (else 409 with which are
  pending).
- **Frontend:** button + proposal panel in `QueueBuilder.tsx` (diff view: old → new
  position arrows), Apply/Dismiss.

**Tests:** cost function properties (identical songs → ~0; clash+gap → large);
Held-Karp beats greedy on a crafted 6-song instance; pin_first respected; half-time
pair scores as compatible; endpoint 409 on unanalyzed songs.

**Effort:** medium. **Depends on:** F4d's ratio helper (shareable), F7 optional
(occasion energy floor biases the energy term). **Risk:** low — suggestion-only.

---

## F9. Hook teasing

**Goal.** The signature human trick: sneak 2–4 bars of the *next* song's vocal hook
over the current track minutes before the transition, so the drop pays off familiarity.

**Behavior.** Opt-in per queue ("tease hooks" toggle). When the planner finds a fit, B's
recognizable vocal line appears once, beat-matched and key-matched, over an
instrumental pocket late in A; the full transition into B happens normally afterward.

**Design.**
- **Hook finder** (`services/mixer/hooks.py`, pure):
  `find_hook(transcription_segments, aligned_words, sections, energy_curve) ->
  Hook | None` where Hook = `(start, end, text)`. Heuristic: normalize segment texts,
  find the most-repeated line (≥ 2 occurrences, 3–12 words); prefer the occurrence
  inside the highest-energy section; clamp to 2–8 s; extend ends to word boundaries
  from aligned_words. Fallback: first vocal phrase of the highest-energy section.
- **Pocket finder:** on A, a vocal-safe span (existing `safe_regions`) of ≥ tease
  length + 2 s, with `drum_density ≥ 0.5` (envelopes), located in the last 25% of A
  *before* the seam (guarantees it's after the stitch junction into this pair's render:
  the junction lands near the midpoint between the previous crossfade end and this
  seam — enforce `pocket_start > seam_a - 0.25 * a.duration` AND
  `pocket_start > previous-safe margin` computed conservatively as `0.6 * seam_a`).
- **Plan tool:** `vocal_tease {song: "B", stem: "vocals", hook_start, hook_end,
  at_time, bpm_from, bpm_to, semitones, gain: 0.9}` — added to `LEGAL_TOOLS` +
  executor pre-fx on the **A** timeline: load B's vocal stem slice, `pyrb.time_stretch`
  to A's tempo, `pitch_shift` by the camelot-derived semitones (skip the tease entirely
  if |shift| > 2 — reuse `PITCH_SHIFT_CAP`), 50 ms edge fades, add into A's `other`+mix
  at `at_time`, and duck A's `other` stem by 0.7 for the duration (reuse
  `_apply_volume_fade`). Executor already loads B stems — slice before stretch, cheap.
- **Decision:** boolean knob `tease: bool` on `TransitionDecision` (default False).
  The LLM opts in; the *placement* is deterministic (archetypes call the pocket/hook
  finders; if either misses, the tease is silently dropped and logged). Prompt rule:
  use sparingly — max ~2 per set, only when B's hook is genuinely iconic
  (lyrics_preview + your song knowledge).
- **Settings/UI:** `Queue.tease_hooks bool default False` + toggle in QueueBuilder.

**Tests:** hook finder on synthetic repeated-chorus segments; pocket finder respects
junction-safe zone; expansion emits vocal_tease only when both finders hit; executor
places stretched content at `at_time` (mock pyrb, assert slice + placement); tease
dropped when keys need > 2 semitones.

**Effort:** medium-high (the finders are the real work). **Depends on:** F1 useful but
not required. **Risk:** medium — a bad hook pick is *audible*; mitigations: opt-in
toggle, sparse usage rule, F13 QA covers the render.

---

## F10. Arc templates

**Goal.** Named set shapes a human DJ thinks in: slow-burn, peak-early, wave, steady,
sunset (wind down).

**Behavior.** Dropdown at queue lock (defaults from occasion, F7). The set planner
receives the arc; the energy dial (F6) and ordering (F8) respect it.

**Design.**
- **Model:** `Queue.arc_template String nullable` + same migration as F7.
- **Templates** (`services/mixer/occasions.py`, same module): name →
  `{description, energy_targets: list[float]}` where energy_targets is a normalized
  curve sampled per pair index (e.g. wave: [0.4, 0.7, 0.5, 0.8, 0.6, 0.9]).
- **Wiring:** `plan_set` interpolates targets to the actual pair count and includes
  per-pair target energy in `set_plan_user_prompt` (*"pair 3 should land ~0.8"*);
  F8's ordering cost adds `|pair_target - b_peak_energy|` when an arc is set.
- **Frontend:** selector next to the occasion chips.

**Tests:** interpolation (6 targets → 4 pairs); prompt includes targets; ordering bias.

**Effort:** small. **Depends on:** F7 (same migration/UI area), F8 optional.

---

## F11. TTS host / mic drops

**Goal.** A voice. The single biggest "someone is DJing" signal: a set intro, a couple
of tasteful mid-set shout-outs, an outro — ducked under the music like a radio DJ.

**Behavior.** Player render card gains "Host: off / intro only / sparse / chatty" +
persona (hype / late-night / radio / minimal; default from occasion). The stitched mix
opens with e.g. *"…locked in — next hour's all yours, starting with Kaytranada…"* over
song 1's intro, music ducked ~7 dB under the voice, back up when it ends.

**Design.**
- **Config (DECIDED):** `tts_provider: str = "kokoro"` (`kokoro | openai | off`) —
  Kokoro-82M local (Apache-2.0, free, runs on CPU/MPS, no API key) is the default so
  the feature costs nothing out of the box; OpenAI `gpt-4o-mini-tts` available by
  switching the config + key for higher polish. `tts_voice: str` per provider.
  Provider abstraction `services/host/tts.py::synthesize(text, voice) ->
  np.ndarray (44.1k stereo)` so adding providers later is one class.
- **Script** (`services/host/script.py`): one LLM call at stitch time (provider
  already abstracted via `get_llm_provider`): input = ordered songs (title/artist),
  occasion/vibe/arc, transition styles, host_frequency; output JSON:
  `{segments: [{slot: "intro" | pair_index, text, max_seconds}]}`. New
  `HOST_SCRIPT_SYSTEM_PROMPT` in `prompts.py` with per-persona style blocks. Hard
  rules in prompt: ≤ 25 words per segment, no fake facts about the songs, never talk
  over a drop.
- **Placement + mixdown** (`services/host/mixdown.py`, pure numpy): runs inside
  `stitch_queue` after `final_audio` is assembled, before ffmpeg:
  - intro slot: place over song 1's pre-first-vocal region (song 1's vocal-safety data
    — first safe span ≥ clip length + 1 s, else first 15 s at lowered gain);
  - pair slots: inside the transition region (timeline gives output-time spans), start
    at the transition end (B's intro, typically low vocal density); skip the slot if no
    fit — silence is better than talking over a vocal.
  - Ducking: gain envelope 1.0 → 0.45 (200 ms attack) held while |voice| > threshold,
    500 ms release. Voice normalized to -14 dBFS RMS before mix.
  - Every placed segment appended to `QueueRender.timeline["host"]` for the UI.
- **Caching:** TTS clips stored at `host_clips/{sha1(text+voice)}.wav` (add prefix to
  `SWEPT_PREFIXES` + valid-keys logic in `eviction.py`).
- **Failure tolerance:** any exception → log, stitch without voice (mirror timeline's
  best-effort pattern).
- **Frontend:** host controls on the render card; mic icons on the seek bar at host
  moments.

**Tests:** script prompt/parse with StubProvider; placement picks safe spans (fixture
timeline + safety regions); ducking envelope monotonic attack/release; stitch survives
TTS failure; eviction keeps live host clips. Manual listen for voice quality.

**Effort:** large. **Depends on:** F7 (persona defaults) soft; stitcher only.
**Risk:** medium — quality lives or dies on script restraint; start with
`intro_only` as default.

---

## F12. Radio-clean mode — DROPPED (user decision 2026-07-03)

Not being built. Design removed from scope; no profanity detection or vocal-word
muting anywhere in the system.

---

## F13. Render QA + auto-reroll

**Goal.** The DJ listens to their own output. Score every rendered transition; if it's
objectively broken, re-plan with a different style automatically — the washout click
would have been caught here.

**Behavior.** Invisible when all is well. MixPlanDebug UI shows a QA badge + metrics
per transition; a failed render shows "auto-rerolled (was: wash_out — clipping)".

**Design.**
- **Scorer** (`services/mixer/qa.py`, pure — extracted and extended from
  `scripts/eval_transitions.py::_seam_metrics`):
  `score_render(wav_bytes, plan, a_bundle, b_bundle) -> QAReport` with metrics:
  - `rms_delta_db` pre/post seam (2-bar windows) — flag > 8 dB,
  - `clip_ratio` (|x| ≥ 0.999) — flag > 1e-4,
  - `dropout` (any ≥ 400 ms window under -60 dBFS inside the transition window,
    excluding styles that legitimately stop: vinyl_stop/backspin exempt),
  - `click` (99.9th percentile sample-to-sample delta vs song-body baseline, measured
    in the 4 bars around the seam) — flag > 4× baseline,
  - `lowband_onset_corr` (existing) — informational only, no flag.
  Verdict: `pass | warn | fail` (fail = any hard flag).
- **Model:** `MixPlan.qa_metrics JSONB nullable`, `MixPlan.qa_verdict String nullable`
  + migration.
- **Worker:** in `render_transition`, after `render()`: run scorer (~fast, numpy on
  the in-memory wav). `fail` and `attempts < MAX_QA_ATTEMPTS (2)` → log, bump nonce,
  append the failed style to a local `avoid_styles` context list, re-plan + re-render
  once; persist metrics/verdict of the final attempt either way. Never fail the plan
  on QA alone — a `fail` verdict with exhausted retries still ships (with the badge)
  because *some* transition beats a dead render.
- **Planner:** `context["avoid_styles"]` line: "a previous render of this pair failed
  QA using style X — choose a different style."
- **Eval script:** rewire `scripts/eval_transitions.py` to import from
  `services/mixer/qa.py` (single source of truth).
- **Frontend:** badge + metric table in `MixPlanDebug.tsx`.

**Tests:** each metric on synthetic wavs (inject a click / a dropout / clipping / an
8 dB step and assert the right flag); vinyl_stop dropout exemption; worker retry path
(first render fails QA via a monkeypatched scorer → nonce bumped, second accepted);
metrics persisted; eval script still runs.

**Effort:** medium. **Depends on:** nothing — **build first**; F3/F4/F9 all get a
safety net from it. **Risk:** low; worst case is one extra render per bad pair.

---

## Cross-cutting notes

- **Migrations:** bundle the queue-level columns (F7 occasion/vibe_note, F9
  tease_hooks, F10 arc_template) plus MixPlan's F6 energy_bias into one migration;
  `listener_events` (F5) and MixPlan QA columns (F13) are two more. Three total.
- **Prompt budget:** F5/F6/F7/F10 all add context lines to the decision prompt — keep
  each ≤ 2 lines; the candidate menu remains the bulk of the payload.
- **Modal:** none of these touch `modal_stubs.py` — no redeploy needed.
- **Eviction:** F11 adds `host_clips/` to `SWEPT_PREFIXES` + valid keys.

## Decisions (resolved 2026-07-03)

1. **F11 TTS provider:** both `kokoro` and `openai` behind a config switch; default
   `kokoro` so the feature is free to use out of the box.
2. **F12:** dropped entirely — no profanity cleaning anywhere.
3. **F8:** first track pinned by default when suggesting order.
4. **F4:** all four archetypes (double_drop, backspin, breakdown_blend, half-time).
5. **F6:** 90 s lead time accepted; instant skip-to-next-transition button included.
