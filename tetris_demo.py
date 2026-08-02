#!/usr/bin/env python3
"""
Tetris + Symbioid with a **secret byte control map**.

The agent may emit any byte 0..255 each tick. Only a few secret bytes actually
map to left / right / rotate / hard — the rest are dead. The coach never sees
the cipher table; it must discover which bytes do what by watching the world.

Drop quality is also learned from **real locks only** (no simulate_placement
oracle for scoring). Highscores track (game #, score) across top-outs.

Console: quiet by default; pass ``--verbose`` for six-set / event dumps.
On-screen: live Thought count always shown.

Quit: Esc.  R restarts after top-out (also auto-restarts).

Agent memory (Thoughts + Mind + Action command keys only — not board/score/cipher)
is loaded from / saved to ~/.local/share/symbioid/tetris_memory.json by default.
  --no-memory  --reset-memory  --memory PATH

  PYTHONPATH=. .venv/bin/python tetris_demo.py
  PYTHONPATH=. .venv/bin/python tetris_demo.py --verbose
  PYTHONPATH=. .venv/bin/python tetris_demo.py --spectral
  PYTHONPATH=. .venv/bin/python tetris_demo.py --spectral-primary
  PYTHONPATH=. .venv/bin/python tetris_demo.py --headless --games 3 --no-memory
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    import pygame
except ImportError:
    print("pygame required:  .venv/bin/pip install pygame", file=sys.stderr)
    sys.exit(1)

from symbioid import (
    Sensor,
    Symbioid,
    default_memory_path,
    format_six_set_line,
    save_memory,
    set_console_emit,
    try_load_into,
)
from symbioid.world.tetris import (
    VALID_ACTIONS,
    ActionCipher,
    PIECE_COLORS,
    PIECE_NAMES,
    TetrisWorld,
    piece_cells,
)
from symbioid.world.tetris_learn import TetrisCoach, pose_hole_features

# Stable host id so Thought/Action content keys match across runs
HOST_ID = "sym-tetris-byte-learner"
DEFAULT_MEMORY = default_memory_path("tetris_memory.json")


CELL = 28
COLS, ROWS = 10, 20
BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL
SIDE = 280
# Plots under the board: Active / Inactive / Minted Thoughts vs game turns
PLOT_H = 88  # height per plot panel
PLOT_GAP = 6
PLOT_MARGIN = 10
PLOT_HISTORY = 1024  # game turns (piece locks) on the x-axis window
N_PLOTS = 3
MARGIN_X = 20
MARGIN_Y = 20
FOOTER_H = 28
W = BOARD_W + SIDE + MARGIN_X * 2 + 24
H = (
    MARGIN_Y
    + BOARD_H
    + PLOT_MARGIN
    + N_PLOTS * PLOT_H
    + (N_PLOTS - 1) * PLOT_GAP
    + FOOTER_H
    + MARGIN_Y
)
FPS = 30
# Multi-game survival (v0.0.57): throttle dynamics/placement so N growth does not
# peg CPU by game ~6. Order: sample → sparse pulse → decide (optional settle).
CMD_EVERY = 1
SAMPLE_EVERY = 2  # was 1 — fewer cell admits per second
PULSE_EVERY = 4  # was 1 — full-graph pulse less often
PULSES_PRE_CMD = 0  # was 1 — main settle optional
PULSES_ON_LOCK = 1  # was 2
PLACE_EVERY = 3  # recompute network placement every N cmd frames
MID_GAME_GC_EVERY = 80  # frames; hard-cap GC while playing
GRAVITY_INTERVAL = 30  # ~1.0 s/row at 30 FPS with cmd every frame
# Top-out: half second for Innerface queue (was 1.0 s)
RESTART_DELAY_FRAMES = max(12, FPS // 2)
# Face workers: formation drain only; main loop owns pulse (skip_global_pulse)
FACE_TICK_INTERVAL = 0.1  # was 0.02 (~50 Hz) → ~10 Hz


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Symbioid Tetris learning demo")
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable console dumps (six-sets, map events, coach logs). Default: off.",
    )
    p.add_argument(
        "--memory",
        type=Path,
        default=DEFAULT_MEMORY,
        help=f"Agent memory JSON path (Thoughts+Mind only). Default: {DEFAULT_MEMORY}",
    )
    p.add_argument(
        "--no-memory",
        action="store_true",
        help="Do not load or save agent memory.",
    )
    p.add_argument(
        "--reset-memory",
        action="store_true",
        help="Delete memory file before start (fresh Mind; still saves on exit unless --no-memory).",
    )
    p.add_argument(
        "--spectral",
        action="store_true",
        help="Hybrid residual FFT mix + holonomic + phase Hebb (Links still spread).",
    )
    p.add_argument(
        "--spectral-primary",
        action="store_true",
        help="Mode B: FFT mix only (no Link spread/Hebb); implies spectral substrate.",
    )
    p.add_argument(
        "--no-spectral",
        action="store_true",
        help="Force spectral substrate off (graph dynamics only).",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No GUI; run multi-game placement metric and exit.",
    )
    p.add_argument(
        "--games",
        type=int,
        default=3,
        help="With --headless: number of games to play (default 3).",
    )
    p.add_argument(
        "--max-frames",
        type=int,
        default=12000,
        help="With --headless: max frames per game before force end (default 12000).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1,
        help="With --headless: RNG seed for world+coach (default 1).",
    )
    p.add_argument(
        "--eligibility-window",
        type=int,
        default=24,
        help="Cmd-tick eligibility depth for lock credit (default 24). 0 disables.",
    )
    p.add_argument(
        "--no-eligibility",
        action="store_true",
        help="Disable trajectory eligibility credit (landing-cell only, v0.0.33).",
    )
    p.add_argument(
        "--c-streak-bonus",
        type=float,
        default=0.0,
        help="Phase 2: soft board valence bonus while on C-streak (default 0=off).",
    )
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable positive prior on newly minted Action poles (TFT nice).",
    )
    return p.parse_args(argv)


def thought_count(s: Symbioid) -> int:
    return len(s.thoughts)


def thought_counts_active_inactive(s: Symbioid) -> tuple[int, int]:
    """
    Active = Thoughts currently in an Innerface *active* six-set.
    Inactive = other Thoughts still on the host graph (seeds, laws, awareness,
    superseded scaffolding not yet pruned, etc.).

    Snapshots under locks so concurrent face threads cannot mutate during count.
    """
    active_tids: set[str] = set()
    inner = s.innerface
    with inner._local_lock:
        active_ids = list(inner.active_ids.keys())
        for sid in active_ids:
            store = (
                inner.completed_formations.get(sid)
                or inner.completed_syncs.get(sid)
                or inner.completed_integrates.get(sid)
            )
            if store:
                active_tids.update(list(store.keys()))
    with s.graph_lock:
        host_ids = set(s.thoughts.keys())
    n_active = len(active_tids & host_ids)
    n_inactive = max(0, len(host_ids) - n_active)
    return n_active, n_inactive


def cached_graph_intent(
    s: Symbioid,
    world: TetrisWorld,
    coach: TetrisCoach,
    frame: int,
    *,
    place_every: int = PLACE_EVERY,
) -> tuple[str | None, float, list, str]:
    """
    Throttle expensive network placement scoring (v0.0.57).

    Reuses last preferred intent / poles for ``place_every - 1`` frames unless
    a piece just locked (options board changes).
    """
    pe = max(1, int(place_every))
    last_lock = int(getattr(coach, "last_lock_frame", -999) or -999)
    force = (frame - last_lock) <= 1
    cache = getattr(coach, "_intent_cache", None)
    if (
        not force
        and isinstance(cache, tuple)
        and len(cache) == 5
        and (frame - int(cache[0])) < pe
    ):
        return cache[1], float(cache[2]), list(cache[3]), str(cache[4])
    preferred, g_bias, poles, hint = graph_preferred_intent(s, world, coach)
    coach._intent_cache = (frame, preferred, g_bias, poles, hint or "")  # type: ignore[attr-defined]
    return preferred, float(g_bias), poles, str(hint or "")


def game_boundary_gc(
    s: Symbioid,
    *,
    max_forget_passes: int = 16,
    hard_cap: int = 11000,
) -> dict[str, int]:
    """
    Multi-game survival GC (v0.0.57): unprotect cell-map registry, prune scaffolds,
    cold-forget, then hard-cap host graph if still above ``hard_cap``.

    Call on top-out / between headless games. Returns removal stats.
    """
    with s.graph_lock:
        before = len(s.thoughts)
    mind = s.mind
    old_need = int(getattr(mind, "forget_cold_cycles", 64) or 64)
    old_max = int(getattr(mind, "forget_max_per_pass", 64) or 64)
    mind.forget_cold_cycles = 4
    mind.forget_max_per_pass = max(old_max, 4096)
    purged = 0
    pruned = 0
    forgotten = 0
    hard = 0
    try:
        if hasattr(mind, "purge_ephemeral_registry"):
            purged = int(mind.purge_ephemeral_registry() or 0)
        # Drop inactive six-set membership so prune can free poles
        with s.innerface._local_lock:
            # Keep only a small active window; clear archived inactive stores' poles via prune
            pass
        if hasattr(s.innerface, "prune_inactive_thoughts"):
            pruned = int(s.innerface.prune_inactive_thoughts() or 0)
        for _ in range(max(1, int(max_forget_passes))):
            n = int(s.innerface.forget_cold_thoughts() or 0)
            forgotten += n
            if n == 0:
                break
        if hasattr(s.innerface, "prune_inactive_thoughts"):
            pruned += int(s.innerface.prune_inactive_thoughts() or 0)
        # Hard cap: remove oldest cold unprotected Thoughts
        with s.graph_lock:
            n_now = len(s.thoughts)
        if n_now > int(hard_cap):
            protected = s.innerface._protected_thought_ids()
            # Always keep actions / twin-ish ids
            with mind._lock:
                for t in mind._actions.values():
                    if t is not None:
                        protected.add(t.id)
            victims: list[tuple[int, str]] = []
            with s.graph_lock:
                for tid, th in list(s.thoughts.items()):
                    if tid in protected:
                        continue
                    last = int(getattr(th, "last_hot_cycle", -1) or -1)
                    victims.append((last, tid))
                victims.sort(key=lambda x: x[0])  # oldest hot first (-1 first)
                need = n_now - int(hard_cap)
                for _, tid in victims[: max(0, need)]:
                    if s._remove_thought_unlocked(tid) is not None:
                        hard += 1
    finally:
        mind.forget_cold_cycles = old_need
        mind.forget_max_per_pass = old_max
    with s.graph_lock:
        after = len(s.thoughts)
    return {
        "thoughts_before": before,
        "thoughts_after": after,
        "pruned": pruned,
        "forgotten": forgotten,
        "purged_registry": purged,
        "hard_cap": hard,
        "removed": max(0, before - after),
    }


def build_symbioid(
    world: TetrisWorld,
    *,
    spectral: bool = False,
    spectral_primary: bool = False,
) -> Symbioid:
    """
    Sensors: full 10×20 cell map (block / hole / open) + meta
    (piece_id, next_id, lines, last_byte, holes_n, last_d_holes).

    Cell sensors use awareness=False (terminator only) to avoid 200 full
    awareness six-sets. Sampling is change-only (see sample_into_symbioid).
    Packing meta (holes_n / last_d_holes) gives the Thought graph explicit
    insight into sealed-hole count and last lock's hole delta.

    spectral / spectral_primary: Mind FFT substrate (hybrid residual or Mode B).
    Default (neither): graph-only dynamics for stable baseline play.
    """
    s = Symbioid(id=HOST_ID, label="tetris-byte-learner")
    s.interface.continuous_inputs = False
    s.outerface.wait_for_feedback = False

    if spectral_primary or spectral:
        s.mind.enable_spectral_demo(
            phase_hebb=True, primary=bool(spectral_primary)
        )
    else:
        s.mind.set_dynamics_mode("graph")
        s.mind.spectral_mix_enabled = False
        s.mind.holonomic_store_enabled = False
        s.mind.hebb_phase_enabled = False

    # Face workers: formation drain only; main loop owns full-graph pulse
    s.interface.tick_interval = float(FACE_TICK_INTERVAL)
    s.innerface.tick_interval = float(FACE_TICK_INTERVAL)
    s.outerface.tick_interval = float(FACE_TICK_INTERVAL)
    s.interface.skip_global_pulse = True  # type: ignore[attr-defined]
    # Multi-game survival: forget sooner + larger batches when GC runs
    s.mind.forget_cold_cycles = 24
    s.mind.forget_max_per_pass = 256
    # Learning structure (P0): avoid cell co-fire storms / zombie syncs.
    # Band B (research active-thoughts theory): serious network-primary WM +
    # larger policy registries with act:-preferring hard eviction.
    s.innerface.cofire_meta_only = True
    s.innerface.allow_cross_channel_follows = False
    s.innerface.max_active_syncs = 112
    s.innerface.max_active_senses = 224
    s.innerface.max_active_integrates = 112
    s.innerface.max_active_integrates_per_channel = 8
    # Packing meta participates in co-fire / policy poles
    s.innerface.cofire_meta_labels = tuple(
        dict.fromkeys(
            list(s.innerface.cofire_meta_labels)
            + [
                "holes_n",
                "last_d_holes",
                "well_n",
                "max_well_n",
                "pred_d_holes",
                "holes_freed",
                "holes_fill_n",
            ]
        )
    )
    s.mind.max_follows_registry = 4096
    s.mind.max_integrates_registry = 4096
    s.mind.policy_registry_priority = True
    # Held packing readings for sensor.transfer (updated on sample / lock / target)
    s._pack_holes = 0.0  # type: ignore[attr-defined]
    s._last_d_holes = 0.0  # type: ignore[attr-defined]
    s._pack_well = 0.0  # type: ignore[attr-defined]
    s._pack_max_well = 0.0  # type: ignore[attr-defined]
    # Foresight for current placement target (how many holes this drop frees/creates)
    s._pred_d_holes = 0.0  # type: ignore[attr-defined]
    s._holes_freed = 0.0  # type: ignore[attr-defined]  # max(0, -pred_d_holes)
    s._holes_fill_n = 0.0  # type: ignore[attr-defined]  # landing cells that are holes

    # Last cell readings for change-only formation (sensor_id → float)
    s._cell_last_reading: dict[str, float] = {}  # type: ignore[attr-defined]
    s._cell_rc: dict[str, tuple[int, int]] = {}  # type: ignore[attr-defined]
    # Reverse index for ROI / dirty-rect sampling (avoid walking all 200 sensors)
    s._cell_by_rc: dict[tuple[int, int], object] = {}  # type: ignore[attr-defined]
    s._cell_prev_active: set[tuple[int, int]] = set()  # type: ignore[attr-defined]
    s._cell_last_lines: int = 0  # type: ignore[attr-defined]

    # Full playfield map: one sensor per cell (stable ids; no full awareness)
    for r in range(world.rows):
        for c in range(world.cols):
            label = f"cell_r{r:02d}_c{c:02d}"

            def _cell_xfer(
                _w: dict,
                wo: TetrisWorld = world,
                row: int = r,
                col: int = c,
            ) -> float:
                return wo.cell_reading(row, col, with_active=True)

            sen = s.add_sensor(
                Sensor(id=f"{HOST_ID}:sen:{label}", label=label),
                awareness=False,
            )
            sen.transfer = _cell_xfer
            s._cell_rc[sen.id] = (r, c)  # type: ignore[attr-defined]
            s._cell_by_rc[(r, c)] = sen  # type: ignore[attr-defined]

    def piece_id_n(_w: dict, wo: TetrisWorld = world) -> float:
        if wo.active is None:
            return 0.0
        return PIECE_NAMES.index(wo.active.kind) / 6.0

    def next_id_n(_w: dict, wo: TetrisWorld = world) -> float:
        kind = getattr(wo, "next_kind", None)
        if kind is None or kind not in PIECE_NAMES:
            return 0.0
        return PIECE_NAMES.index(kind) / 6.0

    def holes_n(_w: dict, _s: Symbioid = s) -> float:
        # Absolute sealed-hole count, scaled (~0..1 for typical boards)
        return min(1.0, max(0.0, float(getattr(_s, "_pack_holes", 0.0)) / 40.0))

    def last_d_holes_n(_w: dict, _s: Symbioid = s) -> float:
        # Last lock Δholes, compressed to [-1, 1] (÷8 holes ≈ full-scale)
        return max(-1.0, min(1.0, float(getattr(_s, "_last_d_holes", 0.0)) / 8.0))

    def well_n(_w: dict, _s: Symbioid = s) -> float:
        return min(1.0, max(0.0, float(getattr(_s, "_pack_well", 0.0)) / 40.0))

    def max_well_n(_w: dict, _s: Symbioid = s) -> float:
        return min(1.0, max(0.0, float(getattr(_s, "_pack_max_well", 0.0)) / 20.0))

    def pred_d_holes_n(_w: dict, _s: Symbioid = s) -> float:
        # Predicted net Δholes for current target pose (negative = frees holes)
        return max(-1.0, min(1.0, float(getattr(_s, "_pred_d_holes", 0.0)) / 8.0))

    def holes_freed_n(_w: dict, _s: Symbioid = s) -> float:
        # How many holes this target placement is predicted to free (net)
        return min(1.0, max(0.0, float(getattr(_s, "_holes_freed", 0.0)) / 8.0))

    def holes_fill_n(_w: dict, _s: Symbioid = s) -> float:
        # Existing hole cells covered by target landing footprint
        return min(1.0, max(0.0, float(getattr(_s, "_holes_fill_n", 0.0)) / 4.0))

    for label, transfer in (
        ("piece_id", piece_id_n),
        ("next_id", next_id_n),
        ("lines", lambda w, wo=world: min(1.0, wo.lines / 50.0)),
        ("last_byte", lambda w, wo=world: wo.last_byte / 255.0),
        ("holes_n", holes_n),
        ("last_d_holes", last_d_holes_n),
        ("well_n", well_n),
        ("max_well_n", max_well_n),
        ("pred_d_holes", pred_d_holes_n),
        ("holes_freed", holes_freed_n),
        ("holes_fill_n", holes_fill_n),
    ):
        sen = s.add_sensor(Sensor(id=f"{HOST_ID}:sen:{label}", label=label))
        sen.transfer = transfer

    from symbioid import Actuator

    out = s.add_actuator(Actuator(id=f"{HOST_ID}:act:byte", label="byte"))
    out.output = 0.0
    out.output_step = 1.0 / 255.0

    # Phase B: seed Action poles so recommend_action can hit left/right/rotate/hard
    for tok in VALID_ACTIONS:
        th = s.mind.ensure_action_thought("tetris", tok, host_id=s.id, with_labels=True)
        s.add_thought(th)
    return s


# Meta sensor labels used as stable policy state (always sampled)
_POLICY_META_LABELS = frozenset(
    {
        "piece_id",
        "next_id",
        "lines",
        "last_byte",
        "holes_n",
        "last_d_holes",
        "well_n",
        "max_well_n",
        "pred_d_holes",
        "holes_freed",
        "holes_fill_n",
    }
)
_PACKING_META_LABELS = frozenset(
    {
        "holes_n",
        "last_d_holes",
        "well_n",
        "max_well_n",
        "pred_d_holes",
        "holes_freed",
        "holes_fill_n",
    }
)
_FORESIGHT_META_LABELS = frozenset(
    {"pred_d_holes", "holes_freed", "holes_fill_n"}
)


def policy_state_poles(s: Symbioid, world: TetrisWorld) -> list:
    """
    Phase B state for Mind.recommend_action — not the full 200-cell dump.

    Includes:
      - last Observations for meta sensors (piece/next/lines/byte)
      - registered Mind Observations for those channels
      - Observations for cells currently occupied by the active piece (if known)
    """
    poles: list = []
    seen: set[str] = set()

    def _add(t) -> None:
        if t is None or t.id in seen:
            return
        seen.add(t.id)
        poles.append(t)

    # Last obs by sensor id (innerface map)
    with s.innerface._local_lock:
        last_by = dict(s.innerface._last_obs_by_sensor)
    label_by_sid = {sen.id: (sen.label or "") for sen in s.sensors}
    for sid, obs in last_by.items():
        lab = label_by_sid.get(sid, "")
        if lab in _POLICY_META_LABELS or lab.startswith("cell_"):
            # Prefer meta always; cells only if active-occupied (filtered below)
            if lab in _POLICY_META_LABELS:
                _add(obs)

    # Mind registry: meta content keys + any key matching meta labels
    with s.mind._lock:
        for ck, th in s.mind._observations.items():
            if any(m in ck for m in _POLICY_META_LABELS):
                _add(th)

    # Active piece cells → include matching last-obs / registry poles
    active_rc: set[tuple[int, int]] = set()
    if world.active is not None:
        for r, c in world.active.cells():
            if 0 <= r < world.rows and 0 <= c < world.cols:
                active_rc.add((r, c))
    cell_rc = getattr(s, "_cell_rc", {}) or {}
    for sid, (r, c) in cell_rc.items():
        if (r, c) not in active_rc:
            continue
        obs = last_by.get(sid)
        if obs is not None:
            _add(obs)
        # registry by sensor id fragment
        with s.mind._lock:
            for ck, th in s.mind._observations.items():
                if sid in ck or f"cell_r{r:02d}_c{c:02d}" in ck:
                    _add(th)

    # Fallback: any last obs if still empty (early game)
    if not poles:
        for obs in last_by.values():
            _add(obs)
    return poles


def _cell_obs_index(s: Symbioid) -> dict[str, list[tuple[str, object]]]:
    """
    Map ``cell_rXX_cYY`` label → [(content_key, observation Thought), ...].

    Rebuilt when Mind observation count changes (O(|obs|) once per generation).
    """
    import re

    n = 0
    with s.mind._lock:
        n = len(s.mind._observations)
        if (
            getattr(s, "_cell_obs_gen", None) == n
            and getattr(s, "_cell_obs_index", None) is not None
        ):
            return s._cell_obs_index  # type: ignore[return-value]
        idx: dict[str, list[tuple[str, object]]] = {}
        pat = re.compile(r"cell_r\d{2}_c\d{2}")
        for ck, oth in s.mind._observations.items():
            m = pat.search(str(ck))
            if m:
                idx.setdefault(m.group(0), []).append((str(ck), oth))
    s._cell_obs_index = idx  # type: ignore[attr-defined]
    s._cell_obs_gen = n  # type: ignore[attr-defined]
    return idx


# Credit hygiene (research 2026-07-31 score regression):
# board_quality_reward is usually negative → sticky valence_floor kills :place map.
CREDIT_SCALE = 50.0
CREDIT_NEG_SCALE = 0.35  # multiply |delta| when reward < 0
CREDIT_SKIP_PLACE_ON_TOPOUT = True
PLACE_VALENCE_LEAK = 0.06  # each lock: place_v *= (1 - leak) → pull toward 0
# Phase 2: optional soft reward for cooperative streak (default off)
C_STREAK_BONUS = 0.0


def credit_delta(
    reward: float,
    *,
    scale: float = CREDIT_SCALE,
    neg_scale: float = CREDIT_NEG_SCALE,
) -> float:
    """
    Scale coach reward into valence delta with **asymmetric** negatives.

    Positive rewards use full scale; negatives are attenuated so place heat
    does not floor permanently under height/hole-shaped rewards.
    """
    raw = float(reward) / float(scale)
    if raw < 0.0:
        raw *= float(neg_scale)
    return max(-2.0, min(2.0, raw))


def leak_place_valence(s: Symbioid, *, rate: float = PLACE_VALENCE_LEAK) -> int:
    """
    Soft mean-reversion for ``*:place`` keys toward 0.

    Returns number of keys adjusted. Rate 0 disables.
    """
    r = float(rate)
    if r <= 0.0:
        return 0
    r = min(1.0, r)
    touched = 0
    with s.mind._lock:
        keys = [k for k in s.mind._valence if str(k).endswith(":place") or ":place" in str(k)]
        for ck in keys:
            v = float(s.mind._valence.get(ck, 0.0))
            if abs(v) < 1e-12:
                continue
            nv = v * (1.0 - r)
            if abs(nv) < 1e-6:
                nv = 0.0
            s.mind._valence[ck] = max(
                s.mind.valence_floor, min(s.mind.valence_ceil, nv)
            )
            touched += 1
    return touched


def apply_lock_valence_to_landing_cells(
    s: Symbioid,
    cells: list[tuple[int, int]],
    reward: float,
    *,
    scale: float = CREDIT_SCALE,
    neg_scale: float = CREDIT_NEG_SCALE,
    apply_place_keys: bool = True,
) -> int:
    """
    Closed-loop placement credit: fan coach board reward onto Observation
    valence for cells occupied by the piece at lock.

    Credit hygiene: asymmetric negatives; optional skip of synthetic ``:place``
    keys (e.g. on top-out). Returns number of content keys touched.
    """
    if not cells:
        return 0
    delta = credit_delta(reward, scale=scale, neg_scale=neg_scale)
    if abs(delta) < 1e-12:
        return 0
    lab_index = _cell_obs_index(s)
    rc_to_sid = {rc: sid for sid, rc in (getattr(s, "_cell_rc", {}) or {}).items()}
    with s.innerface._local_lock:
        last_by = dict(s.innerface._last_obs_by_sensor)
    touched = 0
    seen_ck: set[str] = set()
    for r, c in cells:
        lab = f"cell_r{int(r):02d}_c{int(c):02d}"
        for ck, _oth in lab_index.get(lab, ()):
            if ck in seen_ck:
                continue
            seen_ck.add(ck)
            s.mind.note_valence(content_key=str(ck), delta=delta)
            touched += 1
        sid = rc_to_sid.get((int(r), int(c)))
        th = last_by.get(sid) if sid else None
        if th is not None:
            with s.mind._lock:
                ck = s.mind._thought_to_key.get(th.id)
            if ck and ck not in seen_ck:
                seen_ck.add(ck)
                s.mind.note_valence(content_key=str(ck), delta=delta)
                touched += 1
            elif not ck:
                # Direct thought_id path if registry key missing
                s.mind.note_valence(thought_id=th.id, delta=delta)
                touched += 1
        # Stable synthetic key so cold cells still accumulate placement signal
        if apply_place_keys:
            synth = f"{lab}:place"
            if synth not in seen_ck:
                seen_ck.add(synth)
                s.mind.note_valence(content_key=synth, delta=delta * 0.5)
                touched += 1
    return touched


@dataclass
class EligibilityWindow:
    """
    Trajectory eligibility for placement credit (P1).

    Each command tick pushes content keys of state poles that influenced
    placement/policy. On lock, reward fans onto those keys with **linear
    recency weights** (newest ≈ 1.0, oldest ≈ 1/N) in addition to landing-cell
    valence (v0.0.33).
    """

    max_ticks: int = 24
    frames: list[set[str]] = field(default_factory=list)

    def push(self, keys: Iterable[str]) -> None:
        if self.max_ticks <= 0:
            return
        batch = {str(k) for k in keys if k}
        if not batch:
            return
        self.frames.append(batch)
        overflow = len(self.frames) - int(self.max_ticks)
        if overflow > 0:
            del self.frames[:overflow]

    def clear(self) -> None:
        self.frames.clear()

    def credited_keys(self) -> dict[str, float]:
        """Map content_key → recency weight in (0, 1]."""
        n = len(self.frames)
        if n == 0:
            return {}
        acc: dict[str, float] = {}
        for i, keys in enumerate(self.frames):
            w = (i + 1) / float(n)
            for k in keys:
                prev = acc.get(k, 0.0)
                if w > prev:
                    acc[k] = w
        return acc

    def __len__(self) -> int:
        return len(self.frames)


def poles_to_content_keys(s: Symbioid, poles: list) -> set[str]:
    """Map Thought poles → Mind content keys (skip unregistered)."""
    keys: set[str] = set()
    with s.mind._lock:
        t2k = dict(s.mind._thought_to_key)
        recent = list(s.mind._recent_keys[-24:])
    for th in poles:
        if th is None:
            continue
        ck = t2k.get(getattr(th, "id", None))
        if ck:
            keys.add(str(ck))
    # Recent packing-meta keys (holes_n, pred_d_holes, …) if freshly sampled
    for ck in recent:
        cks = str(ck)
        if any(m in cks for m in _PACKING_META_LABELS):
            keys.add(cks)
    return keys


def apply_eligibility_valence(
    s: Symbioid,
    window: EligibilityWindow,
    reward: float,
    *,
    scale: float = CREDIT_SCALE,
    strength: float = 0.55,
    neg_scale: float = CREDIT_NEG_SCALE,
) -> int:
    """
    Fan lock reward onto eligibility-window keys with recency decay.

    Uses the same asymmetric ``credit_delta`` as landing-cell credit.
    ``strength`` scales relative to full landing-cell credit (1.0). Returns
    number of distinct content keys touched.
    """
    if window.max_ticks <= 0 or not window.frames:
        return 0
    delta = credit_delta(reward, scale=scale, neg_scale=neg_scale)
    if abs(delta) < 1e-12:
        return 0
    credited = window.credited_keys()
    if not credited:
        return 0
    touched = 0
    for ck, w in credited.items():
        s.mind.note_valence(
            content_key=str(ck),
            delta=float(delta) * float(strength) * float(w),
        )
        touched += 1
    return touched


def apply_lock_credit(
    s: Symbioid,
    coach: TetrisCoach,
    window: EligibilityWindow,
    *,
    poles: list | None = None,
    push_poles: bool = False,
    skip_place_on_topout: bool = CREDIT_SKIP_PLACE_ON_TOPOUT,
    place_leak: float = PLACE_VALENCE_LEAK,
    neg_scale: float = CREDIT_NEG_SCALE,
) -> dict[str, int]:
    """
    Full P1 lock credit + P0 hygiene: asymmetric negatives, optional skip of
    ``:place`` keys on top-out, and soft ``:place`` valence leak toward 0.

    Callers that already ``window.push`` each cmd tick should leave
    ``push_poles=False`` (default) to avoid duplicate last-frame entries.
    Clears the window after apply so the next piece starts fresh.
    """
    stats: dict[str, Any] = {
        "landing": 0,
        "eligibility": 0,
        "pushed": 0,
        "place_leak": 0,
        "skipped_place": 0,
        "topped_out": 0,
        "grudge_keys": [],
        "forgiven": 0,
        "round": "",
    }
    reward = float(getattr(coach, "last_reward", 0.0) or 0.0)
    topped = bool(getattr(coach, "last_topped_out", False))
    if topped:
        stats["topped_out"] = 1
    apply_place = not (skip_place_on_topout and topped)
    if not apply_place:
        stats["skipped_place"] = 1
    if push_poles and poles:
        keys = poles_to_content_keys(s, poles)
        before = len(window)
        window.push(keys)
        stats["pushed"] = 1 if len(window) > before else 0
    lock_cells = list(getattr(coach, "last_lock_cells", None) or [])
    # Candidate grudge keys (place synth + eligibility) before window clear
    grudge: set[str] = set()
    for r, c in lock_cells:
        grudge.add(f"cell_r{int(r):02d}_c{int(c):02d}:place")
    grudge.update(window.credited_keys().keys())
    if lock_cells:
        stats["landing"] = apply_lock_valence_to_landing_cells(
            s,
            lock_cells,
            reward,
            neg_scale=neg_scale,
            apply_place_keys=apply_place,
        )
    # Eligibility: on top-out skip entirely (same death-spiral as place keys)
    if not topped:
        stats["eligibility"] = apply_eligibility_valence(
            s, window, reward, neg_scale=neg_scale
        )
    else:
        stats["eligibility"] = 0
    stats["place_leak"] = leak_place_valence(s, rate=place_leak)
    window.clear()
    # v0.0.53 iterated twin: C/D round labels + forgiveness
    if topped:
        stats["round"] = "D_env"
        stats["grudge_keys"] = sorted(grudge)
        # Physics end-of-game; System also participated via last place (grudge keys).
        s.mind.note_round(
            "D_env",
            source="self",
            channel="tetris",
            keys=grudge,
        )
    else:
        stats["round"] = "C"
        s.mind.note_round("C", source="env", channel="tetris")
        # Optional soft shaping: tiny positive board valence while on a C-streak
        bonus = float(C_STREAK_BONUS)
        if bonus > 0.0:
            streak = int(getattr(s.mind.tft, "c_streak", 0) or 0)
            if streak >= 2:
                s.mind.note_valence(
                    channel="board",
                    delta=bonus * min(1.0, streak / 10.0),
                    recent=6,
                )
        fg = s.mind.maybe_forgive()
        stats["forgiven"] = int(fg.get("forgiven", 0) or 0)
    return stats


@dataclass
class GameMetric:
    """Per-game packing metric snapshot (multi-game harness)."""

    game: int
    score: int
    lines: int
    pieces: int
    holes: int
    max_height: int
    aggregate_height: int
    frames: int
    top_out: bool
    n_C: int = 0
    n_D: int = 0
    n_U: int = 0
    tft_state: str = "open"
    forgives: int = 0

    def as_dict(self) -> dict:
        return {
            "game": self.game,
            "score": self.score,
            "lines": self.lines,
            "pieces": self.pieces,
            "holes": self.holes,
            "max_height": self.max_height,
            "aggregate_height": self.aggregate_height,
            "frames": self.frames,
            "top_out": self.top_out,
            "n_C": self.n_C,
            "n_D": self.n_D,
            "n_U": self.n_U,
            "tft_state": self.tft_state,
            "forgives": self.forgives,
        }


def summarize_game_metrics(rows: list[GameMetric]) -> dict[str, float]:
    """Means across games for headless reporting / tests."""
    if not rows:
        return {
            "n": 0.0,
            "mean_score": 0.0,
            "mean_lines": 0.0,
            "mean_holes": 0.0,
            "mean_max_height": 0.0,
            "mean_pieces": 0.0,
            "mean_frames": 0.0,
            "mean_C": 0.0,
            "mean_D": 0.0,
            "c_rate": 0.0,
            "top_out_rate": 0.0,
        }
    n = float(len(rows))
    sum_c = sum(r.n_C for r in rows)
    sum_d = sum(r.n_D for r in rows)
    rounds = float(sum_c + sum_d) or 1.0
    top_outs = sum(1 for r in rows if r.top_out)
    return {
        "n": n,
        "mean_score": sum(r.score for r in rows) / n,
        "mean_lines": sum(r.lines for r in rows) / n,
        "mean_holes": sum(r.holes for r in rows) / n,
        "mean_max_height": sum(r.max_height for r in rows) / n,
        "mean_pieces": sum(r.pieces for r in rows) / n,
        "mean_frames": sum(r.frames for r in rows) / n,
        "mean_C": sum_c / n,
        "mean_D": sum_d / n,
        "c_rate": sum_c / rounds,
        "top_out_rate": top_outs / n,
    }


def run_multi_game_metric(
    *,
    games: int = 3,
    max_frames: int = 12000,
    seed: int = 1,
    eligibility_window: int = 24,
    use_eligibility: bool = True,
    spectral: bool = False,
    spectral_primary: bool = False,
    map_threshold: int = 1,
    verbose: bool = False,
    mind_setup: Any = None,
) -> tuple[list[GameMetric], dict[str, float]]:
    """
    Headless multi-game placement metric (no pygame display).

    Plays ``games`` full top-outs (or max_frames each), applying lock credit
    with optional eligibility. Returns (per-game rows, summary means).

    ``mind_setup`` optional callable(s) after ``build_symbioid`` (Phase 4 A/B).
    """
    import random as _random

    # Headless must stay quiet — six-set console dumps destroy FPS
    set_console_emit(False)
    log = print if verbose else (lambda *a, **k: None)
    rng = _random.Random(int(seed))
    cipher = ActionCipher.random(rng)
    world = TetrisWorld(
        cols=COLS,
        rows=ROWS,
        gravity_interval=GRAVITY_INTERVAL,
        cipher=cipher,
        rng=_random.Random(int(seed) + 17),
    )
    coach = TetrisCoach(
        network_primary=True,
        map_threshold=int(map_threshold),
        rng=_random.Random(int(seed) + 31),
    )
    s = build_symbioid(
        world, spectral=bool(spectral), spectral_primary=bool(spectral_primary)
    )
    if callable(mind_setup):
        mind_setup(s)
    coach.graph_placement_weight = 0.60
    coach.graph_placement_bonus = CachedGraphPlacementBonus(s)
    win_n = int(eligibility_window) if use_eligibility else 0
    elig = EligibilityWindow(max_ticks=win_n)
    s.start_processes()
    rows: list[GameMetric] = []
    try:
        for g in range(1, int(games) + 1):
            frame = 0
            last_cmd_poles: list = []
            elig.clear()
            s.mind.tft.reset_episode(clear_counts=True)
            while not world.game_over and frame < int(max_frames):
                if frame % SAMPLE_EVERY == 0:
                    sample_into_symbioid(s, world, tick=frame)
                if s.mind.dynamics_enabled and frame % PULSE_EVERY == 0:
                    s.pulse_tick()
                if s.mind.dynamics_enabled and PULSES_PRE_CMD > 0:
                    for _ in range(int(PULSES_PRE_CMD)):
                        s.pulse_tick()
                if MID_GAME_GC_EVERY > 0 and frame > 0 and frame % MID_GAME_GC_EVERY == 0:
                    game_boundary_gc(s, max_forget_passes=4, hard_cap=11000)
                prev_pieces = world.pieces_placed
                preferred, g_bias, poles, _hint = cached_graph_intent(
                    s, world, coach, frame, place_every=PLACE_EVERY
                )
                update_pred_pack_for_target(s, world, coach)
                sample_packing_meta_into_symbioid(
                    s,
                    world,
                    tick=frame,
                    coach=coach,
                    labels=_FORESIGHT_META_LABELS,
                )
                last_cmd_poles = poles
                if win_n > 0:
                    elig.push(poles_to_content_keys(s, poles))
                coach.tick(
                    world,
                    preferred_intent=preferred,
                    graph_bias=g_bias,
                )
                if world.pieces_placed > prev_pieces:
                    coach.last_lock_frame = frame  # type: ignore[attr-defined]
                    lock_eff = getattr(coach, "last_lock_effect", "") or ""
                    intent = (
                        lock_eff
                        if lock_eff in VALID_ACTIONS
                        else (
                            coach.last_intent
                            if coach.last_intent in VALID_ACTIONS
                            else None
                        )
                    )
                    if intent is not None:
                        s.mind.record_outcome(
                            last_cmd_poles,
                            intent,
                            domain="tetris",
                            host_id=s.id,
                            reward=float(coach.last_reward),
                            host=s,
                        )
                    s.mind.note_valence(
                        channel="board",
                        delta=max(
                            -2.0, min(2.0, float(coach.last_reward) / 50.0)
                        ),
                    )
                    apply_lock_credit(
                        s, coach, elig, poles=last_cmd_poles
                    )
                    s._last_d_holes = float(  # type: ignore[attr-defined]
                        getattr(coach, "last_d_holes", 0.0) or 0.0
                    )
                    sample_packing_meta_into_symbioid(
                        s, world, tick=frame, coach=coach
                    )
                    if s.mind.dynamics_enabled and PULSES_ON_LOCK > 0:
                        for _ in range(int(PULSES_ON_LOCK)):
                            s.pulse_tick()
                frame += 1
            # End-of-game packing snapshot (board at top-out or frame cap)
            snap = s.mind.tft_snapshot()
            cnt = snap.get("counts") or {}
            with s.graph_lock:
                n_th = len(s.thoughts)
            rows.append(
                GameMetric(
                    game=g,
                    score=int(world.score),
                    lines=int(world.lines),
                    pieces=int(world.pieces_placed),
                    holes=int(world.hole_count()),
                    max_height=int(world.max_height()),
                    aggregate_height=int(world.aggregate_height()),
                    frames=frame,
                    top_out=bool(world.game_over),
                    n_C=int(cnt.get("C", 0) or 0),
                    n_D=int(cnt.get("D", 0) or 0),
                    n_U=int(cnt.get("U", 0) or 0),
                    tft_state=str(snap.get("tft_state", "open")),
                    forgives=int(snap.get("forgives", 0) or 0),
                )
            )
            log(
                f"[metric] g={g} score={world.score} lines={world.lines} "
                f"holes={world.hole_count()} maxH={world.max_height()} "
                f"pieces={world.pieces_placed} frames={frame} "
                f"th={n_th} "
                f"C={cnt.get('C', 0)} D={cnt.get('D', 0)} "
                f"tft={snap.get('tft_state')} forgives={snap.get('forgives', 0)}",
                flush=True,
            )
            # Multi-game survival: GC before next game
            gc_stats = game_boundary_gc(s)
            log(
                f"[gc] g={g} th={gc_stats.get('thoughts_before')}→"
                f"{gc_stats.get('thoughts_after')} "
                f"rm={gc_stats.get('removed')} prune={gc_stats.get('pruned')} "
                f"forget={gc_stats.get('forgotten')}",
                flush=True,
            )
            if g < int(games):
                coach.on_new_game(world, record=True)
    finally:
        s.stop_processes()
    summary = summarize_game_metrics(rows)
    log(
        f"[summary] n={summary.get('n')} mean_score={summary.get('mean_score'):.1f} "
        f"mean_frames={summary.get('mean_frames', 0):.0f} "
        f"top_out_rate={summary.get('top_out_rate', 0):.2f} "
        f"mean_C={summary.get('mean_C', 0):.1f} mean_D={summary.get('mean_D', 0):.1f} "
        f"c_rate={summary.get('c_rate', 0):.3f}",
        flush=True,
    )
    return rows, summary


@dataclass
class PlacementScoreContext:
    """Shared board/Mind lookups for scoring many poses (Phase 3)."""

    s: Symbioid
    world: TetrisWorld
    field: list
    valence: dict
    t2k: dict
    last_by: dict
    lab_index: dict
    rc_to_sid: dict
    pre_holes: float
    pre_wm: dict


def build_placement_score_context(s: Symbioid, world: TetrisWorld) -> PlacementScoreContext:
    """One-shot locks + board features for a choose_target batch."""
    from symbioid.world.tetris import well_metrics

    field = world.cell_field_state(with_active=False)
    rc_to_sid = {rc: sid for sid, rc in (getattr(s, "_cell_rc", {}) or {}).items()}
    with s.innerface._local_lock:
        last_by = dict(s.innerface._last_obs_by_sensor)
    with s.mind._lock:
        valence = dict(s.mind._valence)
        t2k = dict(s.mind._thought_to_key)
    lab_index = _cell_obs_index(s)
    pre_holes = float(world.hole_count())
    pre_wm = well_metrics(world.column_heights())
    return PlacementScoreContext(
        s=s,
        world=world,
        field=field,
        valence=valence,
        t2k=t2k,
        last_by=last_by,
        lab_index=lab_index,
        rc_to_sid=rc_to_sid,
        pre_holes=pre_holes,
        pre_wm=pre_wm,
    )


def score_pose_with_context(
    ctx: PlacementScoreContext,
    rot: int,
    col: int,
    *,
    cells: list[tuple[int, int]] | None = None,
    sim: dict | None = None,
) -> float:
    """
    Phase C score for one landing using a shared context.

    Prefers filling holes (0.5), lower d_holes, deeper rows, high Mind valence.
    Illegal landings return a large penalty.

    Phase 3B: pass ``cells`` + ``sim`` from one hard-drop to avoid double drop.
    """
    world = ctx.world
    if cells is None or sim is None:
        cells2, sim2 = world.landing_cells_and_features(rot, col)
        if cells is None:
            cells = cells2
        if sim is None:
            sim = sim2
    if not cells:
        return -8.0
    field = ctx.field
    score = 0.0
    hf = pose_hole_features(
        world,
        rot,
        col,
        pre_holes=ctx.pre_holes,
        pre_wm=ctx.pre_wm,
        field=field,
        cells=cells,
        sim=sim,
    )
    if float(hf.get("ok", 0.0)) >= 0.5:
        d_h = float(hf.get("d_holes", 0.0))
        d_w = float(hf.get("d_well", 0.0))
        d_mw = float(hf.get("d_max_well", 0.0))
        score -= 3.0 * max(0.0, d_h)
        score += 1.5 * max(0.0, -d_h)
        score -= 2.5 * max(0.0, d_w)
        score += 1.8 * max(0.0, -d_w)
        score -= 2.0 * max(0.0, d_mw)
        score += 1.2 * max(0.0, -d_mw)
        score -= 0.8 * float(hf.get("post_max_well", 0.0))
        score += 0.8 * float(hf.get("holes_filled", 0.0))
    else:
        score -= 4.0

    valence = ctx.valence
    t2k = ctx.t2k
    last_by = ctx.last_by
    lab_index = ctx.lab_index
    rc_to_sid = ctx.rc_to_sid

    for r, c in cells:
        if not (0 <= r < world.rows and 0 <= c < world.cols):
            score -= 3.0
            continue
        reading = float(field[r][c])
        if reading >= 0.99:
            score -= 2.5
            continue
        if abs(reading - 0.5) < 0.05:
            score += 1.6
        else:
            score += 0.12
        score += 0.035 * float(r)

        sid = rc_to_sid.get((r, c))
        th = last_by.get(sid) if sid else None
        if th is not None:
            score += 0.28 * float(getattr(th, "activation", 0.0) or 0.0)
            ck = t2k.get(th.id)
            if ck is not None:
                score += 0.18 * float(valence.get(ck, 0.0))
        lab = f"cell_r{r:02d}_c{c:02d}"
        for ck, oth in lab_index.get(lab, ()):
            score += 0.12 * float(getattr(oth, "activation", 0.0) or 0.0)
            score += 0.12 * float(valence.get(ck, 0.0))
            break
        synth = f"{lab}:place"
        score += 0.22 * float(valence.get(synth, 0.0))
    return score


def cell_thought_placement_score(
    s: Symbioid,
    world: TetrisWorld,
    rot: int,
    col: int,
) -> float:
    """
    Phase C: score a landing pose (single-pose API).

    Builds a one-shot context; for many poses prefer
    ``batch_cell_thought_placement_scores`` or ``CachedGraphPlacementBonus``.
    """
    ctx = build_placement_score_context(s, world)
    return score_pose_with_context(ctx, rot, col)


def batch_cell_thought_placement_scores(
    s: Symbioid,
    world: TetrisWorld,
    poses: list[tuple[int, int]],
) -> list[float]:
    """
    Score many (rot, col) landings with one Mind/field snapshot.

    Phase 3B: one hard-drop per pose (shared board template) — no second
    landing_cells + simulate_placement pair.
    """
    if not poses:
        return []
    ctx = build_placement_score_context(s, world)
    landings = world.batch_landing_cells_and_features(list(poses))
    out: list[float] = []
    for (rot, col), (cells, sim) in zip(poses, landings):
        out.append(
            score_pose_with_context(
                ctx, int(rot), int(col), cells=cells, sim=sim
            )
        )
    return out


class CachedGraphPlacementBonus:
    """
    Phase 3/3B graph bonus: prepare then O(1) lookups.

    ``prepare_landings`` reuses choose_target's hard-drop batch (no second sim).
    """

    def __init__(self, s: Symbioid) -> None:
        self.s = s
        self._scores: dict[tuple[int, int], float] = {}

    def prepare(self, world: TetrisWorld, options: list[tuple[int, int]]) -> None:
        scores = batch_cell_thought_placement_scores(self.s, world, list(options))
        self._scores = {
            (int(rot), int(col)): float(sc) for (rot, col), sc in zip(options, scores)
        }

    def prepare_landings(
        self,
        world: TetrisWorld,
        options: list[tuple[int, int]],
        landings: list,
    ) -> None:
        """Score from precomputed (cells, sim) pairs — one drop shared with coach."""
        ctx = build_placement_score_context(self.s, world)
        self._scores = {}
        for (rot, col), (cells, sim) in zip(options, landings):
            sc = score_pose_with_context(
                ctx, int(rot), int(col), cells=cells, sim=sim
            )
            self._scores[(int(rot), int(col))] = float(sc)

    def __call__(self, world: TetrisWorld, rot: int, col: int) -> float:
        key = (int(rot) % 4, int(col))
        if key in self._scores:
            return self._scores[key]
        k2 = (int(rot), int(col))
        if k2 in self._scores:
            return self._scores[k2]
        return float(cell_thought_placement_score(self.s, world, rot, col))


def graph_preferred_intent(
    s: Symbioid,
    world: TetrisWorld,
    coach: TetrisCoach,
) -> tuple[str | None, float, list, str | None]:
    """
    Symbioid → coach control hand-off.

    **Network-primary (default):**
    - **Strategy / placement:** ``choose_target`` with cell-map Thought heat
      yields a geometric micro-intent (rotate/left/right/hard toward target).
    - **Policy:** Mind ``recommend_action`` soft-biases; only overrides geo when
      score is strong or when no target exists yet.
    - **Bias:** high ``graph_bias`` so ``tick`` usually emits the network intent.

    Coach remains: cipher map, residual board value, gravity separation,
    stuck/force-hard survival, cold explore fallback.

    Returns (preferred_intent, graph_bias, poles, hint).
    """
    poles = policy_state_poles(s, world)
    net = bool(getattr(coach, "network_primary", True))
    # Heat Action poles so activation weight can contribute to recommend
    for tok in VALID_ACTIONS:
        th = s.mind.ensure_action_thought("tetris", tok, host_id=s.id)
        s.add_thought(th)

    rec = s.mind.recommend_action(poles, domain="tetris", min_score=0.05)
    # Phase 3: optional TFT retaliate gate (default off; set block_tokens to enable)
    if (
        rec is not None
        and hasattr(s.mind, "should_block_token")
        and s.mind.should_block_token(rec.token)
    ):
        rec = None
    mind_tok: str | None = None
    mind_score = 0.0
    if rec is not None and rec.token in VALID_ACTIONS:
        mind_tok = rec.token
        mind_score = float(getattr(rec, "score", 0.0) or 0.0)

    play = coach.play_ready() or coach.map_complete()

    # --- Geometric intent from network-scored placement target ---
    geo_intent: str | None = None
    if world.active is not None and (play or (net and coach.graph_placement_bonus is not None)):
        if coach._target is None:
            try:
                # Only structured aim once we can hard-drop (or full map)
                if play:
                    coach._target = coach.choose_target(world)
            except Exception:
                pass
        if coach._target is not None:
            tgt_rot, tgt_col = coach._target
            cur = world.active
            if cur.rotation % 4 != tgt_rot % 4:
                geo_intent = "rotate"
            elif cur.col < tgt_col:
                geo_intent = "right"
            elif cur.col > tgt_col:
                geo_intent = "left"
            else:
                geo_intent = "hard"

    # Network-primary strategy: geo from placement owns the path;
    # Mind overrides only when strong or when geo is missing.
    preferred: str | None = None
    hint: str | None = None
    if net:
        if geo_intent is not None:
            preferred = geo_intent
            hint = f"geo:{geo_intent}"
            # Strong Mind signal may override (learned policy over pure geometry)
            if mind_tok is not None and mind_tok != geo_intent and mind_score >= 0.55:
                preferred = mind_tok
                hint = f"{mind_tok}@{mind_score:.2f}>geo"
            elif mind_tok is not None and mind_tok == geo_intent:
                hint = f"geo+mind:{geo_intent}"
        elif mind_tok is not None:
            preferred = mind_tok
            hint = f"{mind_tok}@{mind_score:.2f}"
    else:
        # Legacy coach-primary: Mind first, geo as soft fill
        if mind_tok is not None:
            preferred = mind_tok
            hint = f"{mind_tok}@{mind_score:.2f}"
        elif geo_intent is not None:
            preferred = geo_intent
            hint = f"geo:{geo_intent}"

    want_hard = coach.wants_hard_now(world)
    if want_hard:
        hard_th = s.mind.ensure_action_thought("tetris", "hard", host_id=s.id)
        s.add_thought(hard_th)
        s.stimulate(hard_th, 1.8)
        if hasattr(s.outerface, "add_member"):
            s.outerface.add_member(hard_th.id)
        preferred = "hard"
        hint = "hard@stuck" if coach._stuck_lateral >= 2 else "hard@align"
    elif preferred is not None:
        th = s.mind.ensure_action_thought("tetris", preferred, host_id=s.id)
        s.add_thought(th)
        s.stimulate(th, 1.35 if net else 0.9)
        if hasattr(s.outerface, "add_member"):
            s.outerface.add_member(th.id)

    # Bias: network-primary runs the command path; coach-primary is soft
    if not play:
        # While mapping: still let network try known intents (discover via outcomes)
        if preferred and coach.byte_for_effect(preferred) is not None:
            graph_bias = 0.50 if net else 0.0
        else:
            graph_bias = 0.0
        return preferred, graph_bias, poles, hint

    if net:
        if preferred == "hard":
            graph_bias = 0.98 if coach._stuck_lateral >= 2 else 0.95
        elif preferred is not None:
            graph_bias = 0.93
        else:
            graph_bias = 0.90  # coach fallback rare
    else:
        # Legacy coach-primary soft bias
        if preferred == "hard":
            graph_bias = 0.72
        elif preferred is not None:
            graph_bias = 0.55
        else:
            graph_bias = 0.50

    return preferred, graph_bias, poles, hint


def _update_pack_readings(s: Symbioid, world: TetrisWorld, coach: object | None = None) -> None:
    """Refresh host-held packing readings for meta sensor.transfer callables."""
    try:
        from symbioid.world.tetris_learn import observe_board

        obs = observe_board(world)
        s._pack_holes = float(obs.get("holes", world.hole_count()))  # type: ignore[attr-defined]
        s._pack_well = float(obs.get("well", 0.0))  # type: ignore[attr-defined]
        s._pack_max_well = float(obs.get("max_well", 0.0))  # type: ignore[attr-defined]
    except Exception:
        s._pack_holes = float(world.hole_count())  # type: ignore[attr-defined]
    if coach is not None:
        s._last_d_holes = float(getattr(coach, "last_d_holes", 0.0) or 0.0)  # type: ignore[attr-defined]


def update_pred_pack_for_target(
    s: Symbioid, world: TetrisWorld, coach: object
) -> dict[str, float]:
    """
    Foresight: how many holes the **current placement target** would free/create.

    Uses ``pose_hole_features`` on ``coach._target`` (same sim as placement score).
    Writes host fields read by ``pred_d_holes`` / ``holes_freed`` / ``holes_fill_n``.
    """
    empty = {
        "pred_d_holes": 0.0,
        "holes_freed": 0.0,
        "holes_fill_n": 0.0,
        "ok": 0.0,
    }
    tgt = getattr(coach, "_target", None)
    if world.active is None or world.game_over or tgt is None:
        s._pred_d_holes = 0.0  # type: ignore[attr-defined]
        s._holes_freed = 0.0  # type: ignore[attr-defined]
        s._holes_fill_n = 0.0  # type: ignore[attr-defined]
        return empty
    try:
        rot, col = int(tgt[0]), int(tgt[1])
    except (TypeError, ValueError, IndexError):
        s._pred_d_holes = 0.0  # type: ignore[attr-defined]
        s._holes_freed = 0.0  # type: ignore[attr-defined]
        s._holes_fill_n = 0.0  # type: ignore[attr-defined]
        return empty
    hf = pose_hole_features(world, rot, col)
    if float(hf.get("ok", 0.0)) < 0.5:
        s._pred_d_holes = 0.0  # type: ignore[attr-defined]
        s._holes_freed = 0.0  # type: ignore[attr-defined]
        s._holes_fill_n = 0.0  # type: ignore[attr-defined]
        return empty
    d_h = float(hf.get("d_holes", 0.0))
    filled = float(hf.get("holes_filled", 0.0))
    freed = max(0.0, -d_h)
    s._pred_d_holes = d_h  # type: ignore[attr-defined]
    s._holes_freed = freed  # type: ignore[attr-defined]
    s._holes_fill_n = filled  # type: ignore[attr-defined]
    return {
        "pred_d_holes": d_h,
        "holes_freed": freed,
        "holes_fill_n": filled,
        "ok": 1.0,
    }


def sample_packing_meta_into_symbioid(
    s: Symbioid,
    world: TetrisWorld,
    tick: int,
    *,
    coach: object | None = None,
    labels: frozenset[str] | None = None,
) -> None:
    """Force-sample packing meta sensors so hole insights become Observations."""
    _update_pack_readings(s, world, coach)
    if coach is not None:
        update_pred_pack_for_target(s, world, coach)
    want = labels if labels is not None else _PACKING_META_LABELS
    w = world.sensor_world()
    handoffs = []
    for sen in s.sensors:
        lab = sen.label or ""
        if lab not in want:
            continue
        sense = sen.sample(tick=tick, world=w)
        if sense is None:
            continue
        h = s.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)
    if not handoffs:
        return
    if len(handoffs) > 1:
        s.innerface.post(
            {"kind": "formation_batch", "handoffs": handoffs, "tick": tick}
        )
    else:
        s.innerface.post(handoffs[0])


def sample_into_symbioid(s: Symbioid, world: TetrisWorld, tick: int) -> None:
    """
    Sample meta sensors every call; cell map is **change-only** + **ROI**:

    - Build the field once per sample (not per cell).
    - Skip formation when reading unchanged.
    - Skip initial open (0.0) cells until they first become block/hole.
    - **S0.2** empty top rows above :meth:`TetrisWorld.sky_row`.
    - **S0.1** solid full-width floor at bottom (:meth:`TetrisWorld.solid_floor_start_row`).
    - **S1.1** dirty-rect = previous ∪ current active cells (always considered).
    - **S1.4** sticky locked=1.0 (skip re-form until line clear / resync).
    - **S1.9** invalidate cell last-readings on line clear (or rising ``lines``).
    - Iterate only candidate cells (not all 200 sensors).
    """
    _update_pack_readings(s, world, coach=None)
    w = world.sensor_world()
    w["byte"] = float(s.actuators[0].output)
    handoffs = []
    last: dict[str, float] = getattr(s, "_cell_last_reading", None) or {}
    cell_by_rc: dict[tuple[int, int], object] = getattr(s, "_cell_by_rc", None) or {}
    prev_active: set[tuple[int, int]] = set(getattr(s, "_cell_prev_active", None) or set())
    last_lines = int(getattr(s, "_cell_last_lines", 0) or 0)

    # Line-clear invalidation: forget cell last-readings so ROI resyncs
    force_resync = False
    if world.last_event == "line_clear" or int(world.lines) > last_lines:
        force_resync = True
        last = {}
    s._cell_last_lines = int(world.lines)  # type: ignore[attr-defined]

    cur_active = world.active_cells_set()
    dirty = prev_active | cur_active
    r_lo, r_hi = world.cell_sample_roi(with_active=True)

    # Candidate (r,c): ROI band + dirty (active may sit in former sky)
    candidates: set[tuple[int, int]] = set(dirty)
    if cell_by_rc and r_lo < r_hi:
        for r in range(r_lo, r_hi):
            for c in range(world.cols):
                candidates.add((r, c))
    elif cell_by_rc and force_resync:
        # Empty ROI but resync requested — touch all cells once
        for r in range(world.rows):
            for c in range(world.cols):
                candidates.add((r, c))

    field = (
        world.cell_field_state(with_active=True)
        if candidates and cell_by_rc
        else None
    )

    for r, c in candidates:
        sen = cell_by_rc.get((r, c))
        if sen is None or field is None:
            continue
        # Outside ROI and not dirty → skip (solid floor / pure sky)
        in_roi = r_lo <= r < r_hi
        is_dirty = (r, c) in dirty
        if not in_roi and not is_dirty and not force_resync:
            continue

        reading = float(field[r][c])
        sid = sen.id
        prev = last.get(sid)

        # Sticky locked block: no re-form until resync / dirty paint edge
        if (
            not force_resync
            and not is_dirty
            and prev is not None
            and abs(prev - 1.0) < 1e-9
            and abs(reading - 1.0) < 1e-9
            and bool(world.board[r][c])
        ):
            continue

        # First sight of open sky: remember only, no Rodin storm
        if prev is None and reading == 0.0:
            last[sid] = reading
            continue
        if prev is not None and abs(prev - reading) < 1e-9:
            continue
        last[sid] = reading
        sense = {
            "sensor_id": sid,
            "label": sen.label,
            "reading": reading,
            "tick": tick,
            "kind": "input",
        }
        h = s.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)

    # Meta sensors (piece, next, lines, byte) — not cell map
    for sen in s.sensors:
        if getattr(s, "_cell_rc", None) and sen.id in s._cell_rc:  # type: ignore[attr-defined]
            continue
        sense = sen.sample(tick=tick, world=w)
        if sense is None:
            continue
        h = s.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)

    s._cell_last_reading = last  # type: ignore[attr-defined]
    s._cell_prev_active = cur_active  # type: ignore[attr-defined]

    if not handoffs:
        return
    if len(handoffs) > 1:
        s.innerface.post({"kind": "formation_batch", "handoffs": handoffs, "tick": tick})
    else:
        s.innerface.post(handoffs[0])


def draw_thought_plot(
    screen: pygame.Surface,
    history: list[int],
    font_sm: pygame.font.Font,
    *,
    ox: int,
    oy: int,
    width: int,
    height: int,
    title: str,
    line_color: tuple[int, int, int],
    marker_color: tuple[int, int, int],
    show_x_labels: bool = True,
) -> None:
    """One line plot of a Thought-count series over the last PLOT_HISTORY turns."""
    pygame.draw.rect(
        screen, (20, 24, 40), (ox - 2, oy - 2, width + 4, height + 4), border_radius=4
    )
    pygame.draw.rect(screen, (8, 10, 18), (ox, oy, width, height))

    pad_l, pad_r, pad_t, pad_b = 36, 8, 14, 20 if show_x_labels else 8
    plot_x = ox + pad_l
    plot_y = oy + pad_t
    plot_w = max(1, width - pad_l - pad_r)
    plot_h = max(1, height - pad_t - pad_b)

    pygame.draw.rect(screen, (40, 48, 70), (plot_x, plot_y, plot_w, plot_h), width=1)

    # Title + live value
    cur = history[-1] if history else 0
    screen.blit(
        font_sm.render(f"{title}: {cur} ", True, line_color),
        (ox + 80, oy + 1),
    )

    if len(history) < 2:
        screen.blit(
            font_sm.render("waiting for turns…", True, (100, 110, 130)),
            (plot_x + 8, plot_y + max(0, plot_h // 2 - 6)),
        )
        return

    data = history[-PLOT_HISTORY:]
    n = len(data)
    y_min = min(data)
    y_max = max(data)
    if y_max <= y_min:
        y_max = y_min + 1
    span = y_max - y_min
    y_min = max(0, y_min - max(1, span // 10))
    y_max = y_max + max(1, span // 10)

    def sx(i: int) -> int:
        return plot_x + int(i * (plot_w - 1) / max(1, PLOT_HISTORY - 1))

    def sy(v: int) -> int:
        t = (v - y_min) / (y_max - y_min)
        return plot_y + plot_h - 1 - int(t * (plot_h - 1))

    for tick in (0, 256, 512, 768, 1024):
        tx = plot_x + int(tick * (plot_w - 1) / max(1, PLOT_HISTORY - 1))
        pygame.draw.line(
            screen, (30, 36, 55), (tx, plot_y), (tx, plot_y + plot_h - 1), 1
        )
        if show_x_labels:
            label = font_sm.render(str(tick), True, (90, 100, 120))
            screen.blit(label, (tx - label.get_width() // 2, plot_y + plot_h + 2))

    for v, label_s in ((y_min, str(y_min)), (y_max, str(y_max))):
        ly = sy(v)
        lab = font_sm.render(label_s, True, (90, 100, 120))
        screen.blit(lab, (ox + 2, ly - lab.get_height() // 2))

    offset = PLOT_HISTORY - n
    points = [(sx(offset + i), sy(v)) for i, v in enumerate(data)]
    if len(points) >= 2:
        pygame.draw.lines(screen, line_color, False, points, 2)
    pygame.draw.circle(screen, marker_color, points[-1], 3)


def draw(
    screen: pygame.Surface,
    world: TetrisWorld,
    coach: TetrisCoach,
    s: Symbioid,
    active_history: list[int],
    inactive_history: list[int],
    mint_history: list[int],
    font: pygame.font.Font,
    font_sm: pygame.font.Font,
    *,
    pause_seconds_left: float | None = None,
    graph_hint: str | None = None,
) -> None:
    screen.fill((12, 14, 28))
    ox, oy = MARGIN_X, MARGIN_Y

    pygame.draw.rect(
        screen, (20, 24, 40), (ox - 2, oy - 2, BOARD_W + 4, BOARD_H + 4), border_radius=4
    )
    pygame.draw.rect(screen, (8, 10, 18), (ox, oy, BOARD_W, BOARD_H))

    for r in range(ROWS + 1):
        pygame.draw.line(
            screen, (25, 30, 48), (ox, oy + r * CELL), (ox + BOARD_W, oy + r * CELL), 1
        )
    for c in range(COLS + 1):
        pygame.draw.line(
            screen, (25, 30, 48), (ox + c * CELL, oy), (ox + c * CELL, oy + BOARD_H), 1
        )

    for r in range(world.rows):
        for c in range(world.cols):
            kind = world.board[r][c]
            if not kind:
                continue
            color = PIECE_COLORS.get(kind, (180, 180, 180))
            pygame.draw.rect(
                screen,
                color,
                (ox + c * CELL + 1, oy + r * CELL + 1, CELL - 2, CELL - 2),
                border_radius=3,
            )

    if world.active is not None:
        ghost_r = world.ghost_row()
        if ghost_r is not None:
            gr_off = ghost_r - world.active.row
            for r, c in world.active.cells():
                rr = r + gr_off
                if 0 <= rr < world.rows and 0 <= c < world.cols:
                    pygame.draw.rect(
                        screen,
                        (50, 55, 70),
                        (ox + c * CELL + 1, oy + rr * CELL + 1, CELL - 2, CELL - 2),
                        width=1,
                        border_radius=3,
                    )
        color = PIECE_COLORS.get(world.active.kind, (220, 220, 220))
        for r, c in world.active.cells():
            if 0 <= r < world.rows and 0 <= c < world.cols:
                pygame.draw.rect(
                    screen,
                    color,
                    (ox + c * CELL + 1, oy + r * CELL + 1, CELL - 2, CELL - 2),
                    border_radius=3,
                )

    # Right panel: game / score / lines / pieces / next / highscores only
    sx = ox + BOARD_W + 24
    y = oy
    screen.blit(font.render("Symbioid Tetris", True, (200, 210, 230)), (sx, y))
    y += 32
    for line, color in (
        (f"game  #{coach.game_number}", (180, 190, 210)),
        (f"score  {world.score}", (180, 190, 210)),
        (f"lines  {world.lines}", (180, 190, 210)),
        (f"pieces {world.pieces_placed}", (180, 190, 210)),
    ):
        screen.blit(font_sm.render(line, True, color), (sx, y))
        y += 20

    # Board packing sensors (holes = sealed; well = open trenches, edge-aware)
    try:
        from symbioid.world.tetris_learn import observe_board as _obs_board

        _bo = _obs_board(world)
        n_holes = int(_bo.get("holes", world.hole_count()))
        n_well = float(_bo.get("well", 0.0))
        n_mw = float(_bo.get("max_well", 0.0))
    except Exception:
        n_holes = int(world.hole_count())
        n_well = float(getattr(world, "well_depth", lambda: 0.0)())
        n_mw = float(getattr(world, "max_well_depth", lambda: 0.0)())
    y += 6
    screen.blit(
        font_sm.render(
            f"pack  holes={n_holes}  well={n_well:.0f}  maxW={n_mw:.0f}",
            True,
            (220, 160, 140) if (n_holes > 0 or n_mw >= 2) else (140, 170, 150),
        ),
        (sx, y),
    )
    y += 16
    d_h = float(getattr(s, "_last_d_holes", 0.0) or 0.0)
    screen.blit(
        font_sm.render(
            f"Δh    last={d_h:+.0f}  (past lock)",
            True,
            (220, 150, 130) if d_h > 0 else (130, 190, 150) if d_h < 0 else (120, 140, 160),
        ),
        (sx, y),
    )
    y += 16
    pred_d = float(getattr(s, "_pred_d_holes", 0.0) or 0.0)
    freed = float(getattr(s, "_holes_freed", 0.0) or 0.0)
    fill_n = float(getattr(s, "_holes_fill_n", 0.0) or 0.0)
    screen.blit(
        font_sm.render(
            f"free  predΔ={pred_d:+.0f}  frees={freed:.0f}  fill={fill_n:.0f}",
            True,
            (130, 200, 150) if freed > 0 else (200, 160, 130) if pred_d > 0 else (120, 140, 160),
        ),
        (sx, y),
    )
    y += 16
    # TFT / iterated twin (v0.0.53+)
    try:
        snap = s.mind.tft_snapshot()
        cnt = snap.get("counts") or {}
        tft_line = (
            f"tft   {snap.get('tft_state', 'open')}  "
            f"C={cnt.get('C', 0)} D={cnt.get('D', 0)} U={cnt.get('U', 0)}  "
            f"stk={snap.get('c_streak', 0)} fg={snap.get('forgives', 0)}"
        )
        st = str(snap.get("tft_state", "open"))
        tft_color = (
            (200, 140, 120)
            if st == "retaliate"
            else (140, 190, 160)
            if st == "open"
            else (160, 170, 200)
        )
    except Exception:
        tft_line = "tft   (n/a)"
        tft_color = (120, 130, 150)
    screen.blit(font_sm.render(tft_line, True, tft_color), (sx, y))
    y += 16
    # Clocks: sense / command / pulse periods (frames @ FPS)
    screen.blit(
        font_sm.render(
            f"clk   se/{SAMPLE_EVERY} cmd/{CMD_EVERY} p/{PULSE_EVERY}"
            f"+{PULSES_PRE_CMD}pre +{PULSES_ON_LOCK}lk pl/{PLACE_EVERY}",
            True,
            (110, 130, 150),
        ),
        (sx, y),
    )
    y += 16

    # Active six-set breakdown (Band A caps; sense/sync/integrate)
    n_act, n_inact = thought_counts_active_inactive(s)
    summary = s.innerface.active_set_summary()
    n_sense = int(summary.get("sense", 0))
    n_sync = int(summary.get("sync", 0))
    n_int = int(summary.get("integrate", 0))
    cap_s = int(s.innerface.max_active_senses)
    cap_y = int(s.innerface.max_active_syncs)
    cap_i = int(s.innerface.max_active_integrates)
    y += 8
    screen.blit(
        font_sm.render(f"Thoughts  A={n_act}  I={n_inact}", True, (140, 200, 160)),
        (sx, y),
    )
    y += 16
    screen.blit(
        font_sm.render(
            f"sets  se {n_sense}/{cap_s}  sy {n_sync}/{cap_y}  in {n_int}/{cap_i}",
            True,
            (130, 170, 200),
        ),
        (sx, y),
    )
    y += 16
    n_fl = len(s.mind._follows)
    n_ig = len(s.mind._integrates)
    cap_fl = int(s.mind.max_follows_registry)
    cap_ig = int(s.mind.max_integrates_registry)
    screen.blit(
        font_sm.render(
            f"reg   fl {n_fl}/{cap_fl}  ig {n_ig}/{cap_ig}",
            True,
            (120, 150, 180),
        ),
        (sx, y),
    )
    y += 16
    if graph_hint:
        screen.blit(
            font_sm.render(str(graph_hint)[:36], True, (200, 180, 120)),
            (sx, y),
        )
        y += 16

    y += 12
    screen.blit(font_sm.render("next", True, (140, 150, 170)), (sx, y))
    y += 20
    nc = PIECE_COLORS.get(world.next_kind, (180, 180, 180))
    for dr, dc in piece_cells(world.next_kind, 0):
        pygame.draw.rect(
            screen,
            nc,
            (sx + dc * 18, y + dr * 18, 16, 16),
            border_radius=2,
        )
    y += 80

    screen.blit(
        font_sm.render("highscores  best first", True, (200, 190, 120)), (sx, y)
    )
    y += 18
    if not coach.highscores:
        screen.blit(font_sm.render("(finish a game…)", True, (100, 110, 130)), (sx, y))
    else:
        for i, line in enumerate(coach.highscore_lines(limit=10)):
            color = (240, 210, 100) if i == 0 else (160, 170, 190)
            screen.blit(font_sm.render(line, True, color), (sx, y))
            y += 15

    if world.game_over and pause_seconds_left is not None and pause_seconds_left > 0:
        screen.blit(
            font_sm.render(
                f"top out — {pause_seconds_left:.0f}s",
                True,
                (240, 140, 120),
            ),
            (sx, oy + BOARD_H - 24),
        )

    # Plots under the board: Active / Inactive / Minted Thoughts over turns
    plot_oy = oy + BOARD_H + PLOT_MARGIN
    draw_thought_plot(
        screen,
        active_history,
        font_sm,
        ox=ox,
        oy=plot_oy,
        width=BOARD_W,
        height=PLOT_H,
        title="Active Thoughts",
        line_color=(100, 220, 140),
        marker_color=(180, 255, 160),
        show_x_labels=False,
    )
    draw_thought_plot(
        screen,
        inactive_history,
        font_sm,
        ox=ox,
        oy=plot_oy + PLOT_H + PLOT_GAP,
        width=BOARD_W,
        height=PLOT_H,
        title="Inactive Thoughts",
        line_color=(120, 170, 255),
        marker_color=(180, 210, 255),
        show_x_labels=False,
    )
    draw_thought_plot(
        screen,
        mint_history,
        font_sm,
        ox=ox,
        oy=plot_oy + 2 * (PLOT_H + PLOT_GAP),
        width=BOARD_W,
        height=PLOT_H,
        title="Minted Thoughts",
        line_color=(240, 180, 80),
        marker_color=(255, 210, 120),
        show_x_labels=True,
    )

    screen.blit(
        font_sm.render(
            "Secret bytes → play. Esc quit.  --verbose for console dumps.",
            True,
            (120, 130, 160),
        ),
        (MARGIN_X, H - FOOTER_H + 4),
    )


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        trial = (" ".join(cur + [w])).strip()
        if len(trial) > width and cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return lines or [text]


def main(argv: list[str] | None = None) -> None:
    global C_STREAK_BONUS
    args = parse_args(argv)
    set_console_emit(args.verbose)
    log = print if args.verbose else (lambda *a, **k: None)
    C_STREAK_BONUS = float(getattr(args, "c_streak_bonus", 0.0) or 0.0)

    want_primary = bool(args.spectral_primary) and not bool(args.no_spectral)
    want_spectral = (bool(args.spectral) or want_primary) and not bool(
        args.no_spectral
    )

    # --- Headless multi-game placement metric (no GUI) ---
    if bool(getattr(args, "headless", False)):
        use_elig = not bool(getattr(args, "no_eligibility", False))
        win = int(getattr(args, "eligibility_window", 24) or 0)
        rows, summary = run_multi_game_metric(
            games=int(args.games),
            max_frames=int(args.max_frames),
            seed=int(args.seed),
            eligibility_window=win,
            use_eligibility=use_elig,
            spectral=want_spectral,
            spectral_primary=want_primary,
            verbose=True,  # always print metric lines in headless
        )
        print(
            f"\n=== multi-game metric (n={int(summary['n'])}) "
            f"eligibility={'on' if use_elig and win > 0 else 'off'} "
            f"window={win} seed={args.seed} ===",
            flush=True,
        )
        for r in rows:
            print(
                f"  g{r.game}: score={r.score} lines={r.lines} "
                f"holes={r.holes} maxH={r.max_height} "
                f"aggH={r.aggregate_height} pieces={r.pieces} "
                f"frames={r.frames} top_out={r.top_out} "
                f"C={r.n_C} D={r.n_D} tft={r.tft_state}",
                flush=True,
            )
        print(
            f"  means: score={summary['mean_score']:.1f} "
            f"lines={summary['mean_lines']:.2f} "
            f"holes={summary['mean_holes']:.2f} "
            f"maxH={summary['mean_max_height']:.2f} "
            f"pieces={summary['mean_pieces']:.1f} "
            f"frames={summary.get('mean_frames', 0):.0f} "
            f"top_out_rate={summary.get('top_out_rate', 0):.2f} "
            f"c_rate={summary.get('c_rate', 0):.3f}",
            flush=True,
        )
        return

    # Fresh secret cipher each run (4 live bytes among 256)
    rng_world = __import__("random").Random()
    cipher = ActionCipher.random(rng_world)
    world = TetrisWorld(
        cols=COLS,
        rows=ROWS,
        gravity_interval=GRAVITY_INTERVAL,
        cipher=cipher,
        rng=rng_world,
    )
    coach = TetrisCoach(network_primary=True)
    s = build_symbioid(
        world, spectral=want_spectral, spectral_primary=want_primary
    )
    if bool(getattr(args, "no_warm_start", False)):
        s.mind.warm_start_actions = False
    # Co-lead blend: network heat + coach residual (research 2026-07-26).
    # Floor clamp in choose_target is 0.35 so 0.60 is honored.
    # Phase 3: batch Mind/field locks once per choose_target via prepare().
    coach.graph_placement_weight = 0.60
    coach.graph_placement_bonus = CachedGraphPlacementBonus(s)

    mem_path = Path(args.memory)
    use_memory = not args.no_memory
    if use_memory and args.reset_memory and mem_path.is_file():
        mem_path.unlink()
        log(f"[memory] reset {mem_path}", flush=True)
    if use_memory and mem_path.is_file():
        if try_load_into(s, mem_path):
            log(
                f"[memory] loaded {mem_path} "
                f"Thoughts={thought_count(s)} actions={len(s.mind._actions)}",
                flush=True,
            )
        else:
            log(f"[memory] failed to load {mem_path}; starting fresh", flush=True)

    twin = s.twin_seed_thoughts()
    if twin:
        log(format_six_set_line("twin", twin, index=0), flush=True)
    log(
        "Tetris + Symbioid: SECRET byte map "
        f"({len(cipher)} live / 256). Agent must discover it.",
        flush=True,
    )
    log("(Cipher hidden from learner; not printed here.)", flush=True)
    if use_memory:
        log(f"(Agent memory: {mem_path})", flush=True)
    dyn = getattr(s.mind, "dynamics_mode", "hybrid")
    log(
        f"(Spectral: mode={dyn} mix={s.mind.spectral_mix_enabled} "
        f"holonomic={s.mind.holonomic_store_enabled} "
        f"primary={want_primary})",
        flush=True,
    )

    s.start_processes()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    caption = "Symbioid Tetris — secret byte controls"
    if want_primary:
        caption += " [spectral-primary]"
    elif want_spectral:
        caption += " [spectral]"
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 20)
    font_sm = pygame.font.SysFont("DejaVu Sans", 14)
    frame = 0
    sample_every = SAMPLE_EVERY
    log_every = 90
    game_over_at: int | None = None
    was_mapped = False
    # Thought counts once per game turn (piece lock)
    active_history: list[int] = []
    inactive_history: list[int] = []
    mint_history: list[int] = []
    last_pieces_for_plot = world.pieces_placed
    last_graph_hint: str | None = None
    # State poles at last command (for outcome write on lock)
    last_cmd_poles: list = []
    last_cmd_intent: str = "explore"
    use_elig = not bool(getattr(args, "no_eligibility", False))
    elig_n = int(getattr(args, "eligibility_window", 24) or 0)
    if not use_elig:
        elig_n = 0
    eligibility = EligibilityWindow(max_ticks=elig_n)
    log(
        f"(Lock credit: landing-cell + eligibility window={elig_n})",
        flush=True,
    )

    def _record_thought_sample() -> None:
        a, i = thought_counts_active_inactive(s)
        active_history.append(a)
        inactive_history.append(i)
        mint_history.append(int(s.mind.admits_mint))
        if len(active_history) > PLOT_HISTORY:
            del active_history[: len(active_history) - PLOT_HISTORY]
        if len(inactive_history) > PLOT_HISTORY:
            del inactive_history[: len(inactive_history) - PLOT_HISTORY]
        if len(mint_history) > PLOT_HISTORY:
            del mint_history[: len(mint_history) - PLOT_HISTORY]

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r and world.game_over:
                        gc_stats = game_boundary_gc(s)
                        entry = coach.on_new_game(world, record=True)
                        game_over_at = None
                        last_pieces_for_plot = world.pieces_placed
                        log(
                            f"[restart] {entry} | map {coach.map_progress()} "
                            f"| gc th→{gc_stats.get('thoughts_after')} "
                            f"rm={gc_stats.get('removed')}",
                            flush=True,
                        )

            # --- Network / game clocks (sense ≥ cmd; pulse settles before decide) ---
            is_cmd = (not world.game_over) and (frame % CMD_EVERY == 0)
            is_sample = frame % sample_every == 0

            # 1) Sense: refresh cell map + meta before any command on this frame
            if is_sample:
                sample_into_symbioid(s, world, tick=frame)

            # 2) Baseline dynamics
            if s.mind.dynamics_enabled and frame % PULSE_EVERY == 0:
                s.pulse_tick()

            # 3) Command: extra settle pulses, then intent + byte
            if is_cmd:
                if s.mind.dynamics_enabled and PULSES_PRE_CMD > 0:
                    for _ in range(int(PULSES_PRE_CMD)):
                        s.pulse_tick()
                prev_event = world.last_event
                prev_pieces = world.pieces_placed
                # Symbioid-primary control: throttled network placement
                preferred, g_bias, poles, g_hint = cached_graph_intent(
                    s, world, coach, frame, place_every=PLACE_EVERY
                )
                # Foresight: how many holes the current target placement frees
                update_pred_pack_for_target(s, world, coach)
                sample_packing_meta_into_symbioid(
                    s,
                    world,
                    tick=frame,
                    coach=coach,
                    labels=_FORESIGHT_META_LABELS,
                )
                last_graph_hint = g_hint
                last_cmd_poles = poles
                code = coach.tick(
                    world,
                    preferred_intent=preferred,
                    graph_bias=g_bias,
                )
                last_cmd_intent = coach.last_intent
                # Trajectory eligibility: poles that steered this cmd tick
                if eligibility.max_ticks > 0:
                    eligibility.push(poles_to_content_keys(s, poles))
                if getattr(coach, "last_network_cmd", False) and preferred:
                    last_graph_hint = f"NET {preferred}@{g_bias:.2f}"
                elif preferred and coach.last_intent == preferred:
                    last_graph_hint = f"NET {preferred}@{g_bias:.2f}"
                elif coach.last_intent:
                    last_graph_hint = f"fb {coach.last_intent}" + (
                        f" (want {preferred})" if preferred else ""
                    )
                s.actuators[0].output = code / 255.0
                # One sample per game turn (piece lock)
                if world.pieces_placed > prev_pieces:
                    coach.last_lock_frame = frame  # type: ignore[attr-defined]
                    # Phase A: label outcomes with true lock effect when known
                    # (e.g. hard), not coach last_intent=="explore"
                    lock_eff = getattr(coach, "last_lock_effect", "") or ""
                    intent = (
                        lock_eff
                        if lock_eff in VALID_ACTIONS
                        else last_cmd_intent
                        if last_cmd_intent in VALID_ACTIONS
                        else None
                    )
                    if intent is not None:
                        s.mind.record_outcome(
                            last_cmd_poles,
                            intent,
                            domain="tetris",
                            host_id=s.id,
                            reward=float(coach.last_reward),
                            host=s,
                        )
                    # Feeling bridge: coach board reward → Mind valence on recent obs
                    s.mind.note_valence(
                        channel="board",
                        delta=max(-2.0, min(2.0, float(coach.last_reward) / 50.0)),
                    )
                    # P1: landing-cell (0.0.33) + eligibility-window credit
                    apply_lock_credit(
                        s, coach, eligibility, poles=last_cmd_poles
                    )
                    # Network insight: mint Observations of holes_n / last_d_holes
                    s._last_d_holes = float(  # type: ignore[attr-defined]
                        getattr(coach, "last_d_holes", 0.0) or 0.0
                    )
                    sample_packing_meta_into_symbioid(
                        s, world, tick=frame, coach=coach
                    )
                    # Burst pulse so outcome valence can spread before next cmd
                    if s.mind.dynamics_enabled and PULSES_ON_LOCK > 0:
                        for _ in range(int(PULSES_ON_LOCK)):
                            s.pulse_tick()
                    _record_thought_sample()
                    last_pieces_for_plot = world.pieces_placed
                if coach.map_complete() and not was_mapped:
                    was_mapped = True
                    log(
                        f"[map complete] {coach.map_progress()} "
                        f"after {len(coach.bytes_tried)} bytes tried",
                        flush=True,
                    )
                if world.last_event != prev_event and world.last_event in (
                    "line_clear",
                    "top_out",
                    "lock",
                ):
                    log(
                        f"[{world.last_event}] g#{coach.game_number} "
                        f"score={world.score} byte=0x{code:02X} "
                        f"seen={coach.last_effect} | {coach.map_progress()}",
                        flush=True,
                    )

            pause_left = None
            if world.game_over:
                if game_over_at is None:
                    game_over_at = frame
                    # Final sample at game end
                    _record_thought_sample()
                    log(
                        f"[top_out] game #{coach.game_number} "
                        f"final_score={world.score} "
                        f"tried={len(coach.bytes_tried)}/256 "
                        f"— pausing {RESTART_DELAY_FRAMES / FPS:.0f}s for Innerface",
                        flush=True,
                    )
                elapsed = frame - game_over_at
                pause_left = max(0.0, (RESTART_DELAY_FRAMES - elapsed) / FPS)
                # Sensors still sample above each frame%sample_every so Innerface
                # can drain the formation queue during the pause.
                if elapsed >= RESTART_DELAY_FRAMES:
                    gc_stats = game_boundary_gc(s)
                    entry = coach.on_new_game(world, record=True)
                    game_over_at = None
                    last_pieces_for_plot = world.pieces_placed
                    log(
                        f"[auto-restart] {entry} best={coach.best_score()} "
                        f"highscores={coach.highscores[-6:]} "
                        f"| gc th→{gc_stats.get('thoughts_after')} "
                        f"rm={gc_stats.get('removed')}",
                        flush=True,
                    )

            if frame > 0 and frame % log_every == 0 and not world.game_over:
                log(
                    f"t={frame} g#{coach.game_number} score={world.score} "
                    f"Thoughts={thought_count(s)} "
                    f"0x{coach.last_byte:02X}→{coach.last_effect} "
                    f"intent={coach.last_intent} tried={len(coach.bytes_tried)} "
                    f"| {coach.map_progress()}",
                    flush=True,
                )

            draw(
                screen,
                world,
                coach,
                s,
                active_history,
                inactive_history,
                mint_history,
                font,
                font_sm,
                pause_seconds_left=pause_left,
                graph_hint=last_graph_hint,
            )
            pygame.display.flip()
            clock.tick(FPS)
            frame += 1
    finally:
        s.stop_processes()
        if use_memory:
            try:
                save_memory(s, mem_path)
                log(
                    f"[memory] saved {mem_path} "
                    f"Thoughts={thought_count(s)} actions={len(s.mind._actions)}",
                    flush=True,
                )
            except OSError as exc:
                log(f"[memory] save failed: {exc}", flush=True)
        pygame.quit()
        if not world.game_over and world.pieces_placed > 0:
            coach.record_game_score(world)
        log(
            f"\nstopped: game=#{coach.game_number} score={world.score}\n"
            f"  {coach.summary()}\n"
            f"  highscores: {coach.highscores}\n"
            f"  Thoughts={thought_count(s)} "
            f"formations={len(s.innerface.completed_formations)} "
            f"beliefs={len(s.outerface.active_belief_ids)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
