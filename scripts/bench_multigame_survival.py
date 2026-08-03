#!/usr/bin/env python3
"""
Multi-game survival check (v0.0.57 goal).

Success (research goal-loop):
  after 6 games: thoughts ≤ ~15k, mean frame work ≤ ~50 ms
  (instrumented place+pulse wall per frame).

Usage:
  PYTHONPATH=. .venv/bin/python scripts/bench_multigame_survival.py
  PYTHONPATH=. .venv/bin/python scripts/bench_multigame_survival.py --games 6 --max-frames 400
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_demo import (  # noqa: E402
    COLS,
    GRAVITY_INTERVAL,
    PLACE_EVERY,
    PULSE_EVERY,
    PULSES_ON_LOCK,
    PULSES_PRE_CMD,
    ROWS,
    SAMPLE_EVERY,
    CachedGraphPlacementBonus,
    EligibilityWindow,
    apply_lock_credit,
    build_symbioid,
    cached_graph_intent,
    game_boundary_gc,
    maybe_mid_game_gc,
    poles_to_content_keys,
    sample_into_symbioid,
    sample_packing_meta_into_symbioid,
    set_console_emit,
    update_pred_pack_for_target,
)
from symbioid.world.tetris import ActionCipher, TetrisWorld  # noqa: E402
from symbioid.world.tetris_learn import TetrisCoach  # noqa: E402
import random as _random


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-thoughts", type=int, default=15000)
    p.add_argument("--max-ms-frame", type=float, default=50.0)
    args = p.parse_args(argv)

    set_console_emit(False)
    seed = int(args.seed)
    rng = _random.Random(seed)
    cipher = ActionCipher.random(rng)
    world = TetrisWorld(
        cols=COLS,
        rows=ROWS,
        gravity_interval=GRAVITY_INTERVAL,
        cipher=cipher,
        rng=_random.Random(seed + 17),
    )
    coach = TetrisCoach(
        network_primary=True,
        map_threshold=1,
        rng=_random.Random(seed + 31),
    )
    s = build_symbioid(world)
    coach.graph_placement_weight = 0.60
    coach.graph_placement_bonus = CachedGraphPlacementBonus(s)
    elig = EligibilityWindow(max_ticks=24)
    s.start_processes()
    rows: list[dict] = []
    try:
        for g in range(1, int(args.games) + 1):
            frame = 0
            elig.clear()
            s.mind.tft.reset_episode(clear_counts=True)
            t_pulse = t_place = 0.0
            n_pulse = n_place = 0
            t0g = time.perf_counter()
            while not world.game_over and frame < int(args.max_frames):
                if frame % SAMPLE_EVERY == 0:
                    sample_into_symbioid(s, world, tick=frame)
                if s.mind.dynamics_enabled and frame % PULSE_EVERY == 0:
                    t0 = time.perf_counter()
                    s.pulse_tick()
                    t_pulse += time.perf_counter() - t0
                    n_pulse += 1
                if s.mind.dynamics_enabled and PULSES_PRE_CMD > 0:
                    for _ in range(int(PULSES_PRE_CMD)):
                        t0 = time.perf_counter()
                        s.pulse_tick()
                        t_pulse += time.perf_counter() - t0
                        n_pulse += 1
                maybe_mid_game_gc(s, frame)
                prev_pieces = world.pieces_placed
                t0 = time.perf_counter()
                preferred, g_bias, poles, _ = cached_graph_intent(
                    s, world, coach, frame, place_every=PLACE_EVERY
                )
                update_pred_pack_for_target(s, world, coach)
                t_place += time.perf_counter() - t0
                n_place += 1
                sample_packing_meta_into_symbioid(s, world, tick=frame, coach=coach)
                elig.push(poles_to_content_keys(s, poles))
                coach.tick(world, preferred_intent=preferred, graph_bias=g_bias)
                if world.pieces_placed > prev_pieces:
                    coach.last_lock_frame = frame  # type: ignore[attr-defined]
                    apply_lock_credit(s, coach, elig, poles=poles)
                    if s.mind.dynamics_enabled and PULSES_ON_LOCK > 0:
                        for _ in range(int(PULSES_ON_LOCK)):
                            t0 = time.perf_counter()
                            s.pulse_tick()
                            t_pulse += time.perf_counter() - t0
                            n_pulse += 1
                frame += 1
            wall = time.perf_counter() - t0g
            with s.graph_lock:
                n_th = len(s.thoughts)
            ms_pulse = (t_pulse / max(1, n_pulse)) * 1000
            ms_place = (t_place / max(1, n_place)) * 1000
            # Approximate frame work: place + (pulses per frame * ms_pulse)
            pulses_per_frame = (1.0 / max(1, PULSE_EVERY)) + (
                PULSES_ON_LOCK / max(1.0, frame / max(1, world.pieces_placed or 1))
            )
            # Simpler: total pulse+place wall / frames
            ms_frame = ((t_pulse + t_place) / max(1, frame)) * 1000
            fps = frame / max(1e-9, wall)
            row = {
                "g": g,
                "frames": frame,
                "thoughts": n_th,
                "ms_pulse": round(ms_pulse, 2),
                "ms_place": round(ms_place, 2),
                "ms_frame": round(ms_frame, 2),
                "fps": round(fps, 1),
                "wall_s": round(wall, 2),
                "score": world.score,
            }
            rows.append(row)
            print(row, flush=True)
            gc = game_boundary_gc(s)
            print(
                f"  [gc] th {gc['thoughts_before']}→{gc['thoughts_after']} "
                f"rm={gc['removed']}",
                flush=True,
            )
            if g < int(args.games):
                coach.on_new_game(world, record=True)
    finally:
        s.stop_processes()

    last = rows[-1] if rows else {}
    max_th = max((r["thoughts"] for r in rows), default=0)
    max_ms = max((r["ms_frame"] for r in rows), default=0.0)
    ok_th = max_th <= int(args.max_thoughts)
    ok_ms = max_ms <= float(args.max_ms_frame)
    print(
        f"\n=== survival max_thoughts={max_th} (≤{args.max_thoughts}? {ok_th}) "
        f"max_ms_frame={max_ms:.1f} (≤{args.max_ms_frame}? {ok_ms}) ===",
        flush=True,
    )
    print(
        f"clocks PULSE_EVERY={PULSE_EVERY} PLACE_EVERY={PLACE_EVERY} "
        f"PULSES_PRE={PULSES_PRE_CMD} ON_LOCK={PULSES_ON_LOCK}",
        flush=True,
    )
    return 0 if (ok_th and ok_ms) else 1


if __name__ == "__main__":
    raise SystemExit(main())
