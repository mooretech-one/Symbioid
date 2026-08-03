"""v0.0.57+ multi-game survival helpers (incl. v0.0.61 mid-game GUI GC)."""

from __future__ import annotations

from tetris_demo import (
    MID_GAME_GC_EVERY,
    MID_GAME_HARD_CAP,
    build_symbioid,
    game_boundary_gc,
    maybe_mid_game_gc,
    thought_counts_active_inactive,
)
from symbioid.world.tetris import TetrisWorld


def test_thought_counts_lock_safe():
    w = TetrisWorld()
    s = build_symbioid(w)
    a, i = thought_counts_active_inactive(s)
    assert a >= 0 and i >= 0
    assert a + i == len(s.thoughts)


def test_game_boundary_gc_runs():
    w = TetrisWorld()
    s = build_symbioid(w)
    from tetris_demo import sample_into_symbioid

    for t in range(5):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    before = len(s.thoughts)
    stats = game_boundary_gc(s)
    assert "thoughts_after" in stats
    assert stats["thoughts_before"] == before
    assert stats["thoughts_after"] <= before
    assert "archives_trimmed" in stats


def test_maybe_mid_game_gc_skips_off_interval():
    w = TetrisWorld()
    s = build_symbioid(w)
    assert maybe_mid_game_gc(s, 0) is None
    assert maybe_mid_game_gc(s, 1) is None  # not multiple of MID_GAME_GC_EVERY


def test_maybe_mid_game_gc_force_always_runs():
    w = TetrisWorld()
    s = build_symbioid(w)
    from tetris_demo import sample_into_symbioid

    for t in range(8):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    forced = maybe_mid_game_gc(s, 1, inactive_count=99999, force=True)
    assert forced is not None
    assert forced["thoughts_after"] <= forced["thoughts_before"]
    frame = MID_GAME_GC_EVERY if MID_GAME_GC_EVERY > 0 else 45
    stats = maybe_mid_game_gc(s, frame)
    # Interval path may skip when graph still tiny
    if stats is not None:
        assert stats["thoughts_after"] <= max(stats["thoughts_before"], MID_GAME_HARD_CAP + 500)


def test_trim_inactive_archives_via_gc():
    w = TetrisWorld()
    s = build_symbioid(w)
    stats = game_boundary_gc(s, max_inactive_archives=2)
    assert stats["archives_trimmed"] >= 0
