"""v0.0.57 multi-game survival helpers."""

from __future__ import annotations

from tetris_demo import build_symbioid, game_boundary_gc, thought_counts_active_inactive
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
    # mint some cold structure via sampling
    from tetris_demo import sample_into_symbioid

    for t in range(5):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    before = len(s.thoughts)
    stats = game_boundary_gc(s)
    assert "thoughts_after" in stats
    assert stats["thoughts_before"] == before
    assert stats["thoughts_after"] <= before
