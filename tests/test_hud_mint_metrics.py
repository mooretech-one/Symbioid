"""v0.0.62 HUD: mint rate + live residual (no admits_mint GC)."""

from __future__ import annotations

from tetris_demo import (
    game_boundary_gc,
    live_observation_registry_count,
    mint_rate_delta,
    sample_into_symbioid,
    build_symbioid,
)
from symbioid.world.tetris import TetrisWorld


def test_mint_rate_delta_first_and_positive():
    assert mint_rate_delta(10, None) == 0
    assert mint_rate_delta(10, 10) == 0
    assert mint_rate_delta(15, 10) == 5
    assert mint_rate_delta(8, 10) == 0  # never negative rate


def test_mint_rate_does_not_mutate_admits_mint():
    w = TetrisWorld()
    s = build_symbioid(w)
    before = int(s.mind.admits_mint)
    _ = mint_rate_delta(before + 3, before)
    assert int(s.mind.admits_mint) == before


def test_live_residual_falls_or_stable_after_gc():
    w = TetrisWorld()
    s = build_symbioid(w)
    for t in range(12):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    admits_before = int(s.mind.admits_mint)
    residual_before = live_observation_registry_count(s.mind)
    assert residual_before >= 0
    stats = game_boundary_gc(s)
    residual_after = live_observation_registry_count(s.mind)
    # GC may purge cell/packing keys → residual can fall; never grows from GC alone
    assert residual_after <= residual_before
    # admits_mint is audit-only — never decremented by GC
    assert int(s.mind.admits_mint) == admits_before
    assert "thoughts_after" in stats
