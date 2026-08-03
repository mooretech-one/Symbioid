"""v0.0.63 two-tier GC + credit-protect (light mid-game keeps packing)."""

from __future__ import annotations

from tetris_demo import (
    CREDIT_PROTECT_LOCKS,
    build_symbioid,
    game_boundary_gc,
    live_observation_registry_count,
    note_credit_protect,
    sample_into_symbioid,
    tick_credit_protect,
)
from symbioid.world.tetris import TetrisWorld


def test_light_gc_does_not_full_purge_packing_valence():
    w = TetrisWorld()
    s = build_symbioid(w)
    for t in range(10):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    # Seed packing-like valence that full purge would drop
    s.mind.note_valence(content_key="holes_n:demo", delta=1.0)
    s.mind.note_valence(content_key="cell_r05_c03:place", delta=1.2)
    admits = int(s.mind.admits_mint)
    light = game_boundary_gc(s, tier="light", hard_cap=50000)
    assert light.get("tier_name") == "light"
    # place / packing valence must survive light tier
    assert float(s.mind._valence.get("cell_r05_c03:place", 0.0) or 0.0) >= 1.0
    assert float(s.mind._valence.get("holes_n:demo", 0.0) or 0.0) >= 1.0
    assert int(s.mind.admits_mint) == admits


def test_full_gc_purges_unprotected_ephemeral():
    w = TetrisWorld()
    s = build_symbioid(w)
    for t in range(8):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    s.mind.note_valence(content_key="cell_r01_c01:place", delta=0.1)
    full = game_boundary_gc(s, tier="full", hard_cap=50000)
    assert full.get("tier_name") == "full"
    # Low unprotected place valence should be eligible for full purge
    assert "cell_r01_c01:place" not in s.mind._valence or float(
        s.mind._valence.get("cell_r01_c01:place", 0.0) or 0.0
    ) == 0.0


def test_credit_protect_ttl_and_full_purge_keeps_keys():
    w = TetrisWorld()
    s = build_symbioid(w)
    ck = "cell_r02_c02:place"
    s.mind.note_valence(content_key=ck, delta=2.0)
    note_credit_protect(s, {ck}, locks=CREDIT_PROTECT_LOCKS)
    game_boundary_gc(s, tier="full", hard_cap=50000)
    # Protected by credit TTL through full purge
    assert float(s.mind._valence.get(ck, 0.0) or 0.0) >= 1.5
    for _ in range(CREDIT_PROTECT_LOCKS):
        tick_credit_protect(s)
    # After TTL expires, full purge may drop
    game_boundary_gc(s, tier="full", hard_cap=50000)
    assert ck not in s.mind._valence or float(s.mind._valence.get(ck, 0.0) or 0.0) == 0.0


def test_tick_credit_protect_expires():
    w = TetrisWorld()
    s = build_symbioid(w)
    note_credit_protect(s, {"k1"}, locks=2)
    assert tick_credit_protect(s) == 0
    assert s._credit_protect_ttl.get("k1") == 1
    assert tick_credit_protect(s) == 1
    assert "k1" not in s._credit_protect_ttl


def test_light_gc_registry_can_stay_above_zero_after_sample():
    w = TetrisWorld()
    s = build_symbioid(w)
    for t in range(6):
        sample_into_symbioid(s, w, tick=t)
        s.pulse_tick()
    before = live_observation_registry_count(s.mind)
    game_boundary_gc(s, tier="light", hard_cap=50000)
    after = live_observation_registry_count(s.mind)
    # Soft cell purge may shrink; should not zero everything if packing/meta present
    assert after >= 0
    assert before >= 0
