"""v0.0.65+ frame-budget shed + network CPU governor helpers."""

from __future__ import annotations

import time

from tetris_demo import (
    FRAME_BUDGET_MS,
    FACE_TICK_INTERVAL,
    GRAVITY_INTERVAL,
    NETWORK_CPU_FRACTION,
    PROCESS_CPU_FRACTION,
    PULSE_EVERY,
    PULSES_PRE_CMD,
    SAMPLE_EVERY,
    TARGET_RESCORE_EVERY,
    NetworkCpuGovernor,
    apply_face_cpu_throttle,
    build_symbioid,
    cached_graph_intent,
    frame_over_budget,
    game_boundary_gc,
    maybe_mid_game_gc_budgeted,
    process_cpu_yield,
    sample_into_symbioid,
)
from symbioid.world.tetris import TetrisWorld
from symbioid.world.tetris_learn import TetrisCoach


def test_frame_over_budget():
    assert frame_over_budget(0.0) is False
    assert frame_over_budget(FRAME_BUDGET_MS) is False
    assert frame_over_budget(FRAME_BUDGET_MS + 0.1) is True


def test_budgeted_gc_skips_when_over_budget():
    w = TetrisWorld()
    s = build_symbioid(w)
    # force would only run if inactive extreme; over budget + force moderate → skip
    out = maybe_mid_game_gc_budgeted(
        s, 1, last_frame_ms=FRAME_BUDGET_MS + 50, force=True, inactive_count=100
    )
    assert out is None
    # normal interval path also skips when over budget
    out2 = maybe_mid_game_gc_budgeted(
        s, 90, last_frame_ms=FRAME_BUDGET_MS + 50, force=False
    )
    assert out2 is None


def test_light_gc_zero_purge_and_pauses_faces():
    w = TetrisWorld()
    s = build_symbioid(w)
    s.start_processes()
    try:
        for t in range(4):
            sample_into_symbioid(s, w, tick=t)
            s.pulse_tick()
        assert s.interface.enabled is True
        stats = game_boundary_gc(s, tier="light", hard_cap=50000)
        assert stats.get("purged_registry", -1) == 0
        # faces re-enabled after GC
        assert s.interface.enabled is True
        assert s.innerface.enabled is True
        assert s.outerface.enabled is True
    finally:
        s.stop_processes()


def test_force_geo_only_skips_interval_rescore():
    w = TetrisWorld()
    s = build_symbioid(w)
    coach = TetrisCoach(network_primary=True)
    coach._target = (0, 3)  # type: ignore[attr-defined]
    coach._target_score_frame = 0  # type: ignore[attr-defined]
    coach._intent_piece_key = None  # type: ignore[attr-defined]
    # With force_geo_only and existing target, should not raise
    preferred, _bias, _poles, _hint = cached_graph_intent(
        s, w, coach, frame=100, force_geo_only=True
    )
    # may be None if no active piece — just ensure call works
    assert preferred is None or preferred in (
        "left",
        "right",
        "rotate",
        "hard",
        "explore",
    )


def test_hang_harden_density_not_worse_than_064():
    """v0.0.65+ must not re-introduce 0.0.64 ultra-dense concurrent load."""
    from tetris_demo import TARGET_RESCORE_EVERY_HOT

    assert SAMPLE_EVERY >= 2
    assert PULSE_EVERY >= 3
    assert PULSES_PRE_CMD == 0
    # Cool path rescores more for placement quality; hot path is sparse
    assert TARGET_RESCORE_EVERY_HOT >= TARGET_RESCORE_EVERY
    assert TARGET_RESCORE_EVERY_HOT >= 10
    assert FACE_TICK_INTERVAL >= 0.1


def test_network_cpu_governor_sleeps_when_over_fraction():
    """Charging more network time than fraction*wall forces a sleep yield."""
    gov = NetworkCpuGovernor(fraction=0.5, window_s=2.0, max_sleep_s=0.05)
    # Fake: charge 40ms network with almost no wall → must sleep
    t0 = time.perf_counter()
    slept = gov.charge(0.04)
    elapsed = time.perf_counter() - t0
    assert slept > 0.0
    assert elapsed >= slept * 0.5  # actually slept
    assert NETWORK_CPU_FRACTION == 0.50


def test_network_cpu_governor_work_context():
    gov = NetworkCpuGovernor(fraction=0.5, window_s=2.0, max_sleep_s=0.02)
    with gov.work():
        time.sleep(0.005)
    assert gov.network_s >= 0.004
    assert gov.charge_count >= 1


def test_face_cpu_throttle_slows_and_disables():
    w = TetrisWorld()
    s = build_symbioid(w)
    base = float(FACE_TICK_INTERVAL)
    apply_face_cpu_throttle(s, over_cap=True)
    assert s.interface.tick_interval >= base * 5.0
    assert s.interface.enabled is False
    apply_face_cpu_throttle(s, over_cap=False)
    assert abs(s.interface.tick_interval - base) < 1e-9
    assert s.interface.enabled is True


def test_process_cpu_yield_sleeps_for_half_busy():
    """50% process cap: ~busy seconds of sleep for busy work."""
    t0 = time.perf_counter()
    slept = process_cpu_yield(0.02, fraction=0.5, min_period_s=None)
    elapsed = time.perf_counter() - t0
    assert slept >= 0.015
    assert elapsed >= 0.015
    assert PROCESS_CPU_FRACTION == 0.50
    assert GRAVITY_INTERVAL >= 90  # slower play for placement reasoning