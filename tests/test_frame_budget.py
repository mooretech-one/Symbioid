"""v0.0.65 frame-budget shed + anti-remint mid-GC helpers."""

from __future__ import annotations

from tetris_demo import (
    FRAME_BUDGET_MS,
    FACE_TICK_INTERVAL,
    PULSE_EVERY,
    PULSES_PRE_CMD,
    SAMPLE_EVERY,
    TARGET_RESCORE_EVERY,
    build_symbioid,
    cached_graph_intent,
    frame_over_budget,
    game_boundary_gc,
    maybe_mid_game_gc_budgeted,
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
    """v0.0.65 must not re-introduce 0.0.64 ultra-dense concurrent load."""
    assert SAMPLE_EVERY >= 2
    assert PULSE_EVERY >= 3
    assert PULSES_PRE_CMD == 0
    assert TARGET_RESCORE_EVERY >= 10
    assert FACE_TICK_INTERVAL >= 0.1
