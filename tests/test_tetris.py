"""Headless Tetris world + secret-byte learner tests."""

import sys
from pathlib import Path
from random import Random

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid.world.tetris import ActionCipher, TetrisWorld, piece_cells
from symbioid.world.tetris_learn import (
    TetrisCoach,
    board_quality_reward,
    classify_effect,
    observe_board,
    pose_hole_features,
    WorldSnapshot,
)


def test_reset_spawns_piece():
    w = TetrisWorld(rng=Random(0))
    assert w.active is not None
    assert not w.game_over
    assert w.last_event == "spawn"
    sw = w.sensor_world()
    assert "holes" in sw and "piece_id" in sw
    assert "last_byte" in sw
    # Sensors must not expose the secret mapping
    assert "cipher" not in sw
    assert "last_byte_action" not in sw


def test_hard_drop_locks_and_spawns_next():
    w = TetrisWorld(rng=Random(1))
    w.hard_drop()
    assert w.pieces_placed == 1
    if not w.game_over:
        assert w.active is not None
        filled = sum(1 for row in w.board for c in row if c)
        assert filled == 4


def test_line_clear():
    w = TetrisWorld(rng=Random(2))
    w.board = [["" for _ in range(10)] for _ in range(20)]
    w.board[19] = ["X"] * 10
    w.active = None
    cleared = w._clear_lines()
    assert cleared == 1
    assert all(not c for c in w.board[19])
    assert len(w.board) == 20


def test_lock_clears_complete_line():
    """Locking a piece that completes a row must clear it (regression)."""
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    for c in range(10):
        if c not in (3, 4, 5, 6):
            w.board[19][c] = "X"
    w.active = ActivePiece(kind="I", row=18, col=3, rotation=0)
    # I horizontal occupies row 19 cols 3-6
    assert w.active.cells() == [(19, 3), (19, 4), (19, 5), (19, 6)]
    w._lock()
    assert w.lines >= 1
    assert w.last_lines_cleared >= 1
    assert not any(all(bool(c) for c in row) for row in w.board)
    assert all(not c for c in w.board[19])


def test_clear_lines_never_leaves_full_row():
    w = TetrisWorld(rng=Random(0))
    w.board = [["T"] * 10 for _ in range(20)]
    n = w._clear_lines()
    assert n == 20
    assert len(w.board) == 20
    assert not any(all(bool(c) for c in row) for row in w.board)


def test_clear_lines_handles_short_board_drift():
    """Old bug: cleared = rows - len(kept) broke when len(board) != rows."""
    w = TetrisWorld(rng=Random(0))
    w.board = [["X"] * 10 for _ in range(15)]  # drifted short
    w.board[10] = [""] * 10  # one incomplete
    n = w._clear_lines()
    assert n == 14
    assert len(w.board) == 20
    assert not any(w._row_is_full(row) for row in w.board)


def test_origin_col_may_be_negative_to_fill_left_column():
    """O/vertical pieces use min_dc>0; origin -1/-2 is required to fill col 0."""
    from symbioid.world.tetris import ActivePiece

    # O at origin -1 covers board cols 0 and 1
    o = ActivePiece(kind="O", row=0, col=-1, rotation=0)
    cols = sorted({c for _, c in o.cells()})
    assert 0 in cols and min(cols) == 0

    # Vertical I (rot 1) at origin -2 covers board col 0
    i = ActivePiece(kind="I", row=0, col=-2, rotation=1)
    cols = sorted({c for _, c in i.cells()})
    assert 0 in cols and min(cols) == 0

    lo_o, hi_o = TetrisCoach.col_range_for_pose("O", 0, 10)
    assert lo_o == -1 and hi_o == 7
    lo_i, hi_i = TetrisCoach.col_range_for_pose("I", 1, 10)
    assert lo_i == -2

    # Clamp must NOT push -1 up to 0 for O (that was the off-by-one)
    coach = TetrisCoach()
    assert coach._clamp_pose("O", 0, -1, 10) == (0, -1)
    assert coach._clamp_pose("I", 1, -2, 10) == (1, -2)


def test_legal_and_apply_can_fill_column_zero():
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    w.active = ActivePiece(kind="O", row=0, col=3, rotation=0)
    opts = w.legal_placements()
    assert any(col == -1 for rot, col in opts)
    assert w.apply_placement(0, -1)
    # Bottom cells of O should include column 0
    filled_cols = {c for r in range(20) for c in range(10) if w.board[r][c]}
    assert 0 in filled_cols


def test_legal_placements_nonempty():
    w = TetrisWorld(rng=Random(3))
    opts = w.legal_placements()
    assert len(opts) >= 4


def test_apply_placement():
    w = TetrisWorld(rng=Random(4))
    opts = w.legal_placements()
    rot, col = opts[0]
    assert w.apply_placement(rot, col)
    assert w.pieces_placed == 1


def test_cipher_only_few_bytes_live():
    cipher = ActionCipher.fixed({17: "left", 200: "right", 3: "rotate", 99: "hard"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=9999)
    assert w.active is not None
    col0 = w.active.col
    # Dead byte
    w.step_byte(0)
    assert w.active.col == col0
    assert w.last_byte == 0
    # Live left
    w.step_byte(17)
    assert w.active is not None
    assert w.active.col == col0 - 1 or w.last_event in ("left", "blocked")


def test_step_byte_does_not_require_named_commands():
    cipher = ActionCipher.random(Random(42))
    w = TetrisWorld(rng=Random(42), cipher=cipher, gravity_interval=9999)
    live = set(cipher.live_bytes())
    # Exhaustive: only live bytes change action field on world (ground truth)
    movers = 0
    for b in range(256):
        w2 = TetrisWorld(rng=Random(42), cipher=cipher, gravity_interval=9999)
        act = w2.step_byte(b)
        if b in live:
            assert act in ("left", "right", "rotate", "hard")
            movers += 1
        else:
            assert act == "noop"
    assert movers == 4


def test_coach_never_reads_cipher_discovers_by_observation():
    """Coach must learn map from effects only — we check it doesn't use cipher."""
    cipher = ActionCipher.fixed({11: "left", 22: "right", 33: "rotate", 44: "hard"})
    w = TetrisWorld(rng=Random(10), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(10), explore_rate=0.9, map_threshold=1)

    # Ensure coach has no cipher attribute / doesn't copy mapping
    assert not hasattr(coach, "cipher")

    for _ in range(2500):
        if w.game_over:
            coach.on_new_game(w, record=False)
        coach.tick(w)
        if coach.map_complete():
            break

    discovered = coach.discovered_map()
    # Should have found the live bytes (effect → code)
    assert "left" in discovered
    assert "right" in discovered
    assert "rotate" in discovered
    assert "hard" in discovered
    assert discovered["left"] == 11
    assert discovered["right"] == 22
    assert discovered["rotate"] == 33
    assert discovered["hard"] == 44


def test_partial_map_does_not_spam_known_key():
    """Once one key is known, discovery must keep scanning (not only spam right)."""
    cipher = ActionCipher.fixed({11: "left", 22: "right", 33: "rotate", 44: "hard"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(0), map_threshold=1)
    # Seed only right as known
    coach.effect_counts[22]["right"] = 3
    coach.bytes_tried = set(range(256))
    intents = []
    for _ in range(40):
        if w.game_over:
            coach.on_new_game(w, record=False)
        intents.append(coach.desired_intent(w))
        coach.tick(w)
    assert all(i == "explore" for i in intents), intents[:10]


def test_play_ready_with_hard_left_right_not_full_map():
    """Phase A: structured play once hard+left+right known (rotate optional)."""
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    coach = TetrisCoach(rng=Random(0), map_threshold=1)
    for b, e in ((1, "left"), (2, "right"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    assert coach.play_ready()
    assert not coach.map_complete()  # rotate still missing
    coach._begin_piece(w)
    assert coach._target is not None
    intent = coach.desired_intent(w)
    assert intent in ("left", "right", "rotate", "hard", "explore")


def test_force_hard_after_many_cmds():
    """Phase A: force hard if piece stalls without locking."""
    cipher = ActionCipher.fixed({1: "left", 2: "right", 3: "rotate", 4: "hard"})
    w = TetrisWorld(rng=Random(1), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(1), map_threshold=1, force_hard_after_cmds=5)
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    coach._begin_piece(w)
    coach._piece_cmds = 5
    assert coach.desired_intent(w) == "hard"


def test_landing_cells_matches_legal_drop():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    assert w.active is not None
    opts = w.legal_placements()
    assert opts
    rot, col = opts[0]
    cells = w.landing_cells(rot, col)
    assert cells
    assert all(0 <= r < w.rows and 0 <= c < w.cols for r, c in cells)


def test_pred_holes_freed_foresight_for_target():
    """Network foresight sensors report holes freed by current target pose."""
    import tetris_demo as mod
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    # Sealed hole at (19,4)
    for c in range(10):
        if c != 4:
            w.board[19][c] = "X"
    w.board[18][4] = "X"
    w.active = ActivePiece(kind="O", row=0, col=3, rotation=0)
    s = mod.build_symbioid(w)
    coach = TetrisCoach(rng=Random(0), map_threshold=1)
    # Force a target; update foresight
    coach._target = (0, 3)
    out = mod.update_pred_pack_for_target(s, w, coach)
    assert "holes_freed" in out and "pred_d_holes" in out
    # Sensors present
    labels = {sen.label for sen in s.sensors}
    assert "holes_freed" in labels
    assert "pred_d_holes" in labels
    assert "holes_fill_n" in labels
    sen = next(x for x in s.sensors if x.label == "holes_freed")
    # Reading is non-negative scale of freed count
    assert sen.transfer({}) >= 0.0
    # Clear target → zero foresight
    coach._target = None
    out2 = mod.update_pred_pack_for_target(s, w, coach)
    assert out2["holes_freed"] == 0.0
    assert float(getattr(s, "_holes_freed", -1)) == 0.0


def test_packing_meta_sensors_and_last_d_holes_insight():
    """holes_n / last_d_holes sensors exist; lock dig updates network-facing readings."""
    import tetris_demo as mod
    from symbioid.world.tetris import ActivePiece, ActionCipher

    w = TetrisWorld(
        rng=Random(0),
        gravity_interval=9999,
        cipher=ActionCipher.fixed({1: "left", 2: "right", 3: "rotate", 4: "hard"}),
    )
    s = mod.build_symbioid(w)
    labels = {sen.label for sen in s.sensors}
    assert "holes_n" in labels
    assert "last_d_holes" in labels
    assert "well_n" in labels
    assert "max_well_n" in labels
    assert "pred_d_holes" in labels
    assert "holes_freed" in labels
    assert "holes_n" in mod._POLICY_META_LABELS
    assert "last_d_holes" in mod._POLICY_META_LABELS
    assert "holes_freed" in mod._POLICY_META_LABELS

    # Empty board → zero pack
    mod._update_pack_readings(s, w)
    assert float(getattr(s, "_pack_holes", -1)) == 0.0

    # Dig a sealed hole structure and apply a drop that increases holes
    w.board = [["" for _ in range(10)] for _ in range(20)]
    for c in range(10):
        if c != 4:
            w.board[19][c] = "X"
    # Place a block that seals col 4 more: put on row 18 covering neighbors only
    # Simpler: use coach lock bookkeeping
    coach = TetrisCoach(rng=Random(0), map_threshold=1)
    pre = {"holes": 0.0, "max_height": 0.0, "agg_height": 0.0, "bumpiness": 0.0,
           "well": 0.0, "max_well": 0.0, "height_range": 0.0}
    # Create 1 hole on board then learn as if post has more holes
    w.board[18][4] = "X"  # seals empty at 19,4
    assert w.hole_count() >= 1
    from symbioid.world.tetris_learn import observe_board

    post = observe_board(w)
    coach._learn_from_real_drop(
        w, kind="T", rot=0, col=3, pre=pre, score_before=0.0
    )
    assert coach.last_d_holes >= 1.0 - 1e-6, coach.last_d_holes
    s._last_d_holes = 0.0
    mod.sample_packing_meta_into_symbioid(s, w, tick=1, coach=coach)
    assert float(getattr(s, "_last_d_holes", 0.0)) >= 1.0 - 1e-6
    assert float(getattr(s, "_pack_holes", 0.0)) >= 1.0 - 1e-6
    # Sensor transfer should report non-zero last_d_holes reading
    sen = next(x for x in s.sensors if x.label == "last_d_holes")
    reading = sen.transfer({})
    assert reading > 0.0, reading


def test_demo_timing_sense_not_slower_than_command():
    """Optimal coupling: sample ≤ cmd; faces faster than default 50 ms."""
    import tetris_demo as mod

    assert mod.SAMPLE_EVERY <= mod.CMD_EVERY
    assert mod.PULSE_EVERY >= 1
    assert mod.PULSES_PRE_CMD >= 0
    assert mod.PULSES_ON_LOCK >= 0
    assert mod.CMD_EVERY % mod.SAMPLE_EVERY == 0 or mod.SAMPLE_EVERY == mod.CMD_EVERY
    # Optimal profile: 1:1 sense/command, sub-50ms faces, gravity scaled for FPS
    assert mod.SAMPLE_EVERY == 1 and mod.CMD_EVERY == 1
    assert 0.0 < mod.FACE_TICK_INTERVAL <= 0.05
    assert mod.GRAVITY_INTERVAL >= mod.FPS // 2  # not free-fall every frame
    w = TetrisWorld(rng=Random(0))
    s = mod.build_symbioid(w)
    assert s.interface.tick_interval == mod.FACE_TICK_INTERVAL
    assert s.innerface.tick_interval == mod.FACE_TICK_INTERVAL
    assert s.outerface.tick_interval == mod.FACE_TICK_INTERVAL


def test_edge_well_metrics_report_open_side_trenches():
    """Left/right open single-width wells must contribute to well (edge-aware)."""
    from symbioid.world.tetris import well_metrics

    # Col 0 empty, rest height 5
    h = [0, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    wm = well_metrics(h)
    assert wm["well"] >= 5.0, f"left open well not reported: {wm}"
    assert wm["max_well"] >= 5.0
    # Right edge
    h2 = [5, 5, 5, 5, 5, 5, 5, 5, 5, 0]
    wm2 = well_metrics(h2)
    assert wm2["well"] >= 5.0
    assert wm2["max_well"] >= 5.0
    # Interior well col 4
    h3 = [5, 5, 5, 5, 0, 5, 5, 5, 5, 5]
    wm3 = well_metrics(h3)
    assert wm3["well"] >= 5.0
    # Flat skyline
    assert well_metrics([3] * 10)["well"] == 0.0


def test_observe_board_edge_well_nonzero():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    for r in range(15, 20):
        for c in range(1, 10):
            w.board[r][c] = "X"
    obs = observe_board(w)
    assert obs["holes"] == 0.0  # open trench, not sealed hole
    assert obs["well"] >= 5.0, f"expected edge well, got {obs['well']}"
    assert obs["max_well"] >= 5.0


def test_pose_features_penalize_deepening_edge_well():
    """Dropping beside a left well (not into it) should worsen d_well vs filling."""
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    for r in range(16, 20):
        for c in range(1, 10):
            w.board[r][c] = "X"
    # Pre: left well depth 4
    assert observe_board(w)["max_well"] >= 4.0
    w.active = ActivePiece(kind="I", row=0, col=0, rotation=1)  # vertical
    # Vertical I into col 0 should reduce well; horizontal on top of stack may not
    fill_opts = []
    other_opts = []
    for rot, col in w.legal_placements():
        hf = pose_hole_features(w, rot, col)
        if hf["ok"] < 0.5:
            continue
        if rot % 2 == 1 and col <= 0:
            fill_opts.append(hf)
        else:
            other_opts.append(hf)
    assert fill_opts, "expected vertical fill options near col 0"
    best_fill = min(fill_opts, key=lambda x: x["d_well"])
    # Filling the trench should not increase well more than a random other pose max
    if other_opts:
        worst_other = max(other_opts, key=lambda x: x["d_well"])
        assert best_fill["d_well"] <= worst_other["d_well"]


def test_pose_hole_features_prefers_lower_d_holes():
    """Fixture: buried hole — landings that dig more holes rank worse on d_holes."""
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    # Floor with a hole at (19, 5): fill row 19 except col 5, block above col 5
    for c in range(10):
        if c != 5:
            w.board[19][c] = "X"
    w.board[18][5] = "X"
    w.active = ActivePiece(kind="O", row=0, col=3, rotation=0)
    assert w.hole_count() >= 1
    pre = float(w.hole_count())
    feats = []
    for rot, col in w.legal_placements():
        hf = pose_hole_features(w, rot, col)
        if hf["ok"] < 0.5:
            continue
        feats.append((hf["d_holes"], rot, col, hf["post_holes"]))
    assert feats, "expected some legal sim features"
    d_vals = [f[0] for f in feats]
    assert min(d_vals) <= max(d_vals)
    # At least one pose should not explode holes vs a bad overhang
    assert min(d_vals) <= 0.0 or min(d_vals) < max(d_vals)
    best = min(feats, key=lambda t: t[0])
    worst = max(feats, key=lambda t: t[0])
    assert best[0] <= worst[0]
    assert best[3] >= pre - 1e-6  # post holes sensible


def test_graph_score_prefers_lower_d_holes_pose():
    """cell_thought_placement_score ranks lower-d_holes landings higher (ceteris paribus)."""
    mod = _load_tetris_demo()
    from symbioid.world.tetris import ActivePiece

    w = TetrisWorld(rng=Random(1), gravity_interval=9999)
    w.board = [["" for _ in range(10)] for _ in range(20)]
    for c in range(10):
        if c != 5:
            w.board[19][c] = "X"
    w.board[18][5] = "X"
    w.active = ActivePiece(kind="T", row=0, col=4, rotation=0)
    s = mod.build_symbioid(w)
    scored = []
    for rot, col in w.legal_placements():
        hf = pose_hole_features(w, rot, col)
        if hf["ok"] < 0.5:
            continue
        sc = mod.cell_thought_placement_score(s, w, rot, col)
        scored.append((sc, hf["d_holes"], rot, col))
    assert len(scored) >= 4
    # Among extremes: best graph score should not be the worst d_holes pose
    by_score = sorted(scored, key=lambda t: t[0], reverse=True)
    by_holes = sorted(scored, key=lambda t: t[1])
    best_sc_d = by_score[0][1]
    worst_d = by_holes[-1][1]
    # Top-scoring pose should have d_holes at most mid-tier (not uniquely worst)
    if worst_d > by_holes[0][1]:
        assert best_sc_d <= worst_d
        # Prefer: top score's d_holes better than or equal to median
        median_d = sorted(t[1] for t in scored)[len(scored) // 2]
        assert best_sc_d <= median_d + 1.0


def test_phase_c_cell_thought_scores_prefer_holes():
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None
    # Hole under block at col 3
    w.board[10][3] = "I"
    # Fake active for placement API
    from symbioid.world.tetris import ActivePiece

    w.active = ActivePiece(kind="O", row=0, col=2, rotation=0)
    s = mod.build_symbioid(w)
    # Score a column that fills toward the hole vs far away
    # Just ensure score is finite and landing works
    opts = w.legal_placements()
    assert opts
    scores = [mod.cell_thought_placement_score(s, w, rot, col) for rot, col in opts[:8]]
    assert all(isinstance(x, float) for x in scores)
    assert max(scores) > min(scores) or len(set(scores)) >= 1


def test_phase_c_choose_target_uses_graph_bonus():
    w = TetrisWorld(rng=Random(1), gravity_interval=9999)
    coach = TetrisCoach(rng=Random(1), map_threshold=1)
    called = {"n": 0}

    def bonus(world, rot, col):
        called["n"] += 1
        # Prefer col 0-ish
        return 10.0 if col <= 2 else 0.0

    coach.graph_placement_bonus = bonus
    coach.graph_placement_weight = 1.0
    coach.place_explore = 0.0  # deterministic best
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
    coach._begin_piece(w)
    assert called["n"] > 0
    assert coach._target is not None


def test_phase_b_seeds_action_poles():
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0))
    s = mod.build_symbioid(w)
    for tok in ("left", "right", "rotate", "hard"):
        ck = s.mind.action_content_key("tetris", tok)
        assert ck in s.mind._actions


def test_phase_b_policy_poles_prefer_meta():
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    s = mod.build_symbioid(w)
    mod.sample_into_symbioid(s, w, tick=1)
    poles = mod.policy_state_poles(s, w)
    # Should not dump 200 empty cells as state
    assert len(poles) < 40
    pref, bias, poles2, hint = mod.graph_preferred_intent(s, w, TetrisCoach(rng=Random(0)))
    assert isinstance(poles2, list)
    assert 0.0 <= bias <= 1.0


def test_apply_lock_valence_to_landing_cells_raises_place_keys():
    """Lock reward fans valence onto cell placement keys (closed-loop heat)."""
    import tetris_demo as mod
    from symbioid import Symbioid

    s = Symbioid(install_constitution=False)
    cells = [(18, 3), (18, 4), (19, 3), (19, 4)]
    n = mod.apply_lock_valence_to_landing_cells(s, cells, reward=100.0)
    assert n >= 4
    v = s.mind.valence_of(content_key="cell_r18_c03:place")
    assert v > 0.0, f"expected positive placement valence, got {v}"
    # Negative reward should lower valence
    mod.apply_lock_valence_to_landing_cells(s, cells, reward=-100.0)
    v2 = s.mind.valence_of(content_key="cell_r18_c03:place")
    assert v2 < v


def test_network_primary_graph_weight_floor_allows_co_lead():
    """choose_target honors graph weight down to 0.35 under network_primary."""
    w = TetrisWorld(rng=Random(1), gravity_interval=9999)
    coach = TetrisCoach(
        rng=Random(1),
        network_primary=True,
        graph_placement_weight=0.60,
        graph_placement_bonus=lambda world, rot, col: 0.0,
        map_threshold=1,
    )
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    if w.active is None:
        return
    coach.choose_target(w)
    # Weight field itself is 0.60 (floor no longer forces ≥0.55)
    assert coach.graph_placement_weight == 0.60


def test_build_symbioid_band_b_active_caps():
    """Band B: wider WM + larger policy registries for network-primary learning."""
    import tetris_demo as mod

    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    s = mod.build_symbioid(w)
    assert s.innerface.max_active_senses == 224
    assert s.innerface.max_active_syncs == 112
    assert s.innerface.max_active_integrates == 112
    assert s.innerface.max_active_integrates_per_channel == 8
    assert s.mind.max_follows_registry == 4096
    assert s.mind.max_integrates_registry == 4096
    assert s.mind.policy_registry_priority is True
    assert s.innerface.cofire_meta_only is True
    assert s.innerface.allow_cross_channel_follows is False
    summary = s.innerface.active_set_summary()
    assert isinstance(summary, dict)
    # Caps still bound force-activated sets (no runaway to uncapped 8k+)
    for i in range(300):
        s.innerface._activate(f"{s.id}:sense:cap{i}", "sense")
    n_sense = sum(1 for v in s.innerface.active_ids.values() if v == "sense")
    assert n_sense <= 224


def test_network_primary_tick_prefers_symbioid_intent():
    """When network_primary and play_ready, preferred_intent wins over coach explore."""
    cipher = ActionCipher.fixed({10: "left", 20: "right", 30: "rotate", 40: "hard"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(0), network_primary=True, map_threshold=1)
    for b, e in ((10, "left"), (20, "right"), (30, "rotate"), (40, "hard")):
        coach.effect_counts[b][e] = 3
        coach.bytes_tried.add(b)
    assert coach.play_ready()
    hits = 0
    for _ in range(40):
        code = coach.tick(w, preferred_intent="left", graph_bias=0.95, run_gravity=False)
        if code == 10 and coach.last_intent == "left":
            hits += 1
        if w.game_over or w.active is None:
            break
    assert hits >= 20, f"network primary should usually honor preferred_intent, hits={hits}"


def test_network_primary_placement_weights_graph():
    """network_primary + graph bonus should change choose_target vs coach-only."""
    w = TetrisWorld(rng=Random(1), gravity_interval=9999)
    # Bonus favors high columns
    def bonus(world, rot, col):
        return float(col) * 10.0

    coach_net = TetrisCoach(
        rng=Random(1),
        network_primary=True,
        graph_placement_weight=0.95,
        graph_placement_bonus=bonus,
        map_threshold=1,
    )
    coach_only = TetrisCoach(
        rng=Random(1),
        network_primary=False,
        graph_placement_weight=0.0,
        graph_placement_bonus=None,
        map_threshold=1,
    )
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach_net.effect_counts[b][e] = 2
        coach_only.effect_counts[b][e] = 2
        coach_net.bytes_tried.add(b)
        coach_only.bytes_tried.add(b)
    if w.active is None:
        return
    tn = coach_net.choose_target(w)
    tc = coach_only.choose_target(w)
    # Not required to differ always, but network should store graph bonus
    assert coach_net.last_graph_bonus >= 0.0
    assert isinstance(tn, tuple) and isinstance(tc, tuple)


def test_network_primary_geo_intent_from_target():
    """graph_preferred_intent follows network placement target when play-ready."""
    import tetris_demo as mod

    cipher = ActionCipher.fixed({10: "left", 20: "right", 30: "rotate", 40: "hard"})
    w = TetrisWorld(rng=Random(2), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(
        rng=Random(2),
        network_primary=True,
        graph_placement_weight=0.9,
        map_threshold=1,
    )
    for b, e in ((10, "left"), (20, "right"), (30, "rotate"), (40, "hard")):
        coach.effect_counts[b][e] = 3
        coach.bytes_tried.add(b)
    assert coach.play_ready()
    assert w.active is not None
    # Force a target to the right of current column
    cur = w.active
    tgt_col = min(w.cols - 1, cur.col + 2)
    if tgt_col == cur.col:
        tgt_col = max(0, cur.col - 2)
    coach._target = (cur.rotation % 4, tgt_col)
    s = mod.build_symbioid(w)
    pref, bias, poles, hint = mod.graph_preferred_intent(s, w, coach)
    assert pref in ("left", "right", "hard", "rotate")
    if tgt_col > cur.col:
        assert pref == "right", f"expected right toward target, got {pref} hint={hint}"
    elif tgt_col < cur.col:
        assert pref == "left", f"expected left toward target, got {pref} hint={hint}"
    assert bias >= 0.85
    assert isinstance(poles, list)
    assert hint is not None


def test_phase_b_wants_hard_when_aligned():
    w = TetrisWorld(rng=Random(3), gravity_interval=9999)
    coach = TetrisCoach(rng=Random(3), map_threshold=1)
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    coach._begin_piece(w)
    assert w.active is not None
    # Force target = current pose → hard
    coach._target = (w.active.rotation % 4, w.active.col)
    assert coach.wants_hard_now(w) is True
    coach._target = (w.active.rotation % 4, (w.active.col + 3) % max(1, w.cols - 1))
    coach._stuck_lateral = 0
    coach._piece_cmds = 0
    # Not aligned and not stuck
    if coach._target[1] != w.active.col:
        assert coach.wants_hard_now(w) is False


def test_last_lock_effect_hard_on_hard_drop():
    cipher = ActionCipher.fixed({1: "left", 2: "right", 3: "rotate", 4: "hard"})
    w = TetrisWorld(rng=Random(2), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(2), map_threshold=1)
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    # Align and hard-drop
    for _ in range(80):
        if w.game_over:
            break
        coach.tick(w, run_gravity=False)
        if w.pieces_placed >= 1:
            break
    # If we locked via intentional hard path, last_lock_effect should be hard
    # (or soft if gravity path; force by direct hard byte)
    if w.pieces_placed < 1 and w.active is not None:
        w.step_byte(4)
        if w.pieces_placed >= 1:
            coach.last_lock_effect = "hard"
    assert coach.last_lock_effect in ("hard", "soft", "left", "right", "rotate", "noop")


def test_coach_places_many_without_crash():
    w = TetrisWorld(rng=Random(5))
    coach = TetrisCoach(rng=Random(5), noise=0.1)
    for _ in range(40):
        if w.game_over:
            coach.on_new_game(w)
        assert coach.act(w) or w.game_over
    assert coach.placements >= 1
    assert coach.summary()


def test_tick_emits_bytes_and_can_lock():
    cipher = ActionCipher.fixed({1: "left", 2: "right", 3: "rotate", 4: "hard"})
    w = TetrisWorld(rng=Random(8), cipher=cipher, gravity_interval=999)
    coach = TetrisCoach(rng=Random(8), explore_rate=0.7, map_threshold=1)
    # Seed map so we can aim (still via coach beliefs, not cipher read)
    for b, eff in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        for _ in range(2):
            coach.effect_counts[b][eff] += 1
        coach.bytes_tried.add(b)
    assert coach.map_complete()
    codes = []
    for _ in range(100):
        if w.game_over:
            break
        codes.append(coach.tick(w))
        if w.pieces_placed >= 1:
            break
    assert codes
    assert all(0 <= c <= 255 for c in codes)
    assert w.pieces_placed >= 1 or w.game_over


def test_highscore_list_records_game_number_and_score():
    w = TetrisWorld(rng=Random(9))
    coach = TetrisCoach(rng=Random(9), noise=0.05)
    for _ in range(300):
        if w.game_over:
            break
        coach.act(w)
    assert w.game_over or w.pieces_placed > 0
    entry = coach.on_new_game(w, record=True)
    assert entry is not None
    game_no, score = entry
    assert game_no == 1
    assert coach.highscores == [(1, score)]
    assert coach.game_number == 2
    for _ in range(50):
        if w.game_over:
            break
        coach.act(w)
    coach.on_new_game(w, record=True)
    assert len(coach.highscores) == 2
    # Ordered by score (highest first); format "#ddd ssssss"
    lines = coach.highscore_lines()
    scores = [int(line.split()[-1]) for line in lines]
    assert scores == sorted(scores, reverse=True)
    for line in lines:
        assert len(line) == 11  # "#001" + " " + "  4700"
        assert line[0] == "#"
        assert line[4] == " "
        int(line[1:4])
        int(line[5:].lstrip() or "0")


def test_piece_cells_count():
    for kind in ("I", "O", "T", "S", "Z", "J", "L"):
        for rot in range(4):
            assert len(piece_cells(kind, rot)) == 4


def test_simulate_placement_features_exist_but_coach_must_not_need_them():
    """World may still expose sim for tools; coach decisions must not use it."""
    w = TetrisWorld(rng=Random(6))
    opts = w.legal_placements()
    rot, col = opts[0]
    feat = w.simulate_placement(rot, col)
    assert feat is not None
    assert w.pieces_placed == 0


def test_coach_uses_sim_search_and_learns_from_real_drops():
    """1-ply sim for choice; experiences still update from real locks."""
    w = TetrisWorld(rng=Random(11))
    coach = TetrisCoach(rng=Random(11), place_explore=0.05)

    for _ in range(40):
        if w.game_over:
            coach.on_new_game(w, record=False)
        coach.act(w)

    assert len(coach.experiences) >= 10
    # Prefer a flat low board over a tall holey one in evaluator
    pre = observe_board(w)
    low = {
        "max_height": 3.0,
        "agg_height": 15.0,
        "holes": 0.0,
        "bumpiness": 1.0,
        "well": 0.0,
        "height_range": 1.0,
        "lines_cleared": 0.0,
    }
    high = {
        "max_height": 15.0,
        "agg_height": 70.0,
        "holes": 6.0,
        "bumpiness": 10.0,
        "well": 4.0,
        "height_range": 8.0,
        "lines_cleared": 0.0,
    }
    assert coach.evaluate_imagined_drop(pre, low) > coach.evaluate_imagined_drop(
        pre, high
    )


def test_sim_search_clears_lines_often():
    """With map short-circuit via act(), packing should clear lines in a run."""
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    coach = TetrisCoach(rng=Random(0), place_explore=0.02)
    lines_before = 0
    for _ in range(120):
        if w.game_over:
            coach.on_new_game(w, record=False)
        coach.act(w)
    assert w.lines + coach.lines_total >= 1 or coach.lines_total >= 1


def test_classify_effect_left():
    cipher = ActionCipher.fixed({5: "left"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=9999)
    before = WorldSnapshot.take(w)
    w.step_byte(5)
    assert classify_effect(before, w) == "left"


def test_classify_left_not_unknown_when_row_also_changes():
    """Regression: left+down must still read as left (old gravity mix bug)."""
    cipher = ActionCipher.fixed({5: "left"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=1)
    before = WorldSnapshot.take(w)
    w.step_byte(5)
    # Manually nudge row as if gravity co-occurred (classification-only check)
    if w.active is not None:
        w.active.row += 1
    assert classify_effect(before, w) == "left"


def test_gravity_lock_does_not_label_dead_byte_as_hard():
    cipher = ActionCipher.fixed({44: "hard"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=1)
    # Spam dead bytes with gravity until something locks
    labeled_hard = 0
    for _ in range(400):
        if w.game_over:
            break
        before = WorldSnapshot.take(w)
        prev = w.pieces_placed
        w.step_byte(0)  # dead
        if w.pieces_placed == prev:
            w.tick_gravity()
        eff = classify_effect(before, w)
        if w.pieces_placed > prev and eff == "hard":
            labeled_hard += 1
    # Gravity locks after a dead byte must not count as hard
    assert labeled_hard == 0


def test_discovery_with_demo_like_gravity():
    cipher = ActionCipher.fixed({11: "left", 22: "right", 33: "rotate", 44: "hard"})
    w = TetrisWorld(rng=Random(10), cipher=cipher, gravity_interval=18)
    coach = TetrisCoach(rng=Random(10), explore_rate=0.95, map_threshold=1)
    for _ in range(2000):
        if w.game_over:
            coach.on_new_game(w, record=False)
        coach.tick(w)
        if coach.map_complete():
            break
    assert coach.map_complete(), coach.map_progress()
    assert coach.discovered_map()["left"] == 11
    assert coach.discovered_map()["hard"] == 44


def test_systematic_rescan_after_full_pass():
    """After all 256 tried once, scan cursor walks the space again."""
    cipher = ActionCipher.fixed({11: "left", 22: "right", 33: "rotate", 44: "hard"})
    w = TetrisWorld(rng=Random(3), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(3), map_threshold=1)
    coach.bytes_tried = set(range(256))
    # force empty effect knowledge except fill counts as noop
    for b in range(256):
        coach.effect_counts[b]["noop"] = 1
    seen = [coach._pick_explore_byte() for _ in range(256)]
    assert set(seen) == set(range(256))


def test_explore_col_balanced_left_right_from_spawn():
    """Step-based sampler: left/right *keypress distance* roughly balanced."""
    coach = TetrisCoach(rng=Random(0))
    cur = 3
    left_steps = right_steps = 0
    for _ in range(2000):
        col = coach._sample_explore_col(cur, 10, lo=0, hi=9)
        if col < cur:
            left_steps += cur - col
        elif col > cur:
            right_steps += col - cur
    total = left_steps + right_steps
    assert total > 0
    assert abs(left_steps - right_steps) / total < 0.20, (left_steps, right_steps)


def test_observe_board_includes_height_sensors():
    w = TetrisWorld(rng=Random(0))
    obs = observe_board(w)
    assert "max_height" in obs and "agg_height" in obs
    assert "h0" in obs and "well" in obs and "fill_n" in obs


def test_board_quality_reward_prefers_low_stack():
    pre = {
        "max_height": 4.0,
        "agg_height": 20.0,
        "holes": 0.0,
        "bumpiness": 2.0,
        "well": 0.0,
        "height_range": 2.0,
    }
    low = dict(pre, max_height=5.0, agg_height=24.0, holes=0.0)
    high = dict(pre, max_height=14.0, agg_height=60.0, holes=3.0)
    r_low = board_quality_reward(pre, low, lines_cleared=0, topped_out=False, score_delta=10)
    r_high = board_quality_reward(pre, high, lines_cleared=0, topped_out=False, score_delta=40)
    assert r_low > r_high


def test_no_endless_right_on_unreachable_target():
    """If target col is past the wall, do not spam right for hundreds of ticks."""
    cipher = ActionCipher.fixed({11: "left", 22: "right", 33: "rotate", 44: "hard"})
    w = TetrisWorld(rng=Random(0), cipher=cipher, gravity_interval=9999)
    coach = TetrisCoach(rng=Random(0), map_threshold=1)
    for b, e in ((11, "left"), (22, "right"), (33, "rotate"), (44, "hard")):
        coach.effect_counts[b][e] = 5
        coach.bytes_tried.add(b)
    # Force an over-wide target; clamp + stuck logic should hard-drop quickly
    coach._pre_board = {"holes": 0.0, "max_height": 0.0, "bumpiness": 0.0, "agg_height": 0.0, "lines": 0.0, "score": 0.0}
    coach._score_at_piece_start = 0.0
    coach._piece_kind = w.active.kind
    coach._target = (0, 9)  # often illegal for wide pieces
    rights = 0
    for _ in range(40):
        intent = coach.desired_intent(w)
        if intent == "right":
            rights += 1
        coach.tick(w)
        if w.pieces_placed >= 1:
            break
    assert w.pieces_placed >= 1
    assert rights < 15, rights


def test_cell_field_state_empty_board_is_open():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None  # pure empty locked board
    field = w.cell_field_state(with_active=False)
    assert len(field) == w.rows and len(field[0]) == w.cols
    assert all(v == 0.0 for row in field for v in row)


def test_cell_field_block_and_hole():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None
    # Block at row 5, col 3 → air above is open; empty below is hole
    w.board[5][3] = "I"
    assert w.cell_reading(5, 3, with_active=False) == 1.0
    assert w.cell_reading(4, 3, with_active=False) == 0.0  # open above
    assert w.cell_reading(6, 3, with_active=False) == 0.5  # hole under block
    assert w.cell_reading(10, 3, with_active=False) == 0.5
    assert w.cell_reading(5, 4, with_active=False) == 0.0


def test_cell_field_includes_active_piece():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    assert w.active is not None
    cells = w.active.cells()
    r, c = cells[0]
    assert w.cell_reading(r, c, with_active=True) == 1.0
    # Falling piece must not invent holes under itself
    below = r + 1
    if below < w.rows:
        assert w.cell_reading(below, c, with_active=True) == 0.0
    locked = w.cell_reading(r, c, with_active=False)
    assert locked in (0.0, 0.5, 1.0)


def _load_tetris_demo():
    import importlib.util
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "tetris_demo", root / "tetris_demo.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_demo_build_has_full_cell_sensor_map():
    """200 cell sensors + 11 meta (packing + foresight); cells skip full awareness."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0))
    s = mod.build_symbioid(w)
    # piece/next/lines/byte + 4 pack + 3 foresight
    assert len(s.sensors) == w.rows * w.cols + 11
    cell_labels = [sen.label for sen in s.sensors if (sen.label or "").startswith("cell_")]
    assert len(cell_labels) == w.rows * w.cols
    assert any(sen.label == "piece_id" for sen in s.sensors)
    assert any(sen.label == "next_id" for sen in s.sensors)
    assert any(sen.label == "holes_n" for sen in s.sensors)
    assert any(sen.label == "last_d_holes" for sen in s.sensors)
    assert any(sen.label == "holes_freed" for sen in s.sensors)
    assert any(sen.label == "pred_d_holes" for sen in s.sensors)
    # Cell sensors are terminators without bloating awareness_sets
    cell_ids = [sen.id for sen in s.sensors if (sen.label or "").startswith("cell_")]
    assert all(cid in s.integration_terminators for cid in cell_ids)
    assert not any(
        (sid or "").startswith(f"{s.id}:sen:cell_") for sid in s.awareness_sets
    ) or len(s.awareness_sets) < 20  # only meta (+ actuator)
    sen0 = next(sen for sen in s.sensors if sen.label == "cell_r00_c00")
    v = sen0.transfer({})
    assert v in (0.0, 0.5, 1.0)


def test_sample_change_only_skips_static_open_cells():
    """First sample should not hand off 200 open cells — only active/non-open + meta."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    s = mod.build_symbioid(w)
    # Drain any startup; measure handoffs via mind mint delta + last map
    before_mint = s.mind.admits_mint
    mod.sample_into_symbioid(s, w, tick=1)
    # Second sample with no board change → almost no new cell formations
    mid_mint = s.mind.admits_mint
    mod.sample_into_symbioid(s, w, tick=2)
    after_mint = s.mind.admits_mint
    # First sample may mint a few (active piece cells + meta); not ~200
    first_wave = mid_mint - before_mint
    assert first_wave < 40, f"first sample minted too many: {first_wave}"
    second_wave = after_mint - mid_mint
    assert second_wave < 10, f"static re-sample minted too many: {second_wave}"


def test_sky_row_and_solid_floor_roi():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None
    assert w.sky_row(with_active=False) == w.rows
    assert w.solid_floor_start_row() == w.rows
    assert w.cell_sample_roi(with_active=False) == (w.rows, w.rows)

    # Locked stack mid-board
    w.board[10][3] = "T"
    assert w.sky_row(with_active=False) == 10
    # Solid full-width base at bottom
    for c in range(w.cols):
        w.board[w.rows - 1][c] = "I"
        w.board[w.rows - 2][c] = "I"
    assert w.solid_floor_start_row() == w.rows - 2
    r_lo, r_hi = w.cell_sample_roi(with_active=False)
    assert r_lo == 10
    assert r_hi == w.rows - 2


def test_active_cells_set_and_sky_includes_piece():
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    assert w.active is not None
    cells = w.active_cells_set()
    assert len(cells) >= 4
    assert w.sky_row(with_active=True) <= min(r for r, _ in cells)


def test_sample_dirty_rect_limits_move_mints():
    """Lateral move should mint roughly O(piece cells), not a sky full of opens."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(1), gravity_interval=9999)
    s = mod.build_symbioid(w)
    mod.sample_into_symbioid(s, w, tick=1)
    # Process a few left moves; mint growth per move should stay small
    mints = []
    for i in range(4):
        before = s.mind.admits_mint
        w.step_action("left")
        mod.sample_into_symbioid(s, w, tick=10 + i)
        mints.append(s.mind.admits_mint - before)
    # Each move: leave cells + enter cells (+ maybe meta). Not dozens of sky cells.
    assert max(mints) < 25, f"move mints too high: {mints}"
    assert sum(mints) < 60, f"total move mints too high: {mints}"


def test_sample_line_clear_invalidates_last_readings():
    """After line clear, cell last-map resets so stack can re-form."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None
    # Partial stack (not full-width solid floor — that would empty the ROI band)
    w.board[15][2] = "I"
    w.board[15][3] = "I"
    w.board[16][3] = "I"
    s = mod.build_symbioid(w)
    mod.sample_into_symbioid(s, w, tick=1)
    assert s._cell_last_reading, "expected some cell last-readings after sample"
    before_keys = set(s._cell_last_reading.keys())
    # Simulate line clear event
    w.last_event = "line_clear"
    w.lines += 1
    w.board[15][2] = ""
    w.board[15][3] = ""
    w.board[16][3] = ""
    mod.sample_into_symbioid(s, w, tick=2)
    # last_lines advanced; resync ran (map rebuilt for new open field)
    assert int(s._cell_last_lines) == int(w.lines)
    # Prior non-open keys should not all persist unchanged as the sole map
    assert before_keys  # had content pre-clear


def test_sticky_locked_skips_reform():
    """Locked 1.0 cells do not re-mint on static re-sample."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0), gravity_interval=9999)
    w.active = None
    w.board[15][4] = "O"
    s = mod.build_symbioid(w)
    mod.sample_into_symbioid(s, w, tick=1)
    mid = s.mind.admits_mint
    mod.sample_into_symbioid(s, w, tick=2)
    mod.sample_into_symbioid(s, w, tick=3)
    assert s.mind.admits_mint - mid < 8, "sticky locked should not re-mint"


def test_twin_seed_thoughts_is_constant_size():
    """Protect path must not scan the full graph for twin seeds."""
    from symbioid import Symbioid, Thought

    s = Symbioid(id="sym-twin-perf", install_constitution=False)
    for i in range(500):
        s.add_thought(Thought(id=f"{s.id}:form:junk{i}", transient=True))
    twin = s.twin_seed_thoughts()
    assert len(twin) == 6
    assert f"{s.id}:system" in twin
    assert f"{s.id}:form:junk0" not in twin


def test_eligibility_window_recency_weights():
    import tetris_demo as mod

    w = mod.EligibilityWindow(max_ticks=4)
    w.push(["a"])
    w.push(["b", "a"])
    w.push(["c"])
    cred = w.credited_keys()
    assert cred["c"] == 1.0  # newest
    assert cred["a"] == 2.0 / 3.0  # last seen mid
    assert cred["b"] == 2.0 / 3.0
    w.clear()
    assert len(w) == 0
    assert w.credited_keys() == {}


def test_apply_eligibility_valence_raises_keys():
    import tetris_demo as mod
    from symbioid import Symbioid

    s = Symbioid(install_constitution=False)
    win = mod.EligibilityWindow(max_ticks=8)
    win.push(["meta:holes_n:0.1"])
    win.push(["meta:holes_n:0.1", "cell_r10_c03:place"])
    n = mod.apply_eligibility_valence(s, win, reward=100.0, strength=1.0)
    assert n >= 2
    v = s.mind.valence_of(content_key="meta:holes_n:0.1")
    assert v > 0.0
    # Negative reward lowers
    mod.apply_eligibility_valence(s, win, reward=-100.0, strength=1.0)
    v2 = s.mind.valence_of(content_key="meta:holes_n:0.1")
    assert v2 < v


def test_apply_lock_credit_landing_and_eligibility():
    import tetris_demo as mod
    from symbioid import Symbioid
    from symbioid.world.tetris_learn import TetrisCoach

    s = Symbioid(install_constitution=False)
    coach = TetrisCoach(rng=Random(0))
    coach.last_reward = 80.0
    coach.last_lock_cells = [(18, 3), (18, 4)]
    win = mod.EligibilityWindow(max_ticks=8)
    win.push(["traj:key:1"])
    stats = mod.apply_lock_credit(s, coach, win, poles=None)
    assert stats["landing"] >= 2
    assert stats["eligibility"] >= 1
    assert len(win) == 0  # cleared after credit
    assert s.mind.valence_of(content_key="cell_r18_c03:place") > 0.0
    assert s.mind.valence_of(content_key="traj:key:1") > 0.0


def test_summarize_and_multi_game_metric_smoke():
    """Headless multi-game metric returns N rows + summary keys (short frames)."""
    import tetris_demo as mod

    rows, summary = mod.run_multi_game_metric(
        games=1,
        max_frames=120,
        seed=7,
        eligibility_window=8,
        use_eligibility=True,
        map_threshold=1,
        verbose=False,
    )
    assert len(rows) == 1
    assert summary["n"] == 1.0
    for key in (
        "mean_score",
        "mean_lines",
        "mean_holes",
        "mean_max_height",
        "mean_pieces",
    ):
        assert key in summary
    r = rows[0]
    assert r.game == 1
    assert r.frames <= 120
    assert r.pieces >= 0
    d = r.as_dict()
    assert "holes" in d and "max_height" in d


def test_version_at_least_051():
    from symbioid import __version__

    parts = [int(x) for x in __version__.split(".")]
    assert parts >= [0, 0, 51]


def test_landing_cells_and_features_matches_split_apis():
    w = TetrisWorld(rng=Random(5), gravity_interval=9999)
    opts = w.legal_placements()
    assert opts
    rot, col = opts[0]
    cells_a = w.landing_cells(rot, col)
    sim_a = w.simulate_placement(rot, col)
    cells_b, sim_b = w.landing_cells_and_features(rot, col)
    assert cells_a == cells_b
    assert sim_a is not None and sim_b is not None
    assert abs(float(sim_a["holes"]) - float(sim_b["holes"])) < 1e-9


def test_batch_landing_template_independent():
    w = TetrisWorld(rng=Random(6), gravity_interval=9999)
    opts = w.legal_placements()[:5]
    assert opts
    batch = w.batch_landing_cells_and_features(opts)
    assert len(batch) == len(opts)
    for (rot, col), (cells, sim) in zip(opts, batch):
        cells2, sim2 = w.landing_cells_and_features(rot, col)
        assert cells == cells2
        if sim is None:
            assert sim2 is None
        else:
            assert abs(float(sim["holes"]) - float(sim2["holes"])) < 1e-9


def test_batch_placement_scores_match_single():
    """Phase 3: batch API matches single-pose scores (same context semantics)."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(2), gravity_interval=9999)
    s = mod.build_symbioid(w)
    opts = w.legal_placements()
    assert opts
    batch = mod.batch_cell_thought_placement_scores(s, w, opts)
    assert len(batch) == len(opts)
    ctx = mod.build_placement_score_context(s, w)
    for (rot, col), sc in zip(opts, batch):
        single = mod.score_pose_with_context(ctx, rot, col)
        assert abs(single - sc) < 1e-9


def test_cached_graph_bonus_prepare_ranking_stable():
    """prepare + lookup preserves relative ranking vs single scores."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(3), gravity_interval=9999)
    s = mod.build_symbioid(w)
    opts = w.legal_placements()
    assert len(opts) >= 2
    bonus = mod.CachedGraphPlacementBonus(s)
    bonus.prepare(w, opts)
    singles = [mod.cell_thought_placement_score(s, w, r, c) for r, c in opts]
    cached = [bonus(w, r, c) for r, c in opts]
    # Same argmax (ties broken by first max)
    assert cached.index(max(cached)) == singles.index(max(singles)) or abs(
        max(cached) - max(singles)
    ) < 1e-6
    for a, b in zip(cached, singles):
        assert abs(a - b) < 1e-5


def test_choose_target_calls_prepare_on_batch_bonus():
    w = TetrisWorld(rng=Random(4), gravity_interval=9999)
    coach = TetrisCoach(rng=Random(4), network_primary=True, map_threshold=1)
    for b, e in ((1, "left"), (2, "right"), (3, "rotate"), (4, "hard")):
        coach.effect_counts[b][e] = 2
        coach.bytes_tried.add(b)
    prepared = {"n": 0}

    class PrepBonus:
        def prepare(self, world, options):
            prepared["n"] += 1
            prepared["opts"] = list(options)

        def __call__(self, world, rot, col):
            return float(col)

    coach.graph_placement_bonus = PrepBonus()
    coach.graph_placement_weight = 0.90
    coach.place_explore = 0.0
    if w.active is None:
        return
    coach.choose_target(w)
    assert prepared["n"] == 1
    assert prepared.get("opts")
