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
    """200 cell sensors + 4 meta; cells skip full awareness six-sets."""
    mod = _load_tetris_demo()
    w = TetrisWorld(rng=Random(0))
    s = mod.build_symbioid(w)
    assert len(s.sensors) == w.rows * w.cols + 4
    cell_labels = [sen.label for sen in s.sensors if (sen.label or "").startswith("cell_")]
    assert len(cell_labels) == w.rows * w.cols
    assert any(sen.label == "piece_id" for sen in s.sensors)
    assert any(sen.label == "next_id" for sen in s.sensors)
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
