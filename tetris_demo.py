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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from symbioid.world.tetris_learn import TetrisCoach

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
CMD_EVERY = 2
GRAVITY_INTERVAL = 18
# Pause after top-out so Innerface formations can catch up before next game
RESTART_DELAY_FRAMES = FPS * 1


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
    return p.parse_args(argv)


def thought_count(s: Symbioid) -> int:
    return len(s.thoughts)


def thought_counts_active_inactive(s: Symbioid) -> tuple[int, int]:
    """
    Active = Thoughts currently in an Innerface *active* six-set.
    Inactive = other Thoughts still on the host graph (seeds, laws, awareness,
    superseded scaffolding not yet pruned, etc.).
    """
    active_tids: set[str] = set()
    inner = s.innerface
    with inner._local_lock:
        for sid in inner.active_ids:
            store = (
                inner.completed_formations.get(sid)
                or inner.completed_syncs.get(sid)
                or inner.completed_integrates.get(sid)
            )
            if store:
                active_tids.update(store.keys())
    with s.graph_lock:
        host_ids = set(s.thoughts.keys())
    n_active = len(active_tids & host_ids)
    n_inactive = max(0, len(host_ids) - n_active)
    return n_active, n_inactive


def build_symbioid(world: TetrisWorld) -> Symbioid:
    s = Symbioid(id=HOST_ID, label="tetris-byte-learner")
    s.interface.continuous_inputs = False
    s.outerface.wait_for_feedback = False

    def piece_id_n(_w: dict, wo: TetrisWorld = world) -> float:
        if wo.active is None:
            return 0.0
        return PIECE_NAMES.index(wo.active.kind) / 6.0

    def col_heights_mean(_w: dict, wo: TetrisWorld = world) -> float:
        hs = wo.column_heights()
        return (sum(hs) / len(hs) / wo.rows) if hs else 0.0

    def height_range_n(_w: dict, wo: TetrisWorld = world) -> float:
        hs = wo.column_heights()
        if not hs:
            return 0.0
        return (max(hs) - min(hs)) / wo.rows

    for label, transfer in (
        ("max_height", lambda w, wo=world: wo.max_height() / wo.rows),
        ("agg_height", lambda w, wo=world: wo.aggregate_height() / (wo.rows * wo.cols)),
        ("mean_height", col_heights_mean),
        ("height_range", height_range_n),
        ("holes", lambda w, wo=world: min(1.0, wo.hole_count() / 20.0)),
        ("bumpiness", lambda w, wo=world: min(1.0, wo.bumpiness() / 30.0)),
        ("lines", lambda w, wo=world: min(1.0, wo.lines / 50.0)),
        ("piece_id", piece_id_n),
        ("last_byte", lambda w, wo=world: wo.last_byte / 255.0),
    ):
        # Stable sensor ids → Observation content keys survive reloads
        sen = s.add_sensor(Sensor(id=f"{HOST_ID}:sen:{label}", label=label))
        sen.transfer = transfer

    # Single actuator: raw control byte as 0..1 (×255 inside coach path)
    from symbioid import Actuator

    out = s.add_actuator(Actuator(id=f"{HOST_ID}:act:byte", label="byte"))
    out.output = 0.0
    out.output_step = 1.0 / 255.0
    return s


def sample_into_symbioid(s: Symbioid, world: TetrisWorld, tick: int) -> None:
    w = world.sensor_world()
    w["byte"] = float(s.actuators[0].output)
    handoffs = []
    for sen in s.sensors:
        sense = sen.sample(tick=tick, world=w)
        if sense is None:
            continue
        h = s.interface.start_formation_for_sensor(sen, force=True, sense=sense)
        if h is not None:
            handoffs.append(h)
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
    args = parse_args(argv)
    set_console_emit(args.verbose)
    log = print if args.verbose else (lambda *a, **k: None)

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
    coach = TetrisCoach()
    s = build_symbioid(world)

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

    s.start_processes()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Symbioid Tetris — secret byte controls")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 20)
    font_sm = pygame.font.SysFont("DejaVu Sans", 14)
    frame = 0
    sample_every = 4
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

    def _state_poles():
        with s.innerface._local_lock:
            return list(s.innerface._last_obs_by_sensor.values())

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
                        entry = coach.on_new_game(world, record=True)
                        game_over_at = None
                        last_pieces_for_plot = world.pieces_placed
                        log(
                            f"[restart] {entry} | map {coach.map_progress()}",
                            flush=True,
                        )

            if frame % sample_every == 0:
                sample_into_symbioid(s, world, tick=frame)
            # Continuous decay / fire / spread (Thought-as-neuron)
            if s.mind.dynamics_enabled:
                s.pulse_tick()

            if not world.game_over and frame % CMD_EVERY == 0:
                prev_event = world.last_event
                prev_pieces = world.pieces_placed
                # Graph recommend from minted state→action associations
                poles = _state_poles()
                rec = s.mind.recommend_action(poles, domain="tetris")
                preferred = None
                last_graph_hint = None
                if rec is not None and rec.token in VALID_ACTIONS:
                    preferred = rec.token
                    last_graph_hint = f"{rec.token}@{rec.score:.2f}"
                last_cmd_poles = poles
                code = coach.tick(world, preferred_intent=preferred)
                last_cmd_intent = coach.last_intent
                if preferred and coach.last_intent == preferred:
                    last_graph_hint = f"USE {preferred}"
                s.actuators[0].output = code / 255.0
                # One sample per game turn (piece lock)
                if world.pieces_placed > prev_pieces:
                    # Write outcome into minted graph (state poles ↔ action intent)
                    intent = last_cmd_intent
                    if intent in VALID_ACTIONS:
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
                    entry = coach.on_new_game(world, record=True)
                    game_over_at = None
                    last_pieces_for_plot = world.pieces_placed
                    log(
                        f"[auto-restart] {entry} best={coach.best_score()} "
                        f"highscores={coach.highscores[-6:]}",
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
