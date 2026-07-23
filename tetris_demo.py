#!/usr/bin/env python3
"""
Tetris + Symbioid with a **secret byte control map**.

The agent may emit any byte 0..255 each tick. Only a few secret bytes actually
map to left / right / rotate / hard — the rest are dead. The coach never sees
the cipher table; it must discover which bytes do what by watching the world.

Drop quality is also learned from **real locks only** (no simulate_placement
oracle for scoring). Highscores track (game #, score) across top-outs.

Quit: Esc.  R restarts after top-out (also auto-restarts).

  PYTHONPATH=. .venv/bin/python tetris_demo.py
"""

from __future__ import annotations

import sys

try:
    import pygame
except ImportError:
    print("pygame required:  .venv/bin/pip install pygame", file=sys.stderr)
    sys.exit(1)

from symbioid import Symbioid, format_six_set_line
from symbioid.world.tetris import (
    ActionCipher,
    PIECE_COLORS,
    PIECE_NAMES,
    TetrisWorld,
    piece_cells,
)
from symbioid.world.tetris_learn import TetrisCoach


CELL = 28
COLS, ROWS = 10, 20
BOARD_W, BOARD_H = COLS * CELL, ROWS * CELL
SIDE = 280
W, H = BOARD_W + SIDE + 40, BOARD_H + 40
FPS = 30
CMD_EVERY = 2
GRAVITY_INTERVAL = 18
RESTART_DELAY_FRAMES = FPS * 2


def build_symbioid(world: TetrisWorld) -> Symbioid:
    s = Symbioid(label="tetris-byte-learner")
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
        sen = s.add_sensor(label=label)
        sen.transfer = transfer

    # Single actuator: raw control byte as 0..1 (×255 inside coach path)
    out = s.add_actuator(label="byte")
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


def draw(
    screen: pygame.Surface,
    world: TetrisWorld,
    coach: TetrisCoach,
    font: pygame.font.Font,
    font_sm: pygame.font.Font,
) -> None:
    screen.fill((12, 14, 28))
    ox, oy = 20, 20

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

    sx = ox + BOARD_W + 24
    screen.blit(font.render("Symbioid Tetris", True, (200, 210, 230)), (sx, oy))
    screen.blit(
        font_sm.render(
            f"game #{coach.game_number}   score {world.score}",
            True,
            (180, 190, 210),
        ),
        (sx, oy + 30),
    )
    screen.blit(
        font_sm.render(
            f"lines {world.lines}  pieces {world.pieces_placed}",
            True,
            (180, 190, 210),
        ),
        (sx, oy + 48),
    )

    # Byte stream (what the agent emits — not the secret cipher)
    screen.blit(
        font_sm.render(
            f"byte 0x{coach.last_byte:02X} ({coach.last_byte:3d})  "
            f"seen→{coach.last_effect}",
            True,
            (160, 220, 180),
        ),
        (sx, oy + 72),
    )
    screen.blit(
        font_sm.render(f"intent {coach.last_intent}", True, (140, 180, 160)),
        (sx, oy + 90),
    )
    scan = getattr(coach, "_scan_passes", 0)
    status = (
        "MAP OK"
        if coach.map_complete()
        else f"scan#{scan} miss:{','.join(coach.missing_effects()) or '?'}"
    )
    screen.blit(
        font_sm.render(
            f"tried {len(coach.bytes_tried)}/256  {status}",
            True,
            (220, 200, 120) if coach.map_complete() else (200, 140, 100),
        ),
        (sx, oy + 108),
    )
    # Discovered beliefs only (never ground-truth cipher)
    screen.blit(font_sm.render("learned map:", True, (140, 150, 170)), (sx, oy + 130))
    y = oy + 148
    for line in _wrap(coach.map_progress(), 34):
        screen.blit(font_sm.render(line, True, (180, 190, 210)), (sx, y))
        y += 16
    screen.blit(
        font_sm.render(
            f"drop model: {coach.drop_model_summary()}  R={coach.last_reward:.0f}",
            True,
            (160, 180, 200),
        ),
        (sx, y + 4),
    )
    y += 22

    # Next
    screen.blit(font_sm.render("next", True, (140, 150, 170)), (sx, y + 8))
    nc = PIECE_COLORS.get(world.next_kind, (180, 180, 180))
    for dr, dc in piece_cells(world.next_kind, 0):
        pygame.draw.rect(
            screen,
            nc,
            (sx + dc * 18, y + 28 + dr * 18, 16, 16),
            border_radius=2,
        )
    y += 100

    # Highscores
    screen.blit(
        font_sm.render("highscores  #game  score", True, (200, 190, 120)), (sx, y)
    )
    y += 18
    if not coach.highscores:
        screen.blit(font_sm.render("(finish a game…)", True, (100, 110, 130)), (sx, y))
        y += 16
    else:
        best = coach.best_score()
        for line in coach.highscore_lines(limit=10):
            try:
                sc = int(line.split()[-1])
                color = (240, 210, 100) if sc == best else (160, 170, 190)
            except (ValueError, IndexError):
                color = (160, 170, 190)
            screen.blit(font_sm.render(line, True, color), (sx, y))
            y += 15
        y += 2
        screen.blit(font_sm.render(f"best  {best}", True, (240, 210, 100)), (sx, y))
        y += 18

    # if world.game_over:
        # overlay = font.render("TOP OUT — R restart", True, (240, 120, 120))
        # screen.blit(overlay, (ox + 16, oy + BOARD_H // 2 - 10))

    screen.blit(
        font_sm.render(
            "Any byte 0..255; only a few secret ones move. Discover map, then play. Esc.",
            True,
            (120, 130, 160),
        ),
        (20, H - 24),
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


def main() -> None:
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

    twin = s.twin_seed_thoughts()
    if twin:
        print(format_six_set_line("twin", twin, index=0), flush=True)
    print(
        "Tetris + Symbioid: SECRET byte map "
        f"({len(cipher)} live / 256). Agent must discover it.",
        flush=True,
    )
    print("(Cipher hidden from learner; not printed here.)", flush=True)

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
                        print(
                            f"[restart] {entry} | map {coach.map_progress()}",
                            flush=True,
                        )

            if frame % sample_every == 0:
                sample_into_symbioid(s, world, tick=frame)

            if not world.game_over and frame % CMD_EVERY == 0:
                prev_event = world.last_event
                code = coach.tick(world)
                s.actuators[0].output = code / 255.0
                if coach.map_complete() and not was_mapped:
                    was_mapped = True
                    print(
                        f"[map complete] {coach.map_progress()} "
                        f"after {len(coach.bytes_tried)} bytes tried",
                        flush=True,
                    )
                if world.last_event != prev_event and world.last_event in (
                    "line_clear",
                    "top_out",
                    "lock",
                ):
                    print(
                        f"[{world.last_event}] g#{coach.game_number} "
                        f"score={world.score} byte=0x{code:02X} "
                        f"seen={coach.last_effect} | {coach.map_progress()}",
                        flush=True,
                    )

            if world.game_over:
                if game_over_at is None:
                    game_over_at = frame
                    print(
                        f"[top_out] game #{coach.game_number} "
                        f"final_score={world.score} "
                        f"tried={len(coach.bytes_tried)}/256",
                        flush=True,
                    )
                elif frame - game_over_at >= RESTART_DELAY_FRAMES:
                    entry = coach.on_new_game(world, record=True)
                    game_over_at = None
                    print(
                        f"[auto-restart] {entry} best={coach.best_score()} "
                        f"highscores={coach.highscores[-6:]}",
                        flush=True,
                    )

            if frame > 0 and frame % log_every == 0 and not world.game_over:
                print(
                    f"t={frame} g#{coach.game_number} score={world.score} "
                    f"0x{coach.last_byte:02X}→{coach.last_effect} "
                    f"intent={coach.last_intent} tried={len(coach.bytes_tried)} "
                    f"| {coach.map_progress()}",
                    flush=True,
                )

            draw(screen, world, coach, font, font_sm)
            pygame.display.flip()
            clock.tick(FPS)
            frame += 1
    finally:
        s.stop_processes()
        pygame.quit()
        if not world.game_over and world.pieces_placed > 0:
            coach.record_game_score(world)
        print(
            f"\nstopped: game=#{coach.game_number} score={world.score}\n"
            f"  {coach.summary()}\n"
            f"  highscores: {coach.highscores}\n"
            f"  formations={len(s.innerface.completed_formations)} "
            f"beliefs={len(s.outerface.active_belief_ids)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
