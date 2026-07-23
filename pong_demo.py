#!/usr/bin/env python3
"""
Pong + Symbioid: both paddles **learn** to intercept the ball.

Sensors → ball / tracking error (Symbioid faces still form Thoughts & Beliefs).
Actuators ``left`` / ``right`` ← PaddleLearner policies (intercept + online updates).

Learning:
  - Predict intercept Y (with wall bounces) + learnable bias / vy_trust / gain
  - Exploration noise decays with hits
  - Hits and misses update parameters so play improves over time

Quit: Esc or close window.

  PYTHONPATH=. .venv/bin/python pong_demo.py
"""

from __future__ import annotations

import sys

try:
    import pygame
except ImportError:
    print("pygame required:  .venv/bin/pip install pygame", file=sys.stderr)
    sys.exit(1)

from symbioid import Symbioid, format_six_set_line
from symbioid.world.paddle_learn import DualPaddleCoach
from symbioid.world.pong import PongWorld


W, H = 800, 480
FPS = 60


def ny(y: float) -> int:
    return int((1.0 - y) * 0.5 * (H - 40) + 20)


def nx(x: float) -> int:
    return int((x + 1.0) * 0.5 * (W - 40) + 20)


def build_symbioid(world: PongWorld) -> Symbioid:
    s = Symbioid(label="pong-learner")
    s.interface.continuous_inputs = False
    s.outerface.wait_for_feedback = False

    ball = s.add_sensor(label="ball_y")
    bvy = s.add_sensor(label="ball_vy")
    lerr = s.add_sensor(label="left_err")
    rerr = s.add_sensor(label="right_err")
    ball.transfer = lambda w, wo=world: wo.ball_y
    bvy.transfer = lambda w, wo=world: wo.ball_vy * 20.0  # scale for readability
    lerr.transfer = lambda w, wo=world: wo.ball_y - wo.left_y
    rerr.transfer = lambda w, wo=world: wo.ball_y - wo.right_y

    left = s.add_actuator(label="left")
    right = s.add_actuator(label="right")
    left.output = 0.0
    right.output = 0.0
    left.output_step = 0.02
    right.output_step = 0.02
    return s


def sample_into_symbioid(s: Symbioid, world: PongWorld, tick: int) -> None:
    w = world.sensor_world()
    w["left"] = float(s.actuators[0].output)
    w["right"] = float(s.actuators[1].output)
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
    world: PongWorld,
    coach: DualPaddleCoach,
    font: pygame.font.Font,
    font_sm: pygame.font.Font,
) -> None:
    screen.fill((12, 14, 28))
    for y in range(20, H - 20, 16):
        pygame.draw.rect(screen, (40, 50, 80), (W // 2 - 2, y, 4, 8))
    ph = int(world.paddle_half * H)
    # intercept guides (learned targets)
    if coach.left.coming_toward(world):
        pygame.draw.line(
            screen,
            (40, 90, 60),
            (nx(-0.9) - 20, ny(coach.left.last_target)),
            (nx(-0.9) + 20, ny(coach.left.last_target)),
            2,
        )
    if coach.right.coming_toward(world):
        pygame.draw.line(
            screen,
            (40, 70, 100),
            (nx(0.9) - 20, ny(coach.right.last_target)),
            (nx(0.9) + 20, ny(coach.right.last_target)),
            2,
        )
    pygame.draw.rect(
        screen,
        (80, 200, 120),
        (nx(-0.9) - 8, ny(world.left_y) - ph, 12, ph * 2),
        border_radius=4,
    )
    pygame.draw.rect(
        screen,
        (80, 160, 220),
        (nx(0.9) - 4, ny(world.right_y) - ph, 12, ph * 2),
        border_radius=4,
    )
    pygame.draw.circle(
        screen,
        (240, 220, 80),
        (nx(world.ball_x), ny(world.ball_y)),
        max(6, int(world.ball_r * H * 0.5)),
    )
    hud = font.render(
        f"{world.score_left}  —  {world.score_right}   {world.last_event}",
        True,
        (200, 210, 230),
    )
    screen.blit(hud, (20, 8))
    screen.blit(font_sm.render(coach.left.summary(), True, (120, 200, 140)), (20, 32))
    screen.blit(font_sm.render(coach.right.summary(), True, (120, 170, 220)), (20, 52))
    screen.blit(
        font_sm.render(
            "Both paddles LEARN intercepts (hits/misses update policy). Esc quit.",
            True,
            (120, 130, 160),
        ),
        (20, H - 28),
    )


def main() -> None:
    world = PongWorld()
    world.reset_ball(toward=1)
    s = build_symbioid(world)
    coach = DualPaddleCoach()

    twin = s.twin_seed_thoughts()
    if twin:
        print(format_six_set_line("twin", twin, index=0), flush=True)
    print("Pong + Symbioid: both paddles learn intercept play.", flush=True)
    print("Green=left learner, blue=right. Lines = intercept targets.", flush=True)

    s.start_processes()
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Symbioid Pong — learning paddles")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("DejaVu Sans", 20)
    font_sm = pygame.font.SysFont("DejaVu Sans", 15)
    frame = 0
    sample_every = 4
    log_every = 120  # frames

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if frame % sample_every == 0:
                sample_into_symbioid(s, world, tick=frame)

            # Learn + act (writes actuator outputs)
            lo, ro = coach.control(
                world, s.actuators[0].output, s.actuators[1].output
            )
            s.actuators[0].output = lo
            s.actuators[1].output = ro
            world.set_paddles(lo, ro)

            prev_event = world.last_event
            world.step()
            if world.last_event != prev_event:
                coach.observe_event(world)
                if world.last_event in (
                    "hit_left",
                    "hit_right",
                    "score_left",
                    "score_right",
                ):
                    print(
                        f"[{world.last_event}] {coach.left.summary()} | {coach.right.summary()}",
                        flush=True,
                    )

            if frame > 0 and frame % log_every == 0:
                print(
                    f"t={frame} score={world.score_left}-{world.score_right} "
                    f"| {coach.left.summary()} | {coach.right.summary()}",
                    flush=True,
                )

            draw(screen, world, coach, font, font_sm)
            pygame.display.flip()
            clock.tick(FPS)
            frame += 1
    finally:
        s.stop_processes()
        pygame.quit()
        print(
            f"\nstopped: score={world.score_left}-{world.score_right}\n"
            f"  {coach.left.summary()}\n"
            f"  {coach.right.summary()}\n"
            f"  formations={len(s.innerface.completed_formations)} "
            f"beliefs={len(s.outerface.active_belief_ids)} "
            f"confirm={s.outerface.belief_confirms} challenge={s.outerface.belief_challenges}",
            flush=True,
        )


if __name__ == "__main__":
    main()
