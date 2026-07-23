"""Online learners for Pong paddles (intercept prediction + outcome updates)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from symbioid.world.pong import PongWorld


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def predict_intercept_y(
    ball_x: float,
    ball_y: float,
    ball_vx: float,
    ball_vy: float,
    paddle_x: float,
    *,
    y_min: float = -0.96,
    y_max: float = 0.96,
    max_bounces: int = 12,
) -> float:
    """
    Predict ball Y when it reaches paddle_x, including top/bottom reflections.
    If the ball is moving away, returns current ball_y.
    """
    if abs(ball_vx) < 1e-9:
        return ball_y
    # Moving away from this paddle?
    if (paddle_x - ball_x) * ball_vx <= 0:
        return ball_y

    x, y, vx, vy = ball_x, ball_y, ball_vx, ball_vy
    for _ in range(max_bounces + 1):
        if abs(vx) < 1e-9:
            return y
        t_pad = (paddle_x - x) / vx
        if t_pad < 0:
            return y
        if abs(vy) < 1e-12:
            return _clamp(y, y_min, y_max)
        if vy > 0:
            t_wall = (y_max - y) / vy
        else:
            t_wall = (y_min - y) / vy
        if t_wall < 0:
            t_wall = 1e9
        if t_pad <= t_wall + 1e-12:
            return _clamp(y + vy * t_pad, y_min, y_max)
        # hit wall first
        x += vx * t_wall
        y += vy * t_wall
        y = _clamp(y, y_min, y_max)
        vy = -vy
    return _clamp(y, y_min, y_max)


@dataclass
class PaddleLearner:
    """
    Learns to intercept the ball on one side.

    - Predicts intercept Y (physics + learnable bias / vy trust).
    - Moves paddle toward that target with learnable gain.
    - Updates from hits/misses and online intercept error.
    - Exploration noise decays as hits accumulate.
    """

    side: str  # "left" | "right"
    paddle_x: float
    gain: float = 0.12  # start slow — learning should improve this
    bias: float = 0.0
    vy_trust: float = 0.7  # 0 = ignore vy, 1 = full intercept physics
    noise: float = 0.10
    lr: float = 0.05
    hits: int = 0
    misses: int = 0
    last_target: float = 0.0
    last_intercept: float = 0.0

    def coming_toward(self, world: PongWorld) -> bool:
        if self.side == "left":
            return world.ball_vx < -1e-6
        return world.ball_vx > 1e-6

    def intercept(self, world: PongWorld) -> float:
        pure = predict_intercept_y(
            world.ball_x,
            world.ball_y,
            world.ball_vx,
            world.ball_vy,
            self.paddle_x,
        )
        # Blend raw ball_y with full intercept via vy_trust
        blended = (1.0 - self.vy_trust) * world.ball_y + self.vy_trust * pure
        self.last_intercept = blended + self.bias
        return self.last_intercept

    def act(self, world: PongWorld, paddle_y: float) -> float:
        """Return new paddle output in [-1, 1]."""
        if self.coming_toward(world):
            target = self.intercept(world)
            g = self.gain
            n = self.noise
        else:
            # Ball going away: drift toward center, less noise
            target = 0.0
            g = self.gain * 0.25
            n = self.noise * 0.3
        self.last_target = target
        err = target - paddle_y
        explore = random.gauss(0.0, n) if n > 1e-6 else 0.0
        return _clamp(paddle_y + g * err + explore, -1.0, 1.0)

    def online_tune(self, world: PongWorld, paddle_y: float) -> None:
        """Small LMS step while ball approaches: reduce intercept error."""
        if not self.coming_toward(world):
            return
        ideal = predict_intercept_y(
            world.ball_x,
            world.ball_y,
            world.ball_vx,
            world.ball_vy,
            self.paddle_x,
        )
        # How wrong is our bias relative to pure intercept?
        err = ideal - (paddle_y)  # want paddle at ideal
        self.bias += self.lr * 0.02 * (ideal - (ideal * 0 + self.bias + world.ball_y * 0) - self.bias)
        # clearer: bias moves so (blended + bias) tracks ideal
        pred = (1.0 - self.vy_trust) * world.ball_y + self.vy_trust * ideal + self.bias
        self.bias += self.lr * 0.03 * (ideal - pred)
        self.bias = _clamp(self.bias, -0.4, 0.4)
        # gently raise vy_trust toward physics
        self.vy_trust = _clamp(self.vy_trust + 0.0005, 0.3, 1.0)

    def on_hit(self, world: PongWorld, paddle_y: float) -> None:
        self.hits += 1
        offset = world.ball_y - paddle_y
        # center hits are better — nudge bias opposite small offsets
        self.bias -= self.lr * 0.15 * offset
        self.bias = _clamp(self.bias, -0.4, 0.4)
        self.gain = _clamp(self.gain + 0.015, 0.08, 0.65)
        self.noise = max(0.008, self.noise * 0.94)
        self.vy_trust = _clamp(self.vy_trust + 0.03, 0.3, 1.0)

    def on_miss(self, world: PongWorld, paddle_y: float) -> None:
        self.misses += 1
        # Ball got past us — bias toward where the ball was
        err = world.ball_y - paddle_y
        self.bias += self.lr * 0.6 * err
        self.bias = _clamp(self.bias, -0.4, 0.4)
        self.gain = _clamp(self.gain + 0.03, 0.08, 0.65)
        self.noise = min(0.14, self.noise * 1.08 + 0.01)
        # trust intercept more after misses (stop just chasing current y)
        self.vy_trust = _clamp(self.vy_trust + 0.05, 0.3, 1.0)

    def summary(self) -> str:
        return (
            f"{self.side}: hits={self.hits} misses={self.misses} "
            f"gain={self.gain:.2f} bias={self.bias:+.2f} "
            f"vy_trust={self.vy_trust:.2f} noise={self.noise:.3f}"
        )


@dataclass
class DualPaddleCoach:
    """Owns left/right learners and applies hit/miss events from PongWorld."""

    left: PaddleLearner = field(
        default_factory=lambda: PaddleLearner(side="left", paddle_x=-0.9)
    )
    right: PaddleLearner = field(
        default_factory=lambda: PaddleLearner(side="right", paddle_x=0.9)
    )
    _last_event: str = ""

    def control(self, world: PongWorld, left_out: float, right_out: float) -> tuple[float, float]:
        self.left.online_tune(world, left_out)
        self.right.online_tune(world, right_out)
        return self.left.act(world, left_out), self.right.act(world, right_out)

    def observe_event(self, world: PongWorld) -> None:
        ev = world.last_event
        if ev == self._last_event and ev not in ("hit_left", "hit_right", "score_left", "score_right"):
            return
        # Only react once per discrete event change
        if ev == self._last_event:
            return
        prev = self._last_event
        self._last_event = ev
        if ev == "hit_left":
            self.left.on_hit(world, world.left_y)
        elif ev == "hit_right":
            self.right.on_hit(world, world.right_y)
        elif ev == "score_right":
            # ball passed left paddle
            self.left.on_miss(world, world.left_y)
        elif ev == "score_left":
            self.right.on_miss(world, world.right_y)
        _ = prev
