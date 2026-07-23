"""Minimal Pong physics for Symbioid closed-loop demos (no rendering)."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# Five equal vertical zones on each paddle (bottom → top).
# Center reflects; mid zones ±30°; ends ±60° (applied after x-reflect).
_BOUNCE_ZONE_DEGREES = (-60.0, -30.0, 0.0, 30.0, 60.0)


def bounce_zone_degrees(ball_y: float, paddle_y: float, paddle_half: float) -> float:
    """
    Map hit position on the paddle to a deflection angle in degrees.

    Relative offset rel = (ball_y - paddle_y) / paddle_half in [-1, 1]
    is split into five equal bands of width 0.4.
    """
    if paddle_half <= 0:
        return 0.0
    rel = _clamp((ball_y - paddle_y) / paddle_half, -1.0, 1.0)
    # Map [-1, 1] → zone index 0..4 (equal fifths).
    # Use min so rel == 1.0 lands in the last zone.
    idx = min(4, int((rel + 1.0) / 0.4))
    return _BOUNCE_ZONE_DEGREES[idx]


# Avoid tan() blow-up if stacked angles approach vertical.
_MAX_TRAJECTORY_DEG = 80.0


@dataclass
class PongWorld:
    """
    Unit-square-ish playfield with coordinates in [-1, 1] for y and roughly
    [-1, 1] for x (left wall -1, right wall +1).

    Paddle Y is controlled externally (actuator.output in [-1, 1]).

    Paddle hits use five bounce zones: center pure-reflects (vx flip),
    mid bands add/subtract 30°, ends add/subtract 60°.

    Horizontal speed is constant: |ball_vx| == ball_speed always (zones and
    walls only change direction / vertical component).
    """

    width: float = 2.0
    height: float = 2.0
    paddle_half: float = 0.18
    ball_r: float = 0.04
    ball_speed: float = 0.025
    # state
    ball_x: float = 0.0
    ball_y: float = 0.0
    ball_vx: float = 0.02
    ball_vy: float = 0.015
    left_y: float = 0.0
    right_y: float = 0.0
    score_left: int = 0
    score_right: int = 0
    ticks: int = 0
    last_event: str = ""

    def reset_ball(self, toward: int = 1) -> None:
        """toward: +1 serve right, -1 serve left."""
        self.ball_x = 0.0
        self.ball_y = random.uniform(-0.3, 0.3)
        self.ball_vx = self.ball_speed * toward
        self.ball_vy = self.ball_speed * random.choice([-1.0, 1.0]) * random.uniform(0.5, 1.0)
        self.last_event = "serve"

    def set_paddles(self, left_y: float, right_y: float) -> None:
        lim = 1.0 - self.paddle_half
        self.left_y = _clamp(left_y, -lim, lim)
        self.right_y = _clamp(right_y, -lim, lim)

    def _set_horizontal(self, going_right: bool) -> None:
        """Lock |vx| to ball_speed (constant horizontal speed)."""
        self.ball_vx = self.ball_speed if going_right else -self.ball_speed

    def _apply_paddle_bounce(self, paddle_y: float, *, going_right: bool) -> None:
        """
        Reflect off paddle; apply five-zone angle without changing |vx|.

        Center (0°): pure reflect — flip horizontal direction, keep vy.
        Other zones: add/subtract zone degrees to the current trajectory
        angle, then re-express with fixed horizontal speed so |vx| stays
        ball_speed and only vy absorbs the deflection.

        going_right: True after left-paddle hit (ball should leave to +x).
        """
        h = self.ball_speed
        deg = bounce_zone_degrees(self.ball_y, paddle_y, self.paddle_half)

        if deg == 0.0:
            # Pure x-reflect; vertical component unchanged.
            self._set_horizontal(going_right)
            return

        # Trajectory angle from +x (using current |vx| as horizontal ref).
        hx = abs(self.ball_vx) if abs(self.ball_vx) > 1e-12 else h
        cur = math.atan2(self.ball_vy, hx)
        new_ang = cur + math.radians(deg)
        max_a = math.radians(_MAX_TRAJECTORY_DEG)
        new_ang = _clamp(new_ang, -max_a, max_a)

        self._set_horizontal(going_right)
        self.ball_vy = math.tan(new_ang) * h

    def step(self) -> None:
        """One physics tick."""
        self.ticks += 1
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # top / bottom (vy only — horizontal speed unchanged)
        if self.ball_y >= 1.0 - self.ball_r:
            self.ball_y = 1.0 - self.ball_r
            self.ball_vy = -abs(self.ball_vy)
            self.last_event = "wall_top"
        elif self.ball_y <= -1.0 + self.ball_r:
            self.ball_y = -1.0 + self.ball_r
            self.ball_vy = abs(self.ball_vy)
            self.last_event = "wall_bot"

        # left paddle at x ≈ -0.9
        lx = -0.9
        if self.ball_vx < 0 and self.ball_x - self.ball_r <= lx + 0.03:
            if abs(self.ball_y - self.left_y) <= self.paddle_half + self.ball_r:
                self.ball_x = lx + 0.03 + self.ball_r
                self._apply_paddle_bounce(self.left_y, going_right=True)
                self.last_event = "hit_left"
            elif self.ball_x < -1.05:
                self.score_right += 1
                self.last_event = "score_right"
                self.reset_ball(toward=1)

        # right paddle at x ≈ +0.9
        rx = 0.9
        if self.ball_vx > 0 and self.ball_x + self.ball_r >= rx - 0.03:
            if abs(self.ball_y - self.right_y) <= self.paddle_half + self.ball_r:
                self.ball_x = rx - 0.03 - self.ball_r
                self._apply_paddle_bounce(self.right_y, going_right=False)
                self.last_event = "hit_right"
            elif self.ball_x > 1.05:
                self.score_left += 1
                self.last_event = "score_left"
                self.reset_ball(toward=-1)

    def sensor_world(self) -> dict[str, float]:
        """Values for Sensor.transfer / Interface world map."""
        return {
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "left_y": self.left_y,
            "right_y": self.right_y,
            "left_err": self.ball_y - self.left_y,
            "right_err": self.ball_y - self.right_y,
            "score_left": float(self.score_left),
            "score_right": float(self.score_right),
        }
