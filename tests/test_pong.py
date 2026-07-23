"""Headless Pong world tests (no display)."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid.world.pong import PongWorld, bounce_zone_degrees


def test_pong_step_and_sensor_world():
    w = PongWorld()
    w.reset_ball(toward=1)
    w.set_paddles(0.0, 0.0)
    before = w.ball_x
    w.step()
    assert w.ball_x != before or w.ball_y != 0.0
    sw = w.sensor_world()
    assert "ball_y" in sw and "left_err" in sw


def test_paddle_can_hit_ball():
    w = PongWorld()
    w.ball_x = -0.85
    w.ball_y = 0.0
    w.ball_vx = -w.ball_speed
    w.ball_vy = 0.0
    w.set_paddles(0.0, 0.0)
    hit = False
    for _ in range(30):
        w.step()
        if w.last_event == "hit_left":
            hit = True
            break
    assert hit
    assert abs(w.ball_vx - w.ball_speed) < 1e-12


def test_bounce_zone_degrees_five_bands():
    ph = 0.18
    # Five equal bands on rel ∈ [-1, 1]: ends ±60, mid ±30, center 0.
    assert bounce_zone_degrees(-0.9 * ph, 0.0, ph) == -60.0
    assert bounce_zone_degrees(-0.5 * ph, 0.0, ph) == -30.0
    assert bounce_zone_degrees(0.0, 0.0, ph) == 0.0
    assert bounce_zone_degrees(0.5 * ph, 0.0, ph) == 30.0
    assert bounce_zone_degrees(0.9 * ph, 0.0, ph) == 60.0
    # Band edges (rel = -0.6 is start of -30 band)
    assert bounce_zone_degrees(-0.6 * ph, 0.0, ph) == -30.0
    assert bounce_zone_degrees(0.6 * ph, 0.0, ph) == 60.0


def test_center_hit_pure_reflect():
    """Center zone: vx flips to +ball_speed, vy unchanged (pure reflect)."""
    w = PongWorld()
    w.ball_x = -0.85
    w.ball_y = 0.0
    w.ball_vx = -w.ball_speed
    w.ball_vy = 0.01
    w.set_paddles(0.0, 0.0)
    for _ in range(30):
        w.step()
        if w.last_event == "hit_left":
            break
    assert w.last_event == "hit_left"
    assert abs(w.ball_vx - w.ball_speed) < 1e-12
    assert abs(w.ball_vy - 0.01) < 1e-9


def test_end_zone_adds_60_degrees():
    """Top end of paddle sets trajectory to +60° with constant |vx|."""
    w = PongWorld()
    ph = w.paddle_half
    # Hit near top of paddle (zone +60°) with horizontal approach.
    w.ball_x = -0.85
    w.ball_y = 0.85 * ph
    w.ball_vx = -w.ball_speed
    w.ball_vy = 0.0
    w.set_paddles(0.0, 0.0)
    for _ in range(30):
        w.step()
        if w.last_event == "hit_left":
            break
    assert w.last_event == "hit_left"
    assert abs(w.ball_vx - w.ball_speed) < 1e-12
    assert w.ball_vy > 0
    # Outgoing angle from +x axis ≈ +60°; |vx| unchanged
    ang = math.degrees(math.atan2(w.ball_vy, w.ball_vx))
    assert abs(ang - 60.0) < 1.0
    # Steeper total speed is OK; horizontal component stays ball_speed
    assert abs(w.ball_vy - w.ball_speed * math.tan(math.radians(60))) < 1e-9


def test_mid_zone_adds_30_degrees():
    """Upper-mid band sets trajectory to +30° with constant |vx|."""
    w = PongWorld()
    ph = w.paddle_half
    # rel ≈ 0.4 → zone +30°
    w.ball_x = -0.85
    w.ball_y = 0.4 * ph
    w.ball_vx = -w.ball_speed
    w.ball_vy = 0.0
    w.set_paddles(0.0, 0.0)
    for _ in range(30):
        w.step()
        if w.last_event == "hit_left":
            break
    assert w.last_event == "hit_left"
    assert abs(w.ball_vx - w.ball_speed) < 1e-12
    ang = math.degrees(math.atan2(w.ball_vy, w.ball_vx))
    assert abs(ang - 30.0) < 1.0


def test_horizontal_speed_constant_across_hits():
    """|vx| stays ball_speed through angled hits and wall bounces."""
    w = PongWorld()
    ph = w.paddle_half
    w.ball_x = -0.85
    w.ball_y = 0.85 * ph
    w.ball_vx = -w.ball_speed
    w.ball_vy = 0.0
    w.set_paddles(0.0, 0.0)
    for _ in range(200):
        w.step()
        assert abs(abs(w.ball_vx) - w.ball_speed) < 1e-12, (
            f"|vx| drifted to {w.ball_vx} at event={w.last_event}"
        )
        if w.last_event in ("score_left", "score_right", "serve"):
            break
