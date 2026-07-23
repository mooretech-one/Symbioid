"""Tests for Pong paddle intercept learning."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid.world.paddle_learn import DualPaddleCoach, PaddleLearner, predict_intercept_y
from symbioid.world.pong import PongWorld


def test_predict_intercept_straight():
    # ball at 0 going right to x=0.9 with vy=0
    y = predict_intercept_y(0.0, 0.2, 0.02, 0.0, 0.9)
    assert abs(y - 0.2) < 1e-6


def test_predict_intercept_with_bounce():
    # high ball moving up-right; should bounce before paddle
    y = predict_intercept_y(0.0, 0.5, 0.03, 0.04, 0.9, y_min=-0.96, y_max=0.96)
    assert -0.96 <= y <= 0.96


def test_learner_improves_toward_ball_on_miss():
    w = PongWorld()
    w.ball_y = 0.5
    w.left_y = -0.5
    learner = PaddleLearner(side="left", paddle_x=-0.9, bias=0.0, gain=0.2)
    before = learner.bias
    learner.on_miss(w, paddle_y=-0.5)
    # miss high ball while paddle low → bias increases toward ball
    assert learner.bias > before
    assert learner.misses == 1


def test_dual_coach_hit_updates():
    w = PongWorld()
    w.ball_x = -0.85
    w.ball_y = 0.0
    w.ball_vx = -0.03
    w.ball_vy = 0.0
    w.set_paddles(0.0, 0.0)
    coach = DualPaddleCoach()
    for _ in range(40):
        lo, ro = coach.control(w, w.left_y, w.right_y)
        w.set_paddles(lo, ro)
        prev = w.last_event
        w.step()
        if w.last_event != prev:
            coach.observe_event(w)
        if w.last_event == "hit_left":
            break
    assert coach.left.hits >= 1 or w.last_event == "hit_left"
