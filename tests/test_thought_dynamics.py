"""Thought firing + decay (Thoughts double as neurons)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Link, Symbioid, Thought


def test_thought_decay_toward_rest():
    t = Thought(id="a", activation=2.0, resting=0.0, decay_rate=0.5)
    t.decay_step()
    assert abs(t.activation - 1.0) < 1e-9
    t.decay_step()
    assert abs(t.activation - 0.5) < 1e-9


def test_thought_fires_at_threshold():
    t = Thought(id="b", activation=0.5, threshold=1.0, default_refractory=2)
    assert t.try_fire(cycle=1) is False
    t.receive(0.6)
    assert t.activation >= 1.0
    assert t.try_fire(cycle=2) is True
    assert t.just_fired is True
    assert t.refractory_ticks == 2


def test_refractory_blocks_immediate_refire():
    t = Thought(id="c", activation=2.0, threshold=1.0, default_refractory=3)
    assert t.try_fire(cycle=1) is True
    t.activation = 2.0  # still high
    assert t.try_fire(cycle=2) is False
    t.decay_step()  # refractory 2
    t.decay_step()  # 1
    t.decay_step()  # 0
    t.activation = 2.0
    assert t.try_fire(cycle=5) is True


def test_fire_propagates_along_link_weight():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.propagate_gain = 1.0
    a = Thought(id=f"{s.id}:a", label="A", threshold=1.0)
    b = Thought(id=f"{s.id}:b", label="B", threshold=10.0)  # won't fire
    lt = Thought(id=f"{s.id}:lt", label="R", threshold=10.0)
    link = Link(
        id=f"{s.id}:link",
        label="AtoB",
        source=a,
        link_type=lt,
        target=b,
        weight=1.0,
        threshold=10.0,
    )
    for t in (a, b, lt, link):
        s.add_thought(t)
    s.stimulate(a, 1.5)
    stats = s.pulse_tick()
    assert stats["fired"] >= 1
    assert b.activation > 0.0


def test_interface_stimulus_can_fire_observation():
    s = Symbioid()
    s.mind.dynamics_enabled = True
    s.mind.observation_stimulus = 1.2
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.5
    h = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={})
    )
    assert h is not None
    obs = h["partial"]["observation"]
    assert obs.activation >= 1.0
    s.pulse_tick()
    # Observation should have fired or be hot
    assert s.last_pulse_fired >= 1 or obs.is_hot() or obs.refractory_ticks > 0


def test_pulse_tick_idempotent_safe():
    s = Symbioid()
    s.mind.dynamics_enabled = True
    for _ in range(5):
        st = s.pulse_tick()
        assert "cycle" in st
    assert s.pulse_cycle == 5


def test_dynamics_disabled_skips_pulse():
    s = Symbioid()
    s.mind.dynamics_enabled = False
    a = Thought(id=f"{s.id}:x", activation=2.0)
    s.add_thought(a)
    s.stimulate(a, 1.0)  # no-op when disabled
    st = s.pulse_tick()
    assert st["fired"] == 0
    # activation unchanged by stimulate when disabled
    assert a.activation == 2.0


def test_recommend_prefers_activated_action():
    s = Symbioid()
    s.mind.dynamics_enabled = True
    s.mind.recommend_min_valence = 0.05
    st = Thought(id=f"{s.id}:st", label="state", transient=True)
    with s.mind._lock:
        s.mind._observations["st:r:0.0"] = st
        s.mind._thought_to_key[st.id] = "st:r:0.0"
    s.add_thought(st)
    s.mind.record_outcome(
        [st], "hard", domain="tetris", host_id=s.id, reward=100.0, host=s
    )
    s.mind.record_outcome(
        [st], "left", domain="tetris", host_id=s.id, reward=10.0, host=s
    )
    # Heat hard more
    hard = s.mind._actions["act:tetris:hard"]
    s.stimulate(hard, 2.0)
    s.pulse_tick()
    rec = s.mind.recommend_action([st], domain="tetris")
    assert rec is not None
    assert rec.token == "hard"
