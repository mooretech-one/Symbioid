"""Hebbian Link.weight plasticity — six-set synaptic strengthening."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Link, Symbioid, Thought


def _pair(s: Symbioid, w: float = 1.0) -> tuple[Thought, Thought, Link, Link]:
    a = Thought(id=f"{s.id}:pole_a", label="A", threshold=1.0)
    b = Thought(id=f"{s.id}:pole_b", label="B", threshold=1.0)
    lt = Thought(id=f"{s.id}:lt", label="R", threshold=10.0)
    ab = Link(
        id=f"{s.id}:ab",
        source=a,
        link_type=lt,
        target=b,
        weight=w,
        threshold=10.0,
    )
    ba = Link(
        id=f"{s.id}:ba",
        source=b,
        link_type=lt,
        target=a,
        weight=w,
        threshold=10.0,
    )
    for t in (a, b, lt, ab, ba):
        s.add_thought(t)
    return a, b, ab, ba


def test_co_fire_increases_weight():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = True
    s.mind.hebb_lr = 0.1
    s.mind.hebb_co_fire_scale = 1.0
    s.mind.propagate_gain = 0.1  # keep spread small
    a, b, ab, ba = _pair(s, w=1.0)
    w0 = ab.weight
    s.stimulate(a, 1.5)
    s.stimulate(b, 1.5)
    st = s.pulse_tick()
    assert st["fired"] >= 2
    assert st["hebb"] >= 1
    assert ab.weight > w0 or ba.weight > w0


def test_weight_clamped_at_max():
    s = Symbioid(install_constitution=False)
    s.mind.hebb_enabled = True
    s.mind.weight_max = 2.0
    s.mind.hebb_lr = 1.0
    a, b, ab, ba = _pair(s, w=1.9)
    s.stimulate(a, 2.0)
    s.stimulate(b, 2.0)
    for _ in range(20):
        a.activation = 2.0
        b.activation = 2.0
        a.refractory_ticks = 0
        b.refractory_ticks = 0
        s.mark_hot(a)
        s.mark_hot(b)
        s.pulse_tick()
    assert ab.weight <= 2.0 + 1e-9
    assert ba.weight <= 2.0 + 1e-9


def test_high_weight_recruits_mate():
    """weight * gain large enough that one pole fire pushes mate over threshold."""
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = False  # freeze weights for this test
    s.mind.propagate_gain = 1.0
    a, b, ab, ba = _pair(s, w=2.0)
    b.threshold = 1.0
    b.activation = 0.0
    s.stimulate(a, 1.5)
    # First tick: A fires, B receives 1.5*2*1 = 3.0 → well over threshold
    s.pulse_tick()
    assert a.just_fired or a.refractory_ticks > 0
    assert b.activation >= 1.0
    # Second tick after refractory on A: B can fire from residual
    b.refractory_ticks = 0
    s.mark_hot(b)
    s.pulse_tick()
    assert b.refractory_ticks > 0 or b.just_fired or b.activation >= b.threshold


def test_hebb_disabled_no_weight_change():
    s = Symbioid(install_constitution=False)
    s.mind.hebb_enabled = False
    s.mind.dynamics_enabled = True
    a, b, ab, ba = _pair(s, w=1.0)
    s.stimulate(a, 1.5)
    s.stimulate(b, 1.5)
    s.pulse_tick()
    assert abs(ab.weight - 1.0) < 1e-9


def test_record_outcome_reinforces_edges():
    s = Symbioid()
    s.mind.hebb_enabled = True
    st = Thought(id=f"{s.id}:st", label="state", transient=True, threshold=1.0)
    s.add_thought(st)
    with s.mind._lock:
        s.mind._observations["st:r:0"] = st
        s.mind._thought_to_key[st.id] = "st:r:0"
    act = s.mind.record_outcome(
        [st], "hard", domain="tetris", host_id=s.id, reward=100.0, host=s
    )
    # Edges should exist with weight > 1 (initial + reward nudge)
    edges = [
        t
        for t in s.thoughts.values()
        if isinstance(t, Link)
        and {t.source.id, t.target.id} == {st.id, act.id}
    ]
    assert edges, "expected reciprocal policy edges"
    assert any(e.weight > 1.0 for e in edges)
