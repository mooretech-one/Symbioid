"""Architecture MVP: Thought layers, Mind≠Thought, act_from_graph, nested Energy."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import (
    Energy,
    Mind,
    SEVENSPHERE_ALIGNMENT,
    SIMON_ATOMIC_THOUGHT,
    Symbioid,
    Thought,
    ThoughtLayer,
    assert_mind_not_thought,
    is_mind_aspect,
    is_thought_content,
    layer_for_role,
)


def test_thought_mvp_layers_and_simon_contract():
    t = Thought(id="t1", label="obs", layer=ThoughtLayer.PATTERN)
    assert t.is_pattern()
    t.set_layer("feeling")
    assert t.is_feeling()
    s = Thought(id="s1", layer=ThoughtLayer.STRUCTURE, threshold=10.0)
    assert s.is_structure()
    contract = Thought.simon_contract()
    assert "structure" in contract and "signal" in contract
    assert SIMON_ATOMIC_THOUGHT["not_1to1_neuron"]
    assert ThoughtLayer.STRUCTURE in SEVENSPHERE_ALIGNMENT["map_to_layers"]
    assert layer_for_role("action") is ThoughtLayer.FEELING
    assert layer_for_role("exists_in") is ThoughtLayer.STRUCTURE
    assert layer_for_role("observation") is ThoughtLayer.PATTERN
    d = t.as_dict()
    assert d["layer"] == "feeling"


def test_seed_thoughts_are_structure_layer():
    s = Symbioid(install_constitution=False)
    assert s.system.layer is ThoughtLayer.STRUCTURE
    assert s.environment.layer is ThoughtLayer.STRUCTURE


def test_mind_not_thought_enforced():
    m = Mind()
    assert_mind_not_thought(m)
    assert is_mind_aspect(m)
    assert not is_thought_content(m)
    th = Thought(id="x")
    assert is_thought_content(th)
    assert not is_mind_aspect(th)
    s = Symbioid(install_constitution=False)
    assert is_mind_aspect(s.mind)
    assert not isinstance(s.mind, Thought)


def test_action_poles_are_feeling_layer():
    m = Mind()
    a = m.ensure_action_thought("demo", "wave", host_id="h")
    assert a.layer is ThoughtLayer.FEELING


def test_act_from_graph_and_think_tick_core_agency():
    s = Symbioid(label="demo", install_constitution=True)
    eye = s.add_sensor(label="eye")
    hand = s.add_actuator(label="hand")
    # Seed last observation pole so Outerface has state
    st = Thought(id=f"{s.id}:obs:e", label="eye:0.3", transient=True, layer=ThoughtLayer.PATTERN)
    with s.mind._lock:
        s.mind._observations[f"{eye.id}:r:0.3"] = st
        s.mind._thought_to_key[st.id] = f"{eye.id}:r:0.3"
    with s.innerface._local_lock:
        s.innerface._last_obs_by_sensor[eye.id] = st
    s.mind.record_outcome([st], "wave", domain="demo", host_id=s.id, reward=150.0)
    # Core API (not demo glue)
    results = s.act_from_graph(domain="demo")
    assert results, "expected graph-backed fire attempt"
    ok, reason = results[0]
    assert ok is True or isinstance(reason, str)
    report = s.think_tick(domain="demo", run_agency=True)
    assert "pulse" in report and "actions" in report
    assert "energy_remaining" in report


def test_nested_energy_budget_falsifiable():
    root = Energy(capacity=100.0, remaining=100.0)
    child = root.nest(40.0, label="child")
    assert abs(root.remaining - 60.0) < 1e-6
    assert abs(child.capacity - 40.0) < 1e-6
    assert child.spend(10.0) is True
    assert abs(child.remaining - 30.0) < 1e-6
    # Parent not auto-drained by child spend
    assert abs(root.remaining - 60.0) < 1e-6
    assert child.spend(100.0) is False
    assert child.refuse_count >= 1
    empty_parent = Energy(capacity=5.0, remaining=0.0)
    empty_child = empty_parent.nest(10.0)
    assert empty_child.capacity == 0.0
    assert empty_parent.refuse_count >= 1


def test_symbioid_energy_enforced_think_tick():
    s = Symbioid(install_constitution=False, energy_enforced=True)
    s.energy = Energy(capacity=5.0, remaining=5.0)
    # Stimulate so fires cost energy
    for _ in range(3):
        s.stimulate(s.system, 2.0)
    r1 = s.think_tick(run_agency=False)
    assert r1["energy_remaining"] <= 5.0
    # Drain completely
    s.energy.remaining = 0.0
    r2 = s.think_tick(run_agency=False)
    assert r2["energy_remaining"] == 0.0
    # Nested sub-budget from host
    s.energy = Energy(capacity=50.0, remaining=50.0)
    nested = s.nest_energy(20.0, label="organ")
    assert abs(s.energy.remaining - 30.0) < 1e-6
    assert nested.capacity == 20.0
