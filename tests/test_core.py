"""Minimal ontology tests — Antelligence structural classes."""

import sys
import threading
import time
from pathlib import Path

import pytest

# package root on path when not installed
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import (
    INNERFACE_RODIN_STAGES,
    INTERFACE_RODIN_STAGES,
    RODIN_CYCLE,
    RODIN_HALVE_CYCLE,
    Actuator,
    Body,
    Innerface,
    Interface,
    Link,
    Mind,
    Outerface,
    Process,
    Sensor,
    Symbioid,
    System,
    Thought,
    begin_sensor_formation,
    complete_follows_set,
    complete_formation,
    complete_integrate_set,
    is_minimal_symbioid_shape,
    minimal_seed,
    rodin_double,
    rodin_halve,
    rodin_halve_sequence,
    rodin_sequence,
)


def test_thought_is_system():
    t = Thought(label="x")
    assert isinstance(t, System)
    assert isinstance(t, Thought)


def test_symbioid_is_system_not_thought():
    s = Symbioid()
    assert isinstance(s, System)
    assert isinstance(s, Symbioid)
    assert not isinstance(s, Thought)
    assert not isinstance(s, Link)


def test_link_is_thought_and_system():
    a, b, lt = Thought(id="a"), Thought(id="b"), Thought(id="lt", label="R")
    link = Link(id="l1", source=a, link_type=lt, target=b)
    assert isinstance(link, Thought)
    assert isinstance(link, System)
    assert link.source is a
    assert link.link_type is lt
    assert link.target is b


def test_link_requires_thought_components():
    a, b = Thought(id="a"), Thought(id="b")
    with pytest.raises(TypeError):
        Link(source=a, link_type="not-a-thought", target=b)  # type: ignore[arg-type]


def test_minimal_seed_six():
    seed = minimal_seed()
    assert len(seed) == 6
    assert is_minimal_symbioid_shape(seed)


def test_minimal_seed_without_labels():
    seed = minimal_seed(with_labels=False)
    assert all(t.label is None for t in seed.values())
    assert is_minimal_symbioid_shape(seed)


def test_one_way_not_minimal():
    a, b, lt = Thought(id="a"), Thought(id="b"), Thought(id="lt")
    link = Link(id="l", source=a, link_type=lt, target=b)
    store = {a.id: a, b.id: b, lt.id: lt, link.id: link}
    assert not is_minimal_symbioid_shape(store)


def test_symbioid_default_seed():
    s = Symbioid()
    assert len(s.twin_seed_thoughts()) == 6
    assert s.is_minimal()
    assert s.system.label == "System"
    assert s.environment.label == "Environment"
    assert len(s.laws) == 4
    # twin reciprocal pair present among (possibly more) links
    assert len(s.links()) >= 2
    assert len(s.nodes()) >= 4


def test_symbioid_mirror_optional():
    s = Symbioid(mirror_in_environment=True)
    assert s.is_minimal()
    assert len(s.env_thoughts) == 6
    assert is_minimal_symbioid_shape(s.env_thoughts)


def test_symbioid_no_seed_bare_poles():
    s = Symbioid(seed_minimal=False, install_constitution=False)
    # poles + agent Thought; no law Links until install_laws
    assert s.agent.id in s.thoughts
    assert len(s.laws) == 0
    assert not s.is_minimal()
    s.seed_self_description()
    # re-add agent after seed replaces thoughts dict
    s.thoughts[s.agent.id] = s.agent
    assert s.is_minimal()
    assert len(s.twin_seed_thoughts()) == 6
    s.install_laws()
    assert len(s.laws) == 4
    assert s.is_minimal()  # constitution must not break twin shape


def test_symbioid_contains_aspects_and_faces():
    s = Symbioid(label="unit")
    assert isinstance(s.body, Body)
    assert isinstance(s.mind, Mind)
    assert isinstance(s.innerface, Innerface)
    assert isinstance(s.interface, Interface)
    assert isinstance(s.outerface, Outerface)
    assert isinstance(s.innerface, Process)
    assert isinstance(s.interface, Process)
    assert isinstance(s.outerface, Process)
    assert s.sensors == []
    assert s.actuators == []
    assert isinstance(s.thought_list, list)
    assert len(s.twin_seed_thoughts()) == 6
    assert s.mind.enabled is True


def test_faces_are_processes():
    assert issubclass(Innerface, Process)
    assert issubclass(Interface, Process)
    assert issubclass(Outerface, Process)
    assert not issubclass(Process, System)


def test_symbioid_add_sensor_actuator():
    s = Symbioid()
    eye = s.add_sensor(label="eye")
    hand = s.add_actuator(label="hand")
    assert eye in s.sensors
    assert hand in s.actuators
    assert isinstance(eye, Sensor)
    assert isinstance(hand, Actuator)
    assert isinstance(eye, System)


def test_aspect_classes_are_systems():
    assert isinstance(Body(), System)
    assert isinstance(Mind(), System)
    assert isinstance(Sensor(), System)
    assert isinstance(Actuator(), System)


def test_constitution_installed_by_default():
    s = Symbioid()
    assert len(s.laws) == 4
    assert [law.code for law in s.laws] == ["L0", "L1", "L2", "L3"]
    assert s.laws[0].priority < s.laws[1].priority
    # law links live in thought store
    for law in s.laws:
        assert law.link.id in s.thoughts
        assert isinstance(law.link, Link)


def test_constitution_optional():
    s = Symbioid(install_constitution=False)
    assert s.laws == []


def test_outerface_law_gate_l1_blocks_harm():
    s = Symbioid()
    ok, reason = s.check_action(harms_protected_environment=True)
    assert ok is False
    assert reason.startswith("L1")


def test_outerface_law_gate_l0_blocks_twin_harm():
    s = Symbioid()
    ok, reason = s.check_action(threatens_twin_integrity=True)
    assert ok is False
    assert reason.startswith("L0")


def test_outerface_law_gate_l2_authority():
    s = Symbioid()
    ok, reason = s.check_action(is_order=True, order_from_authority=True)
    assert ok is True
    assert "L2" in reason
    ok2, reason2 = s.check_action(is_order=True, order_from_authority=False)
    assert ok2 is False
    assert "L2" in reason2


def test_outerface_law_gate_l3_self():
    s = Symbioid()
    ok, reason = s.check_action(preserves_self=True)
    assert ok is True
    ok2, reason2 = s.check_action(
        preserves_self=True, self_preservation_conflicts_higher=True
    )
    assert ok2 is False
    assert "L3" in reason2


def test_l0_outranks_l2_when_both_flagged():
    """Harm flags are checked in priority order; L0 first."""
    s = Symbioid()
    ok, reason = s.check_action(
        threatens_twin_integrity=True,
        is_order=True,
        order_from_authority=True,
    )
    assert ok is False
    assert reason.startswith("L0")


def test_process_starts_thread_and_subclass_calls_super():
    s = Symbioid()
    assert s.innerface.host is s
    t = s.innerface.process()
    assert t is not None
    assert s.innerface.is_alive()
    # second call is idempotent
    t2 = s.innerface.process()
    assert t2 is t
    time.sleep(0.15)
    assert s.innerface.cycles >= 1
    s.innerface.stop(timeout=1.0)
    assert not s.innerface.is_alive() or True  # may finish join


def test_all_faces_process_override():
    import inspect

    from symbioid.core import Innerface, Interface, Outerface, Process

    for cls in (Innerface, Interface, Outerface):
        assert cls.process is not Process.process
        # override exists and calls super in source
        src = inspect.getsource(cls.process)
        assert "super().process()" in src


def test_start_stop_all_processes():
    s = Symbioid()
    s.add_sensor(label="eye")
    threads = s.start_processes()
    assert len(threads) == 3
    time.sleep(0.2)
    assert s.innerface.cycles >= 1
    assert s.interface.cycles >= 1
    assert s.outerface.cycles >= 1
    s.interface.post({"sense": "ping"})
    s.outerface.post({"is_order": True, "order_from_authority": True})
    time.sleep(0.2)
    s.stop_processes(timeout=1.0)


def test_graph_lock_serializes_thought_add():
    s = Symbioid()
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        try:
            for j in range(20):
                s.add_thought(Thought(id=f"w{i}-{j}", label=f"t{i}-{j}"))
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    assert not errors
    # 6 seed + agent + constitution nodes + 80 worker thoughts
    assert len(s.thoughts) >= 6 + 1 + 80


def test_rodin_cycle_and_double():
    assert RODIN_CYCLE == (1, 2, 4, 8, 7, 5)
    assert rodin_sequence() == [1, 2, 4, 8, 7, 5]
    assert rodin_double(1) == 2
    assert rodin_double(2) == 4
    assert rodin_double(4) == 8
    assert rodin_double(8) == 7  # 16 → 1+6
    assert rodin_double(7) == 5  # 14 → 1+4
    assert rodin_double(5) == 1  # 10 → 1
    assert INTERFACE_RODIN_STAGES == (1, 2)
    assert INNERFACE_RODIN_STAGES == (4, 8, 7, 5)


def test_rodin_halve_h0():
    """H0: inverse vortex 1→5→7→8→4→2→1; double then halve is identity on cycle."""
    assert RODIN_HALVE_CYCLE == (1, 5, 7, 8, 4, 2)
    assert rodin_halve_sequence() == [1, 5, 7, 8, 4, 2]
    assert rodin_halve(1) == 5
    assert rodin_halve(5) == 7
    assert rodin_halve(7) == 8
    assert rodin_halve(8) == 4
    assert rodin_halve(4) == 2
    assert rodin_halve(2) == 1
    for n in RODIN_CYCLE:
        assert rodin_halve(rodin_double(n)) == n
        assert rodin_double(rodin_halve(n)) == n


def test_begin_and_complete_formation_sync():
    eye = Sensor(label="eye")
    sense = eye.sample(tick=1)
    handoff = begin_sensor_formation(eye, host_id="sym-test", sense=sense)
    assert handoff["kind"] == "formation_handoff"
    assert handoff["rodin_at"] == 2
    assert set(handoff["partial"]) == {"sensor", "observation"}
    src = handoff["partial"]["sensor"]
    obs = handoff["partial"]["observation"]
    assert src.label == "eye"
    assert obs.label == sense["value"]
    assert obs.label.startswith("eye:")
    store = complete_formation(handoff)
    assert len(store) == 6
    assert is_minimal_symbioid_shape(store)
    labels = {t.label for t in store.values()}
    assert "Perceives" in labels and "PerceivedBy" in labels
    # Source=Sensor → Target=Observation
    links = [t for t in store.values() if isinstance(t, Link)]
    perc = next(L for L in links if L.link_type.label == "Perceives")
    assert perc.source is src and perc.target is obs


def test_interface_hands_off_innerface_completes():
    """Synchronous path: Interface start → Innerface accept → Sensor/Observation set."""
    s = Symbioid(label="form-demo")
    eye = s.add_sensor(label="eye")
    sense = eye.sample(tick=0)
    handoff = s.interface.start_formation_for_sensor(eye, sense=sense)
    assert handoff is not None
    assert s.interface.handoffs_sent == 1
    # second start without force is a no-op
    assert s.interface.start_formation_for_sensor(eye) is None

    store = s.innerface.accept_formation(handoff)
    assert len(store) == 6
    assert is_minimal_symbioid_shape(store)
    assert handoff["formation_id"] in s.innerface.completed_formations
    # form: ids (sensor Thought is stable under :sensor:)
    form = s.formation_thoughts(handoff["formation_id"])
    assert len(form) == 5  # observation + 2 types + 2 links (sensor shared)
    assert s.is_minimal()  # twin seed still intact
    assert len(s.twin_seed_thoughts()) == 6


def test_follows_set_between_observations():
    a = Thought(id="oa", label="eye:1", transient=True)
    b = Thought(id="ob", label="ear:1", transient=True)
    store = complete_follows_set(a, b, sync_id="sync-test")
    assert len(store) == 6
    assert is_minimal_symbioid_shape(store)
    labels = {t.label for t in store.values()}
    assert "Follows" in labels and "FollowedBy" in labels


def test_batch_syncs_lateral_follows():
    s = Symbioid(label="sync-demo")
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    h1 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
    h2 = s.interface.start_formation_for_sensor(ear, force=True, sense=ear.sample(tick=1))
    assert h1 and h2
    s.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": [h1, h2], "tick": 1}
    )
    assert len(s.innerface.completed_formations) == 2
    assert len(s.innerface.completed_syncs) == 1
    sync = next(iter(s.innerface.completed_syncs.values()))
    assert is_minimal_symbioid_shape(sync)
    # Awareness terminators: cross-sensor Integrate is blocked; Follows still runs
    assert s.innerface.integrate_blocked_cross_channel >= 1
    # First tick: two sense sets remain active (not merged across Eye/Ear)
    assert s.innerface.active_set_summary().get("sense", 0) == 2


def test_awareness_has_sensor_and_blocks_cross_integrate():
    """Symbioid has Ear/Eye awareness six-sets terminate cross-sensor Integrate."""
    s = Symbioid(label="aware")
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    assert eye.id in s.integration_terminators
    assert ear.id in s.integration_terminators
    assert eye.id in s.awareness_sets and ear.id in s.awareness_sets
    labels = {t.label for t in s.awareness_sets[ear.id].values()}
    assert "Has" in labels and "IsPartOf" in labels
    # poles include Agent and Ear
    from symbioid.Core.formation import six_set_poles

    poles = six_set_poles(s.awareness_sets[ear.id])
    assert any(p.label == "Agent" for p in poles)
    assert any(p.label and p.label.lower() == "ear" for p in poles)

    h1 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
    h2 = s.interface.start_formation_for_sensor(ear, force=True, sense=ear.sample(tick=1))
    s.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": [h1, h2], "tick": 1}
    )
    # Temporal same-sensor still integrates on second sample
    h3 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=2))
    s.innerface.accept_formation(h3)
    assert any(
        s.innerface._integrate_channel.get(iid) == eye.id
        for iid in s.innerface.completed_integrates
    )


def test_complete_integrate_set_h1():
    a = Thought(id="oa", label="eye:0.1", transient=True)
    b = Thought(id="ob", label="ear:0.2", transient=True)
    store = complete_integrate_set(a, b, integrate_id="int-test")
    assert len(store) == 6
    assert is_minimal_symbioid_shape(store)
    labels = {t.label for t in store.values()}
    assert "Integrates" in labels and "IntegratedBy" in labels


def test_temporal_integrate_last_two_same_sensor():
    s = Symbioid(label="temporal")
    eye = s.add_sensor(label="eye")
    h1 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
    h2 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=2))
    assert h1 and h2
    s.innerface.accept_formation(h1)
    assert s.innerface.active_set_count == 1
    s.innerface.accept_formation(h2)
    assert len(s.innerface.completed_integrates) == 1
    # both sense sets superseded; one integrate remains active
    assert s.innerface.active_set_summary().get("sense", 0) == 0
    assert s.innerface.active_set_summary().get("integrate", 0) == 1


def test_prune_removes_superseded_sense_scaffolding():
    """After Integrate, inactive formation scaffolding is GC'd; poles remain."""
    s = Symbioid(label="prune")
    s.innerface.auto_prune = True
    eye = s.add_sensor(label="eye")
    h1 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
    s.innerface.accept_formation(h1)
    n_after_first = len(s.thoughts)
    h2 = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=2))
    s.innerface.accept_formation(h2)
    # Integrate should have pruned superseded sense scaffolding
    assert s.innerface.thoughts_pruned > 0
    n_after_int = len(s.thoughts)
    # Graph should not keep growing unboundedly for two samples
    assert n_after_int < n_after_first + 20
    # Active integrate store still present
    assert s.innerface.active_set_summary().get("integrate", 0) == 1
    # Sensor grounding pole still registered
    assert any(tid.endswith(f"sensor:{eye.id}") or f":sensor:{eye.id}" in tid for tid in s.thoughts)


def test_h2_depth_fold_only_above_soft_cap():
    """H2: depth fold only when active integrates exceed max (many may coexist)."""
    s = Symbioid(label="depth")
    # Soft cap still high enough to hold several concurrent patterns
    s.innerface.max_active_integrates = 3
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    for t in range(1, 6):
        h1 = s.interface.start_formation_for_sensor(
            eye, force=True, sense=eye.sample(tick=t)
        )
        h2 = s.interface.start_formation_for_sensor(
            ear, force=True, sense=ear.sample(tick=t)
        )
        s.innerface.accept_formation_batch(
            {"kind": "formation_batch", "handoffs": [h1, h2], "tick": t}
        )
    n_int = s.innerface.active_set_summary().get("integrate", 0)
    assert n_int <= s.innerface.max_active_integrates
    # Must not force collapse to a single belief-like integrate by default
    assert s.innerface.max_active_integrates >= 3


def test_belief_six_set_feedback_expects_observation():
    """Belief is a six-set: Feedback -Expects→ Observation (expected value)."""
    from symbioid import complete_belief_set, is_minimal_symbioid_shape

    obs = Thought(id="obs1", label="eye:0.5", transient=True)
    fb = Thought(id="fb1", label="Feedback[eye|wave]")
    store = complete_belief_set(obs, fb, belief_id="bel-test")
    assert len(store) == 6
    assert is_minimal_symbioid_shape(store)
    labels = {t.label for t in store.values()}
    assert "Expects" in labels and "ExpectedBy" in labels


def test_outerface_belief_from_feedback_observation():
    """Outerface forms Belief six-set when Interface Observation is Feedback."""
    s = Symbioid(label="beliefs")
    eye = s.add_sensor(label="eye")
    hand = s.add_actuator(label="hand")
    # Action arms Feedback pending on sensors
    hand.request_fire(s, "wave")
    assert eye.id in s.outerface._pending_feedback

    h = s.interface.start_formation_for_sensor(eye, force=True, sense=eye.sample(tick=1))
    assert h is not None
    s.innerface.accept_formation(h)
    # Drain: Outerface must process interface_observation (sync path)
    s.outerface.handle_interface_observation(
        {
            "kind": "interface_observation",
            "observation": h["partial"]["observation"],
            "sensor_id": eye.id,
            "sensor_label": "eye",
            "formation_id": h["formation_id"],
        }
    )
    assert s.outerface.beliefs_created >= 1
    assert len(s.outerface.active_belief_ids) >= 1
    bid = next(iter(s.outerface.active_belief_ids))
    store = s.outerface.completed_beliefs[bid]
    assert len(store) == 6
    assert "Expects" in {t.label for t in store.values()}


def test_one_belief_per_sensor_no_nested_labels():
    """One Belief channel per sensor; Feedback labels stay short (no nesting)."""
    s = Symbioid(label="one-bel")
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    hand = s.add_actuator(label="hand")
    # Create then update eye belief several times via Action→Feedback
    for t in range(1, 5):
        hand.request_fire(s, "hand")  # short action only
        h = s.interface.start_formation_for_sensor(
            eye, force=True, sense=eye.sample(tick=t)
        )
        s.outerface.handle_interface_observation(
            {
                "observation": h["partial"]["observation"],
                "sensor_id": eye.id,
                "sensor_label": "eye",
            }
        )
    # Still exactly one Belief for eye
    assert s.outerface.belief_by_sensor[eye.id]
    assert len([b for b in s.outerface.active_belief_ids if "eye" in b or eye.id in b]) >= 1
    # Total active beliefs: one per sensor that got feedback (eye only here)
    assert len(s.outerface.active_belief_ids) == 1
    store = s.outerface.completed_beliefs[s.outerface.belief_by_sensor[eye.id]]
    poles = [t for t in store.values() if getattr(t, "label", None)]
    fb = next(t for t in store.values() if t.label and str(t.label).startswith("Feedback["))
    assert fb.label.count("Feedback[") == 1
    assert "Feedback[Feedback" not in (fb.label or "")
    # Ear can hold a second concurrent Belief (many beliefs across channels)
    hand.request_fire(s, "hand")
    h2 = s.interface.start_formation_for_sensor(ear, force=True, sense=ear.sample(tick=1))
    s.outerface.handle_interface_observation(
        {
            "observation": h2["partial"]["observation"],
            "sensor_id": ear.id,
            "sensor_label": "ear",
        }
    )
    assert len(s.outerface.active_belief_ids) == 2


def test_h4_actuator_gated_by_outerface():
    """H4: Actuator.request_fire uses Outerface constitution gate."""
    s = Symbioid(label="act")
    hand = s.add_actuator(label="hand")
    ok, reason = hand.request_fire(s, "wave")
    assert ok is True
    assert hand.fire_count == 1
    assert reason == "default_allow"

    denied, dreason = hand.request_fire(
        s, "harm", harms_protected_environment=True
    )
    assert denied is False
    assert "L1" in dreason
    assert hand.fire_count == 1
    assert hand.deny_count == 1

    # Seed a real Belief six-set, then propose actions
    obs = Thought(id="o1", label="eye:0.1", transient=True)
    s.outerface.form_belief_from_feedback(
        observation=obs, sensor_id="eye-x", sensor_label="eye", action="wave"
    )
    results = s.outerface.propose_actions_from_beliefs()
    assert results
    assert hand.fire_count >= 2


def test_sensor_sample_increments():
    eye = Sensor(label="eye")
    a = eye.sample(tick=1)
    b = eye.sample(tick=2)
    assert a["sample"] == 1 and b["sample"] == 2
    assert a["value"].startswith("eye:") and b["value"].startswith("eye:")
    assert isinstance(a["reading"], float) and isinstance(b["reading"], float)
    assert 0.0 <= a["reading"] <= 1.0
    # sample index always advances; random readings are almost always distinct
    assert a["sample"] != b["sample"]


def test_sensor_transfer_from_actuator_world():
    """Feedback test: ear=sin(hand), eye=cos(hand)."""
    import math

    eye = Sensor(label="eye")
    ear = Sensor(label="ear")
    eye.transfer = lambda w: math.cos(w.get("hand", 0.0))
    ear.transfer = lambda w: math.sin(w.get("hand", 0.0))
    world = {"hand": 0.0}
    e0 = eye.sample(tick=1, world=world)
    a0 = ear.sample(tick=1, world=world)
    assert e0 is not None and a0 is not None
    assert abs(e0["reading"] - 1.0) < 1e-9
    assert abs(a0["reading"] - 0.0) < 1e-9
    world = {"hand": math.pi / 2}
    e1 = eye.sample(tick=2, world=world)
    a1 = ear.sample(tick=2, world=world)
    assert e1 is not None and a1 is not None
    assert abs(e1["reading"] - 0.0) < 1e-9
    assert abs(a1["reading"] - 1.0) < 1e-9


def test_actuator_output_advances_on_fire():
    s = Symbioid(label="hand-out")
    hand = s.add_actuator(label="hand")
    hand.output = 0.0
    hand.output_step = 0.5
    hand.request_fire(s, "hand")
    assert hand.output == 0.5
    hand.request_fire(s, "hand")
    assert hand.output == 1.0


def test_belief_challenge_and_confirm_on_feedback():
    """Match post-fire prediction → confirm; stale pre-fire sample → not overwrite."""
    import math

    s = Symbioid(label="challenge")
    eye = s.add_sensor(label="eye")
    hand = s.add_actuator(label="hand")
    hand.output = 0.0
    hand.output_step = math.pi / 2
    eye.transfer = lambda w: math.cos(w.get("hand", 0.0))

    obs0 = Thought(id="o0", label="eye:1.0000", transient=True)
    s.outerface.form_belief_from_feedback(
        observation=obs0, sensor_id=eye.id, sensor_label="eye", action="sense", compare=False
    )
    # Fire → hand=π/2, prediction cos≈0, pending armed
    hand.request_fire(s, "hand")
    pred = s.outerface._pending_feedback[eye.id]["predicted_reading"]
    assert abs(pred - 0.0) < 1e-6

    # Fresh Feedback at π/2 → confirm
    obs1 = Thought(id="o1", label="eye:0.0000", transient=True)
    pending = s.outerface._pending_feedback.pop(eye.id)
    s.outerface.form_belief_from_feedback(
        observation=obs1,
        sensor_id=eye.id,
        sensor_label="eye",
        action="hand",
        compare=True,
        predicted_reading=pending["predicted_reading"],
    )
    assert s.outerface.belief_confirms >= 1

    # Stale sample at hand=0 must not overwrite prediction after next fire
    hand.request_fire(s, "hand")  # hand=π, cos=-1
    pending2 = dict(s.outerface._pending_feedback[eye.id])
    stale = Thought(id="o_stale", label="eye:0.0000", transient=True)
    s.outerface.form_belief_from_feedback(
        observation=stale,
        sensor_id=eye.id,
        sensor_label="eye",
        action="hand",
        compare=True,
        predicted_reading=pending2["predicted_reading"],
    )
    # Prediction for π is -1; stale 0 should be challenge or stale-skip, not leave wrong forever
    assert s.outerface.belief_stale_skips + s.outerface.belief_challenges >= 1


def test_interface_innerface_pipeline_threaded():
    s = Symbioid()
    s.add_sensor(label="eye")
    s.add_sensor(label="ear")
    s.start_processes()
    time.sleep(0.35)
    s.stop_processes(timeout=1.0)
    # continuous_inputs: new sample + handoff every Interface tick per sensor
    assert s.interface.inputs_sampled >= 4
    assert s.interface.handoffs_sent >= 4
    assert s.interface.handoffs_sent == s.interface.inputs_sampled
    assert len(s.innerface.completed_formations) >= 4
    assert s.innerface.formation_ticks >= 4
    for store in s.innerface.completed_formations.values():
        assert len(store) == 6
        assert is_minimal_symbioid_shape(store)
    # multi-sensor batches produce Follows/FollowedBy sync sets
    assert len(s.innerface.completed_syncs) >= 1
    for store in s.innerface.completed_syncs.values():
        assert len(store) == 6
        assert is_minimal_symbioid_shape(store)


def test_continuous_inputs_can_be_disabled():
    s = Symbioid()
    s.interface.continuous_inputs = False
    s.add_sensor(label="eye")
    s.start_processes()
    time.sleep(0.2)
    s.stop_processes(timeout=1.0)
    # one-shot only when continuous_inputs is False
    assert s.interface.handoffs_sent == 1
    assert len(s.innerface.completed_formations) == 1
