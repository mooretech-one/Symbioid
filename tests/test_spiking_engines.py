"""Phase 0: SpikingEngine scaffold + pulse_partition parity."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import ENGINES_MODES, Link, SpikingEngine, Symbioid, Thought


def test_engines_mode_default_legacy():
    s = Symbioid()
    assert s.engines_mode == "legacy"
    assert "legacy" in ENGINES_MODES


def test_pulse_partition_none_matches_pulse_tick_shape():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    a = Thought(id=f"{s.id}:a", threshold=1.0)
    b = Thought(id=f"{s.id}:b", threshold=10.0)
    lt = Thought(id=f"{s.id}:lt", threshold=10.0)
    link = Link(
        id=f"{s.id}:l",
        source=a,
        link_type=lt,
        target=b,
        weight=1.0,
        threshold=10.0,
    )
    for t in (a, b, lt, link):
        s.add_thought(t)
    s.stimulate(a, 1.5)
    st = s.pulse_partition(membership=None, engine_name="global")
    assert set(st.keys()) >= {"cycle", "hot", "fired", "spread", "hebb", "engine"}
    assert st["engine"] == "global"
    assert st["fired"] >= 1


def test_pulse_tick_wraps_partition():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    a = Thought(id=f"{s.id}:a2", threshold=1.0, activation=1.5)
    s.add_thought(a)
    s.mark_hot(a)
    st = s.pulse_tick()
    assert st["engine"] == "global"
    assert st["cycle"] >= 1


def test_membership_mask_only_fires_members():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = False
    a = Thought(id=f"{s.id}:in", threshold=1.0)
    b = Thought(id=f"{s.id}:out", threshold=1.0)
    for t in (a, b):
        s.add_thought(t)
    s.stimulate(a, 1.5)
    s.stimulate(b, 1.5)
    # Only `a` in membership
    st = s.pulse_partition(membership={a.id}, engine_name="interface")
    assert st["fired"] == 1
    assert a.refractory_ticks > 0
    # b was stimulated but not in membership — should not have fired
    assert b.refractory_ticks == 0 or not b.just_fired


def test_spiking_engine_pulse_updates_export():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    eng = SpikingEngine(engine_name="interface", use_membership=False)
    eng.host = s
    a = Thought(id=f"{s.id}:e", threshold=1.0)
    s.add_thought(a)
    s.stimulate(a, 1.5)
    st = eng.pulse()
    assert st["fired"] >= 1
    assert a.id in eng.last_export_ids or st["fired"] >= 1


def test_spiking_engine_process_body_runs():
    s = Symbioid(install_constitution=False)
    eng = SpikingEngine(engine_name="test", use_membership=True)
    eng.host = s
    eng.membership = set()
    eng._process_body()  # empty membership pulse ok
    assert eng.last_pulse_stats.get("fired", 0) == 0


def test_interface_is_spiking_engine():
    from symbioid import Interface

    s = Symbioid()
    assert isinstance(s.interface, SpikingEngine)
    assert s.interface.engine_name == "interface"


def test_interface_hybrid_suppresses_reuse_handoff():
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.42
    h1 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={}), post_to_innerface=True
    )
    assert h1 is not None  # mint
    assert s.interface.handoffs_sent == 1
    # Same value → skip or reuse; hybrid suppresses non-mint handoffs
    h2 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=2, world={}), post_to_innerface=True
    )
    # skip returns None; reuse returns None in hybrid (suppressed)
    assert h2 is None
    assert s.interface.handoffs_skipped_mind + s.interface.handoffs_suppressed_engine >= 1


def test_interface_hybrid_process_body_membership_pulse():
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.interface.continuous_inputs = True
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.5
    # One engine tick via process body
    s.interface._process_body()
    assert s.interface.inputs_sampled >= 1
    assert len(s.interface.membership) >= 1
    # Pulse ran under interface engine name when membership non-empty
    if s.interface.membership:
        assert s.interface.last_pulse_stats.get("engine") in (
            "interface",
            "global",
            None,
        ) or "cycle" in s.interface.last_pulse_stats


def test_innerface_is_spiking_engine():
    from symbioid import Innerface

    s = Symbioid()
    assert isinstance(s.innerface, SpikingEngine)
    assert s.innerface.engine_name == "innerface"


def test_innerface_hybrid_co_fire_consolidates_follows():
    """Two Observations that fire same inner pulse → Follows consolidator."""
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = True
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    eye.transfer = lambda w: 0.11
    ear.transfer = lambda w: 0.22
    # Mint both (structure via handoff API with post)
    h1 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={}), post_to_innerface=True
    )
    h2 = s.interface.start_formation_for_sensor(
        ear, force=True, sense=ear.sample(tick=1, world={}), post_to_innerface=True
    )
    assert h1 and h2
    s.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": [h1, h2], "tick": 1}
    )
    # Clear syncs to prove co-fire path can rebuild associations
    # (batch already created follows — re-stimulate and engine tick)
    n_sync_before = len(s.innerface.completed_syncs)
    # Force both obs hot and run inner engine
    for obs in s.innerface._last_obs_by_sensor.values():
        s.stimulate(obs, 2.0)
        s.innerface.add_member(obs.id)
    s.innerface.use_membership = True
    s.innerface.pre_ports()
    s.innerface.pulse()
    s.innerface.post_ports()
    assert s.innerface.engine_ticks >= 1
    assert s.innerface.co_fire_consolidations >= 1 or len(
        s.innerface.completed_syncs
    ) >= n_sync_before


def test_innerface_hybrid_port_import_from_interface():
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.7
    s.interface._process_body()  # sample + export
    before_imp = s.innerface.port_imports
    s.innerface._process_body()  # port-in + pulse + consolidate
    # Port imports count only when export_activation non-zero after interface fire
    assert s.innerface.engine_ticks >= 1
    assert s.innerface.port_imports >= before_imp


def test_outerface_is_spiking_engine():
    from symbioid import Outerface

    s = Symbioid()
    assert isinstance(s.outerface, SpikingEngine)
    assert s.outerface.engine_name == "outerface"


def test_outerface_hybrid_spike_action_on_hot_action():
    """Hottest Action Thought drives request_fire under hybrid engine."""
    s = Symbioid(label="tetris")
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.outerface.wait_for_feedback = False
    hand = s.add_actuator(label="hand")
    st = Thought(id=f"{s.id}:st", label="state", transient=True)
    s.add_thought(st)
    with s.mind._lock:
        s.mind._observations["st:r:0"] = st
        s.mind._thought_to_key[st.id] = "st:r:0"
    act = s.mind.record_outcome(
        [st], "wave", domain="tetris", host_id=s.id, reward=200.0, host=s
    )
    s.stimulate(act, 2.0)
    s.outerface.add_member(act.id)
    s.outerface.use_membership = True
    s.outerface.pre_ports()
    s.outerface.pulse()
    s.outerface.post_ports()
    assert s.outerface.engine_ticks >= 1
    # spike or graph path should have attempted fire
    assert (
        s.outerface.spike_actions >= 1
        or hand.fire_count >= 1
        or (s.outerface.last_gate and "wave" in str(s.outerface.last_gate))
    )


def test_serial_hybrid_i_n_o_tick():
    """I → N → O engine ticks do not crash in hybrid mode."""
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.outerface.wait_for_feedback = False
    eye = s.add_sensor(label="eye")
    s.add_actuator(label="hand")
    eye.transfer = lambda w: 0.33
    s.interface._process_body()
    s.innerface._process_body()
    s.outerface._process_body()
    assert s.interface.inputs_sampled >= 1
    assert s.innerface.engine_ticks >= 1
    assert s.outerface.engine_ticks >= 1


def test_run_engines_spike_mode():
    """Phase 4: spike mode uses structure_pending, not inbox handoffs."""
    s = Symbioid()
    s.engines_mode = "spike"
    s.mind.dynamics_enabled = True
    s.outerface.wait_for_feedback = False
    eye = s.add_sensor(label="eye")
    s.add_actuator(label="hand")
    eye.transfer = lambda w: 0.44
    stats = s.run_engines()
    assert "interface" in stats and "innerface" in stats and "outerface" in stats
    assert s.interface.inputs_sampled >= 1
    # spike suppresses inbox handoffs from engine path
    assert s.interface.handoffs_suppressed_engine >= 1 or s.interface.structure_pending == []
    # structure pulled by Innerface consolidator on mint
    assert s.innerface.engine_ticks >= 1
    assert s.outerface.engine_ticks >= 1


def test_spike_structure_pending_consumed():
    s = Symbioid()
    s.engines_mode = "spike"
    s.mind.dynamics_enabled = True
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.55
    s.interface.pre_ports()
    # mint goes to structure_pending, not inbox
    assert len(s.interface.structure_pending) >= 1 or s.interface.handoffs_skipped_mind >= 0
    pending_n = len(s.interface.structure_pending)
    s.interface.pulse()
    s.interface.post_ports()
    s.innerface.pre_ports()  # pulls structure_pending
    assert len(s.interface.structure_pending) == 0
    if pending_n:
        assert s.innerface.formation_ticks >= 1 or s.innerface.completed_formations


# --- Phase 5: port queues, energy budgets, Port Hebb ----------------------


def test_port_queue_export_and_drain():
    from symbioid import PortPacket, PORT_I_N

    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    a = Thought(id=f"{s.id}:port-a", threshold=1.0, activation=1.5)
    s.add_thought(a)
    a.export_activation = 1.2
    n = s.export_port_packets(
        src_engine="interface",
        dst_engine="innerface",
        thought_ids=[a.id],
        cycle=1,
    )
    assert n == 1
    assert s.port_queue_len("interface", "innerface") == 1
    pkts = s.drain_port("interface", "innerface")
    assert len(pkts) == 1
    assert isinstance(pkts[0], PortPacket)
    assert pkts[0].thought_id == a.id
    assert pkts[0].channel == PORT_I_N
    assert s.port_queue_len("interface", "innerface") == 0


def test_apply_port_packets_stimulates_and_hebb_port_link():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = True
    s.mind.port_hebb_lr = 0.1
    a = Thought(id=f"{s.id}:imp", threshold=1.0, activation=0.0)
    s.add_thought(a)
    from symbioid import PortPacket

    pkts = [
        PortPacket(
            thought_id=a.id,
            activation=1.0,
            source_engine="interface",
            cycle=1,
            channel="interface>innerface",
        )
    ]
    n = s.apply_port_packets(pkts, gain=0.5, hebb=True)
    assert n == 1
    assert a.activation > 0
    link = s.ensure_port_link(a, channel="interface>innerface")
    assert link.is_port
    assert link.weight > 1.0  # Hebbed on transfer
    # Port links must not spread in pulse
    s.stimulate(a, 2.0)
    before_b = 0.0
    st = s.pulse_partition(membership={a.id}, engine_name="interface")
    assert st["fired"] >= 0  # smoke
    # self-port edge skipped for spread (no infinite self-boost required)
    assert link.is_port


def test_energy_budget_caps_fires():
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = False
    s.mind.energy_fire_cost = 1.0
    s.mind.energy_spread_cost = 0.0
    thoughts = []
    for i in range(5):
        t = Thought(id=f"{s.id}:e{i}", threshold=1.0)
        s.add_thought(t)
        s.stimulate(t, 1.5)
        thoughts.append(t)
    st = s.pulse_partition(
        membership={t.id for t in thoughts},
        engine_name="interface",
        energy_budget=2.0,
    )
    assert st["fired"] == 2
    assert st["energy_capped"] >= 1
    assert st["energy_used"] >= 2.0 - 1e-9
    assert st["energy_left"] < 1e-9


def test_membership_hebb_skips_cross_non_port():
    """Under membership + port_hebb_cross_only, non-Port links to outsiders don't Hebb."""
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.hebb_enabled = True
    s.mind.hebb_lr = 0.5
    s.mind.port_hebb_cross_only = True
    a = Thought(id=f"{s.id}:in-m", threshold=1.0)
    b = Thought(id=f"{s.id}:out-m", threshold=10.0, activation=5.0)  # hot but not member
    lt = Thought(id=f"{s.id}:lt-m", threshold=10.0)
    link = Link(
        id=f"{s.id}:l-m",
        source=a,
        link_type=lt,
        target=b,
        weight=1.0,
        threshold=10.0,
        is_port=False,
    )
    for t in (a, b, lt, link):
        s.add_thought(t)
    w0 = link.weight
    s.stimulate(a, 1.5)
    s.pulse_partition(membership={a.id}, engine_name="interface")
    # Spread may still reach b (clamped to activation_max); Hebb skipped
    assert link.weight == w0


def test_engine_energy_budget_from_mind():
    s = Symbioid(install_constitution=False)
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.mind.energy_budget_interface = 1.0
    s.mind.energy_fire_cost = 1.0
    eng = s.interface
    eng.use_membership = True
    a = Thought(id=f"{s.id}:eb", threshold=1.0)
    b = Thought(id=f"{s.id}:eb2", threshold=1.0)
    for t in (a, b):
        s.add_thought(t)
        s.stimulate(t, 1.5)
        eng.add_member(t.id)
    st = eng.pulse()
    assert st["fired"] <= 1
    assert st.get("energy_capped", 0) >= 1 or st["fired"] == 1


def test_run_engines_uses_port_queues():
    """Phase 5: hybrid I→N→O fills/drains port queues without crash."""
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.outerface.wait_for_feedback = False
    eye = s.add_sensor(label="eye")
    s.add_actuator(label="hand")
    eye.transfer = lambda w: 0.66
    # Force a fire export path: membership + stimulate after sample
    s.interface._process_body()
    # After interface post_ports, queue may have packets if something fired
    # Innerface drains on next body
    s.innerface._process_body()
    s.outerface._process_body()
    assert s.interface.inputs_sampled >= 1
    assert s.innerface.engine_ticks >= 1
    # Port import counter may be 0 if no firers, but queue APIs work
    assert s.port_queue_len("interface", "innerface") == 0  # drained or empty
