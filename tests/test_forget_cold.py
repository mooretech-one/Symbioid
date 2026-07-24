"""Activation-based forgetting (cold unprotected Thoughts)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Symbioid, Thought


def test_forget_cold_enabled_by_default():
    s = Symbioid(install_constitution=False)
    assert s.mind.forget_cold_enabled is True


def test_forget_cold_opt_out():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = False
    # :obs: prefix is not twin-seed protected; not Mind-registered → forgettable when on
    t = Thought(id=f"{s.id}:obs:cold-a", transient=True, threshold=1.0)
    s.add_thought(t)
    s.stimulate(t, 1.5)
    t.activation = 0.0
    t.refractory_ticks = 0
    # age artificially
    t.last_hot_cycle = 0
    s.pulse_cycle = 100
    n = s.innerface.forget_cold_thoughts()
    assert n == 0
    assert t.id in s.thoughts


def test_forget_cold_removes_unprotected_transient():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = True
    s.mind.forget_cold_cycles = 2
    s.mind.forget_transient_only = True
    t = Thought(id=f"{s.id}:obs:cold-b", transient=True, threshold=1.0)
    s.add_thought(t)
    s.stimulate(t, 1.5)
    assert t.last_hot_cycle >= 0
    t.activation = 0.0
    t.resting = 0.0
    t.refractory_ticks = 0
    t.last_hot_cycle = 0
    s.pulse_cycle = 5  # 5 - 0 >= 2
    n = s.innerface.forget_cold_thoughts()
    assert n >= 1
    assert t.id not in s.thoughts
    assert s.mind.forgets_cold >= 1
    assert s.innerface.thoughts_forgotten >= 1


def test_forget_cold_protects_mind_registry():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = True
    s.mind.forget_cold_cycles = 1
    s.mind.recognition_enabled = True
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.123
    h = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={}), post_to_innerface=True
    )
    assert h is not None
    s.innerface.accept_formation(h)
    # Find a registered observation
    with s.mind._lock:
        obs_ids = {t.id for t in s.mind._observations.values()}
    assert obs_ids
    for oid in obs_ids:
        t = s.thoughts.get(oid)
        if t is None:
            continue
        t.activation = 0.0
        t.refractory_ticks = 0
        t.last_hot_cycle = 0
    s.pulse_cycle = 50
    s.innerface.forget_cold_thoughts()
    for oid in obs_ids:
        assert oid in s.thoughts, f"registered {oid} was cold-forgotten"


def test_forget_cold_default_transient_only_is_false():
    s = Symbioid(install_constitution=False)
    assert s.mind.forget_transient_only is False


def test_forget_cold_transient_only_skips_stable():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = True
    s.mind.forget_cold_cycles = 1
    s.mind.forget_transient_only = True  # opt-in restriction
    # non-transient even under :obs: should stay when transient_only
    stable = Thought(id=f"{s.id}:obs:stable", transient=False, threshold=1.0)
    s.add_thought(stable)
    s.stimulate(stable, 1.0)
    stable.activation = 0.0
    stable.refractory_ticks = 0
    stable.last_hot_cycle = 0
    s.pulse_cycle = 10
    s.innerface.forget_cold_thoughts()
    assert stable.id in s.thoughts


def test_forget_cold_non_transient_when_flag_false():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = True
    s.mind.forget_cold_cycles = 1
    s.mind.forget_transient_only = False
    stable = Thought(id=f"{s.id}:obs:stable2", transient=False, threshold=1.0)
    s.add_thought(stable)
    s.stimulate(stable, 1.0)
    stable.activation = 0.0
    stable.refractory_ticks = 0
    stable.last_hot_cycle = 0
    s.pulse_cycle = 10
    n = s.innerface.forget_cold_thoughts()
    assert n >= 1
    assert stable.id not in s.thoughts


def test_forget_cold_never_hot_skipped():
    s = Symbioid(install_constitution=False)
    s.mind.forget_cold_enabled = True
    s.mind.forget_cold_cycles = 1
    t = Thought(id=f"{s.id}:obs:never", transient=True, threshold=1.0)
    s.add_thought(t)
    assert t.last_hot_cycle == -1
    s.pulse_cycle = 100
    n = s.innerface.forget_cold_thoughts()
    assert n == 0
    assert t.id in s.thoughts


def test_stimulate_sets_last_hot_cycle():
    s = Symbioid(install_constitution=False)
    s.pulse_cycle = 7
    t = Thought(id=f"{s.id}:obs:hot", transient=True, threshold=1.0)
    s.add_thought(t)
    s.stimulate(t, 0.5)
    assert t.last_hot_cycle == 7


def test_maybe_gc_smoke_hybrid():
    s = Symbioid()
    s.engines_mode = "hybrid"
    s.mind.dynamics_enabled = True
    s.mind.forget_cold_enabled = False
    s.outerface.wait_for_feedback = False
    eye = s.add_sensor(label="eye")
    s.add_actuator(label="hand")
    eye.transfer = lambda w: 0.2
    s.run_engines()
    assert s.innerface.engine_ticks >= 1
