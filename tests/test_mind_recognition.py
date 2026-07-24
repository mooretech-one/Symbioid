"""Mind recognition / habituation — Thought growth under constant I/O."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Mind, Symbioid
from symbioid.Core.Mind import AdmitResult


def _sense(sensor_id: str, reading: float, tick: int = 0) -> dict:
    return {
        "kind": "input",
        "sensor_id": sensor_id,
        "sensor_label": "eye",
        "tick": tick,
        "sample": tick,
        "value": f"eye:{reading:.4f}",
        "reading": reading,
    }


def test_mind_content_key_quantizes_float():
    m = Mind(quantize_decimals=3)
    k1 = m.content_key("eye", {"reading": 0.1234})
    k2 = m.content_key("eye", {"reading": 0.1231})
    k3 = m.content_key("eye", {"reading": 0.124})
    assert k1 == k2
    assert k1 != k3


def test_mind_identical_input_mints_once_then_skips():
    m = Mind(habituate_after=2, recognition_enabled=True)
    host = "sym-test"
    a1 = m.admit_input("eye", _sense("eye", 0.5, 1), host_id=host)
    assert a1.action == "mint"
    assert a1.observation is not None
    a2 = m.admit_input("eye", _sense("eye", 0.5, 2), host_id=host)
    # streak=2 >= habituate_after → skip (no growth)
    assert a2.action == "skip"
    assert a2.observation_id == a1.observation_id
    a3 = m.admit_input("eye", _sense("eye", 0.5, 3), host_id=host)
    assert a3.action == "skip"
    assert m.admits_mint == 1
    assert m.admits_skip >= 2


def test_mind_return_to_known_key_reuses():
    m = Mind(habituate_after=3, recognition_enabled=True)
    host = "sym-test"
    a = m.admit_input("eye", _sense("eye", 0.1, 1), host_id=host)
    assert a.action == "mint"
    b = m.admit_input("eye", _sense("eye", 0.9, 2), host_id=host)
    assert b.action == "mint"
    # back to first key, streak=1 < 3 → reuse
    c = m.admit_input("eye", _sense("eye", 0.1, 3), host_id=host)
    assert c.action == "reuse"
    assert c.observation_id == a.observation_id
    assert m.admits_reuse == 1


def test_mind_disabled_preserves_legacy_mint():
    m = Mind(recognition_enabled=False)
    a = m.admit_input("eye", _sense("eye", 0.5, 1), host_id="h")
    b = m.admit_input("eye", _sense("eye", 0.5, 2), host_id="h")
    assert a.action == "mint" and b.action == "mint"
    assert a.observation_id != b.observation_id
    assert m.admits_mint == 2


def test_interface_growth_sublinear_same_reading():
    """N identical samples → Thought count ≪ 6N (recognition)."""
    s = Symbioid()
    eye = s.add_sensor(label="eye")
    # Fixed transfer so every sample is the same reading
    eye.transfer = lambda w: 0.42
    n = 40
    before = len(s.thoughts)
    handoffs = 0
    for t in range(n):
        sense = eye.sample(tick=t, world={})
        h = s.interface.start_formation_for_sensor(eye, force=True, sense=sense)
        if h is not None:
            handoffs += 1
            s.innerface.accept_formation(h, integrate_temporal=False)
    after = len(s.thoughts)
    grown = after - before
    # Without recognition, each sample adds ~6 Thoughts → ~240
    assert handoffs <= 2, f"expected ≤2 handoffs after habit, got {handoffs}"
    assert grown < 30, f"expected bounded growth, grown={grown} handoffs={handoffs}"
    assert s.mind.admits_mint == 1
    assert s.mind.admits_skip >= n - 2


def test_innerface_reuse_does_not_inflate_active():
    s = Symbioid()
    s.mind.habituate_after = 5  # allow reuse path before skip
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.7
    h1 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={})
    )
    assert h1 is not None
    s.innerface.accept_formation(h1, integrate_temporal=False)
    active1 = s.innerface.active_set_count
    n1 = len(s.thoughts)
    # Different value → mint
    eye.transfer = lambda w: 0.1
    h2 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=2, world={})
    )
    s.innerface.accept_formation(h2, integrate_temporal=False)
    # Back to 0.7 → reuse
    eye.transfer = lambda w: 0.7
    h3 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=3, world={})
    )
    assert h3 is not None
    assert h3.get("reused") is True
    s.innerface.accept_formation(h3, integrate_temporal=False)
    n3 = len(s.thoughts)
    # Reuse must not add a full new six-set
    assert n3 - n1 < 12
    assert s.mind.admits_reuse >= 1
    # Active set should not explode from reuse
    assert s.innerface.active_set_count <= active1 + 3


def test_valence_protects_registered_observations():
    s = Symbioid()
    eye = s.add_sensor(label="eye")
    eye.transfer = lambda w: 0.55
    h = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={})
    )
    store = s.innerface.accept_formation(h, integrate_temporal=False)
    obs_ids = s.mind.registered_observation_ids()
    assert obs_ids
    s.mind.note_valence(channel="board", delta=2.0)
    # Force deactivate sense set and prune
    with s.innerface._local_lock:
        s.innerface.active_ids.clear()
    removed = s.innerface.prune_inactive_thoughts()
    for oid in obs_ids:
        assert oid in s.thoughts, f"registered obs {oid} was pruned ({removed=})"


def test_admit_result_type():
    m = Mind()
    r = m.admit_input("x", _sense("x", 0.0), host_id="h")
    assert isinstance(r, AdmitResult)


def test_follows_content_key_undirected():
    m = Mind()
    # Register two poles so pair keys use content keys
    a = m.admit_input("eye", _sense("eye", 0.1), host_id="h")
    b = m.admit_input("ear", _sense("ear", 0.2), host_id="h")
    assert a.observation and b.observation
    k1 = m.follows_content_key(a.observation, b.observation)
    k2 = m.follows_content_key(b.observation, a.observation)
    assert k1 == k2
    assert k1.startswith("follows:")


def test_follows_mint_once_then_skip():
    m = Mind(habituate_after=2, recognition_enabled=True)
    a = m.admit_input("eye", _sense("eye", 0.3), host_id="h").observation
    b = m.admit_input("ear", _sense("ear", 0.4), host_id="h").observation
    assert a and b
    f1 = m.admit_follows(a, b, host_id="h")
    assert f1.action == "mint" and f1.kind == "follows"
    assert f1.formation_id and f1.formation_id.startswith("h:sync:")
    f2 = m.admit_follows(a, b, host_id="h")
    assert f2.action == "skip"
    assert f2.formation_id == f1.formation_id
    assert m.follows_mint == 1
    assert m.follows_skip >= 1


def test_integrates_content_key_includes_channel():
    m = Mind()
    a = m.admit_input("eye", _sense("eye", 0.1), host_id="h").observation
    b = m.admit_input("eye", _sense("eye", 0.9), host_id="h").observation
    assert a and b
    k1 = m.integrates_content_key(a, b, channel="eye")
    k2 = m.integrates_content_key(b, a, channel="eye")
    k3 = m.integrates_content_key(a, b, channel="ear")
    assert k1 == k2
    assert k1 != k3
    assert k1.startswith("int:eye:")


def test_integrates_mint_once_then_skip():
    m = Mind(habituate_after=2, recognition_enabled=True)
    a = m.admit_input("eye", _sense("eye", 0.2), host_id="h").observation
    b = m.admit_input("eye", _sense("eye", 0.8), host_id="h").observation
    assert a and b
    i1 = m.admit_integrates(a, b, host_id="h", channel="eye")
    assert i1.action == "mint" and i1.kind == "integrates"
    assert i1.formation_id and ":int:" in i1.formation_id
    i2 = m.admit_integrates(a, b, host_id="h", channel="eye")
    assert i2.action == "skip"
    assert i2.formation_id == i1.formation_id
    assert m.integrates_mint == 1
    assert m.integrates_skip >= 1


def test_temporal_integrate_does_not_remint_same_pair():
    """Same sensor two values integrate once; re-integrate returns same set."""
    s = Symbioid()
    eye = s.add_sensor(label="eye")
    # Two distinct readings so both mint as Observations
    eye.transfer = lambda w: 0.1
    h1 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={})
    )
    s.innerface.accept_formation(h1)
    eye.transfer = lambda w: 0.9
    h2 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=2, world={})
    )
    s.innerface.accept_formation(h2)
    assert s.mind.integrates_mint == 1
    n_int = len(s.innerface.completed_integrates)
    assert n_int == 1
    with s.innerface._local_lock:
        poles = list(s.innerface._last_obs_by_sensor.values())
    # Force another integrate attempt of the same two poles
    # Need both poles — last is only one; get from completed integrate
    store = next(iter(s.innerface.completed_integrates.values()))
    from symbioid.Core.formation import six_set_poles

    pa, pb = six_set_poles(store)[:2]
    s.innerface.integrate_pair(pa, pb, reason="again", channel=eye.id)
    s.innerface.integrate_pair(pa, pb, reason="again2", channel=eye.id)
    assert len(s.innerface.completed_integrates) == 1
    assert s.mind.integrates_mint == 1
    assert s.mind.integrates_skip + s.mind.integrates_reuse >= 1


def test_batch_follows_does_not_remint_same_cooccurrence():
    """Same multi-sensor values → one Follows sync; re-sync skips remint."""
    s = Symbioid()
    eye = s.add_sensor(label="eye")
    ear = s.add_sensor(label="ear")
    eye.transfer = lambda w: 0.11
    ear.transfer = lambda w: 0.22
    h1 = s.interface.start_formation_for_sensor(
        eye, force=True, sense=eye.sample(tick=1, world={})
    )
    h2 = s.interface.start_formation_for_sensor(
        ear, force=True, sense=ear.sample(tick=1, world={})
    )
    s.innerface.accept_formation_batch(
        {"kind": "formation_batch", "handoffs": [h1, h2], "tick": 1}
    )
    assert len(s.innerface.completed_syncs) == 1
    assert s.mind.follows_mint == 1
    # Re-present same Observation poles (as if another co-occurrence tick)
    with s.innerface._local_lock:
        poles = list(s.innerface._last_obs_by_sensor.values())
    assert len(poles) == 2
    s.innerface.synchronize_observations(poles, tick=2, integrate_pairs=False)
    s.innerface.synchronize_observations(poles, tick=3, integrate_pairs=False)
    assert len(s.innerface.completed_syncs) == 1
    assert s.mind.follows_mint == 1
    assert s.mind.follows_skip + s.mind.follows_reuse >= 1
