"""Agent memory persistence — lean registry + Links (no game fields)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import (
    Link,
    Symbioid,
    Thought,
    apply_memory,
    export_memory,
    load_memory,
    save_memory,
    try_load_into,
)
from symbioid.persist import FORBIDDEN_TOP_KEYS, FORMAT, VERSION


HOST = "sym-test-persist"


def _host() -> Symbioid:
    return Symbioid(id=HOST, install_constitution=False)


def test_export_has_no_game_fields():
    s = _host()
    data = export_memory(s)
    assert data["format"] == FORMAT
    assert data["version"] == VERSION
    assert data.get("snapshot") == "lean"
    for k in FORBIDDEN_TOP_KEYS:
        assert k not in data
    assert "mind" in data and "thoughts" in data
    assert "actions" in data["mind"]


def test_lean_skips_unregistered_scaffold():
    s = _host()
    act = s.mind.ensure_action_thought("tetris", "left", host_id=s.id)
    s.add_thought(act)
    junk = Thought(id=f"{s.id}:form:scaffold-junk", transient=True, label="ghost")
    s.add_thought(junk)
    n_before = len(s.thoughts)
    data = export_memory(s, mode="lean")
    ids = {r["id"] for r in data["thoughts"]}
    assert act.id in ids
    assert junk.id not in ids
    assert len(data["thoughts"]) < n_before


def test_lean_includes_incident_link_weight():
    s = _host()
    act = s.mind.ensure_action_thought("tetris", "left", host_id=s.id)
    s.add_thought(act)
    st = Thought(id=f"{s.id}:obs:state", label="state", transient=True)
    s.add_thought(st)
    with s.mind._lock:
        s.mind._observations["st:r:0"] = st
        s.mind._thought_to_key[st.id] = "st:r:0"
        s.mind._valence["st:r:0"] = 1.5
    lt = Thought(id=f"{s.id}:lt", threshold=10.0, dynamics_enabled=False)
    s.add_thought(lt)
    edge = Link(
        id=f"{s.id}:edge",
        source=st,
        link_type=lt,
        target=act,
        weight=2.5,
        threshold=10.0,
    )
    s.add_thought(edge)
    # Unrelated scaffold link (not touching registry) should be dropped
    other = Thought(id=f"{s.id}:form:other", transient=True)
    s.add_thought(other)
    lt2 = Thought(id=f"{s.id}:lt2", threshold=10.0, dynamics_enabled=False)
    s.add_thought(lt2)
    junk_edge = Link(
        id=f"{s.id}:junk-edge",
        source=other,
        link_type=lt2,
        target=other,
        weight=9.0,
        threshold=10.0,
    )
    s.add_thought(junk_edge)

    data = export_memory(s, mode="lean")
    ids = {r["id"] for r in data["thoughts"]}
    assert edge.id in ids
    assert lt.id in ids
    assert junk_edge.id not in ids
    assert other.id not in ids


def test_round_trip_action_command_key_and_link_weight(tmp_path: Path):
    s = _host()
    act = s.mind.ensure_action_thought("tetris", "left", host_id=s.id)
    s.add_thought(act)
    st = Thought(id=f"{s.id}:obs:state", label="state", transient=True)
    s.add_thought(st)
    with s.mind._lock:
        s.mind._observations["st:r:0"] = st
        s.mind._thought_to_key[st.id] = "st:r:0"
        s.mind._valence["st:r:0"] = 1.5
    lt = Thought(id=f"{s.id}:lt", threshold=10.0, dynamics_enabled=False)
    s.add_thought(lt)
    edge = Link(
        id=f"{s.id}:edge",
        source=st,
        link_type=lt,
        target=act,
        weight=2.5,
        threshold=10.0,
    )
    s.add_thought(edge)

    path = tmp_path / "mem.json"
    save_memory(s, path)
    raw = json.loads(path.read_text())
    for k in FORBIDDEN_TOP_KEYS:
        assert k not in raw
    assert raw.get("snapshot") == "lean"
    assert "act:tetris:left" in raw["mind"]["actions"]

    s2 = _host()
    # Shell (seed) still present after lean merge
    n_shell = len(s2.thoughts)
    assert try_load_into(s2, path)
    assert len(s2.thoughts) >= n_shell  # merge, not wipe
    assert "act:tetris:left" in s2.mind._actions
    assert s2.mind._actions["act:tetris:left"].id == act.id
    restored = s2.thoughts.get(edge.id)
    assert isinstance(restored, Link)
    assert abs(restored.weight - 2.5) < 1e-9
    assert s2.mind._valence.get("st:r:0") == 1.5


def test_full_mode_includes_junk(tmp_path: Path):
    s = _host()
    junk = Thought(id=f"{s.id}:form:scaffold-junk", transient=True)
    s.add_thought(junk)
    data = export_memory(s, mode="full")
    ids = {r["id"] for r in data["thoughts"]}
    assert junk.id in ids
    assert data.get("snapshot") == "full"


def test_missing_file_fail_open(tmp_path: Path):
    s = _host()
    assert try_load_into(s, tmp_path / "nope.json") is False


def test_corrupt_file_fail_open(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    s = _host()
    assert load_memory(p) is None
    assert try_load_into(s, p) is False


def test_host_id_mismatch_refuses(tmp_path: Path):
    s = _host()
    path = tmp_path / "m.json"
    save_memory(s, path)
    other = Symbioid(id="sym-other", install_constitution=False)
    assert try_load_into(other, path) is False


def test_activation_reset_on_load(tmp_path: Path):
    s = _host()
    t = Thought(id=f"{s.id}:obs:hot", transient=True, activation=2.0, threshold=1.0)
    s.add_thought(t)
    with s.mind._lock:
        s.mind._observations["hot:r:1"] = t
        s.mind._thought_to_key[t.id] = "hot:r:1"
    path = tmp_path / "a.json"
    save_memory(s, path)
    s2 = _host()
    assert try_load_into(s2, path, reset_activation=True)
    t2 = s2.thoughts[t.id]
    assert abs(t2.activation - t2.resting) < 1e-9


def test_apply_rejects_game_keys():
    s = _host()
    data = export_memory(s)
    data["coach"] = {"bad": True}
    try:
        apply_memory(s, data)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "game" in str(e).lower() or "coach" in str(e).lower()


def test_v1_payload_still_loads():
    """Version 1 full dumps remain loadable."""
    s = _host()
    data = export_memory(s, mode="full")
    data["version"] = 1
    data.pop("snapshot", None)
    s2 = _host()
    apply_memory(s2, data)
