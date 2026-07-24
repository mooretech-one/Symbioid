"""Minted Thoughts drive behavior: record_outcome → recommend_action."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Mind, Symbioid, Thought
from symbioid.Core.Mind import RecommendResult


def test_ensure_action_stable_id():
    m = Mind()
    a = m.ensure_action_thought("tetris", "left", host_id="h")
    b = m.ensure_action_thought("tetris", "left", host_id="h")
    assert a.id == b.id
    assert a.id.startswith("h:act:")
    assert m.actions_mint == 1


def test_record_outcome_mints_integrate_once():
    m = Mind()
    st = Thought(id="h:obs:state1", label="eye:0.5", transient=True)
    # Register as observation pole so pole_content_key is stable
    with m._lock:
        m._observations["eye:r:0.5"] = st
        m._thought_to_key[st.id] = "eye:r:0.5"
    m.record_outcome([st], "hard", domain="tetris", host_id="h", reward=100.0)
    m.record_outcome([st], "hard", domain="tetris", host_id="h", reward=80.0)
    assert m.actions_mint == 1
    assert m.outcomes_recorded == 2
    # Follows + policy integrate associations present
    assert any(k.startswith("follows:") for k in m._follows)
    assert any("policy" in k and "act:tetris:hard" in k for k in m._integrates)


def test_recommend_prefers_high_valence_action():
    m = Mind(recommend_min_valence=0.1)
    st = Thought(id="h:obs:s", label="holes:0.1", transient=True)
    with m._lock:
        m._observations["holes:r:0.1"] = st
        m._thought_to_key[st.id] = "holes:r:0.1"
    # Train: hard is good, left is bad
    m.record_outcome([st], "hard", domain="tetris", host_id="h", reward=200.0)
    m.record_outcome([st], "left", domain="tetris", host_id="h", reward=-100.0)
    rec = m.recommend_action([st], domain="tetris")
    assert isinstance(rec, RecommendResult)
    assert rec.token == "hard"
    assert rec.score >= m.recommend_min_valence


def test_recommend_none_when_cold():
    m = Mind()
    st = Thought(id="h:obs:cold", label="x:0", transient=True)
    with m._lock:
        m._thought_to_key[st.id] = "x:r:0.0"
    rec = m.recommend_action([st], domain="tetris")
    assert rec is None
    assert m.recommends_miss >= 1


def test_outerface_uses_recommend_when_available():
    s = Symbioid(label="tetris")
    eye = s.add_sensor(label="eye")
    hand = s.add_actuator(label="hand")
    # Seed last observation pole
    st = Thought(id=f"{s.id}:obs:e", label="eye:0.3", transient=True)
    with s.mind._lock:
        s.mind._observations[f"{eye.id}:r:0.3"] = st
        s.mind._thought_to_key[st.id] = f"{eye.id}:r:0.3"
    with s.innerface._local_lock:
        s.innerface._last_obs_by_sensor[eye.id] = st
    s.mind.record_outcome(
        [st], "wave", domain="tetris", host_id=s.id, reward=150.0
    )
    results = s.outerface.propose_actions_from_graph()
    assert results
    ok, reason = results[0]
    assert ok is True
    assert hand.last_action == "wave"
    assert s.outerface.last_gate and "graph:wave" in s.outerface.last_gate
