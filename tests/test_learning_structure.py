"""Learning-structure P0: zombie syncs, active caps, co-fire filter, registry hard caps."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Link, Symbioid, Thought


def _obs(host: Symbioid, name: str, *, channel: str) -> Thought:
    t = Thought(id=f"{host.id}:obs:{name}", label=name, transient=True, threshold=1.0)
    host.add_thought(t)
    host.innerface._obs_channel[t.id] = channel
    host.innerface.add_member(t.id)
    return t


def test_cross_channel_integrate_deactivates_sync():
    """Step 1: blocked Integrate must not leave Follows sync in active_ids."""
    s = Symbioid(install_constitution=False)
    s.innerface.allow_cross_channel_follows = True  # allow mint to test deactivate path
    ear = _obs(s, "sound:1", channel=f"{s.id}:sen:ear")
    eye = _obs(s, "sight:1", channel=f"{s.id}:sen:eye")
    # Mark both as terminators
    s.integration_terminators.add(f"{s.id}:sen:ear")
    s.integration_terminators.add(f"{s.id}:sen:eye")

    stores = s.innerface.synchronize_observations(
        [ear, eye], tick=1, with_labels=True, integrate_pairs=True
    )
    # Sync was created then integrate blocked → must not remain active
    active_syncs = [k for k, v in s.innerface.active_ids.items() if v == "sync"]
    assert active_syncs == [], f"zombie syncs remain: {active_syncs}"
    assert s.innerface.integrate_blocked_cross_channel >= 1
    # completed_syncs may still archive the set; active_ids is what matters for protect
    assert stores  # created list non-empty on mint path


def test_active_sync_lru_cap():
    """Step 2: max_active_syncs bounds concurrent active sync six-sets."""
    s = Symbioid(install_constitution=False)
    s.innerface.max_active_syncs = 3
    s.innerface.allow_cross_channel_follows = True
    # Same channel so integrate succeeds OR we just activate syncs directly
    poles = []
    for i in range(8):
        t = _obs(s, f"m{i}", channel=f"{s.id}:sen:meta")
        poles.append(t)
    # Pair adjacent — same channel integrates
    s.innerface.synchronize_observations(poles, tick=1, integrate_pairs=True)
    n_sync = sum(1 for v in s.innerface.active_ids.values() if v == "sync")
    # Integrates may supersede syncs; remaining syncs must respect cap if any linger
    assert n_sync <= s.innerface.max_active_syncs
    # Force-activate many syncs to test LRU
    for i in range(10):
        s.innerface._activate(f"{s.id}:sync:force{i}", "sync")
    n_sync = sum(1 for v in s.innerface.active_ids.values() if v == "sync")
    assert n_sync <= 3


def test_cofire_meta_only_filters_cells():
    """Step 3: cofire_meta_only excludes cell Observations."""
    s = Symbioid(install_constitution=False)
    s.innerface.cofire_meta_only = True
    cell = _obs(s, "cell_r01_c02:1.0", channel=f"{s.id}:sen:cell_r01_c02")
    # Register sensor so label resolution works
    from symbioid import Sensor

    sen = Sensor(id=f"{s.id}:sen:cell_r01_c02", label="cell_r01_c02")
    s.sensors.append(sen)
    s.integration_terminators.add(sen.id)
    s.innerface._obs_channel[cell.id] = sen.id
    assert s.innerface._is_cofire_eligible(cell) is False

    meta = _obs(s, "piece_id:0.5", channel=f"{s.id}:sen:piece_id")
    sen_m = Sensor(id=f"{s.id}:sen:piece_id", label="piece_id")
    s.sensors.append(sen_m)
    s.innerface._obs_channel[meta.id] = sen_m.id
    assert s.innerface._is_cofire_eligible(meta) is True

    act = Thought(id=f"{s.id}:act:hard", label="hard")
    s.add_thought(act)
    assert s.innerface._is_cofire_eligible(act) is True


def test_follows_registry_hard_cap():
    """Step 4: max_follows_registry is a hard ceiling even with high valence."""
    s = Symbioid(install_constitution=False)
    mind = s.mind
    mind.max_follows_registry = 5
    poles = []
    for i in range(12):
        t = Thought(id=f"{s.id}:p{i}", label=f"p{i}", transient=True)
        s.add_thought(t)
        mind._thought_to_key[t.id] = f"ck:{i}"
        mind._observations[f"ck:{i}"] = t
        poles.append(t)
    for i in range(11):
        mind._valence[mind.follows_content_key(poles[i], poles[i + 1])] = 2.0
        mind.admit_follows(poles[i], poles[i + 1], host_id=s.id)
    assert len(mind._follows) <= 5


def test_cross_channel_no_mint_when_disallowed():
    """allow_cross_channel_follows=False skips Follows across terminators."""
    s = Symbioid(install_constitution=False)
    s.innerface.allow_cross_channel_follows = False
    a = _obs(s, "a", channel=f"{s.id}:sen:a")
    b = _obs(s, "b", channel=f"{s.id}:sen:b")
    s.integration_terminators.update([f"{s.id}:sen:a", f"{s.id}:sen:b"])
    before = s.mind.follows_mint
    s.innerface.synchronize_observations([a, b], tick=1, integrate_pairs=True)
    assert s.mind.follows_mint == before
    assert not any(v == "sync" for v in s.innerface.active_ids.values())
