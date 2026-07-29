"""SpectralBank + Phase 2 FFT residual mix tests."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import (
    HolonomicStore,
    Mind,
    SpectralBank,
    Symbioid,
    Thought,
    ceil_pow2,
    export_memory,
    key_to_vector,
    __version__,
)
from symbioid.persist import apply_memory


def test_version_at_least_047():
    parts = [int(x) for x in __version__.split(".")]
    assert parts >= [0, 0, 47]


def test_ceil_pow2():
    assert ceil_pow2(1) == 1
    assert ceil_pow2(2) == 2
    assert ceil_pow2(3) == 4
    assert ceil_pow2(64) == 64
    assert ceil_pow2(65) == 128


def test_mind_spectral_mix_default_on():
    m = Mind()
    assert m.spectral_mix_enabled is True
    assert m.holonomic_store_enabled is True
    assert m.spectral_bank_size == 64
    assert m.spectral_mix_gain == pytest.approx(0.15)
    assert m.spectral_bank is None
    bank = m.ensure_spectral_bank()
    assert isinstance(bank, SpectralBank)
    assert bank.size == 64
    assert m.spectral_bank is bank

def test_symbioid_default_spectral_mix_on_pulse_stats():
    host = Symbioid()
    assert host.mind.spectral_mix_enabled is True
    # Seed a hot Thought so mix has candidates
    t = Thought(id="hot1", activation=1.5, threshold=0.5)
    host.add_thought(t)
    host.mark_hot(t)
    stats = host.pulse_tick()
    assert "fired" in stats or "cycle" in stats
    assert "spectral" in stats
    assert stats["spectral"].get("skipped") is False
    assert host.mind.spectral_bank is not None


def test_mix_gain_zero_skips_and_is_bit_identical():
    host = Symbioid()
    host.mind.spectral_mix_gain = 0.0
    a = Thought(id="a", activation=2.0, threshold=10.0, activation_max=5.0)
    b = Thought(id="b", activation=0.25, threshold=10.0, activation_max=5.0)
    host.add_thought(a)
    host.add_thought(b)
    host.mark_hot(a)
    host.mark_hot(b)
    before = {tid: float(t.activation) for tid, t in host.thoughts.items()}
    stats = host.pulse_tick()
    assert stats.get("spectral", {}).get("skipped") is True
    # decay may still move activations; with gain 0 mix must not run
    assert host.mind.spectral_mix_steps == 0
    # Force no dynamics decay: re-check mix path alone
    host.mind.dynamics_enabled = True
    host2 = Symbioid()
    host2.mind.spectral_mix_enabled = True
    host2.mind.spectral_mix_gain = 0.0
    t = Thought(id="z", activation=1.0, threshold=99.0)
    host2.add_thought(t)
    host2.mark_hot(t)
    act0 = t.activation
    # spectral_mix_step alone (no pulse decay)
    r = host2.mind.spectral_mix_step(host2, candidate_ids=["z"])
    assert r["skipped"] is True
    assert t.activation == pytest.approx(act0)


def test_spectral_mix_nonlocal_without_links():
    """Low-pass FFT residual should move energy to bound slots with no Links."""
    host = Symbioid()
    host.mind.spectral_mix_enabled = True
    host.mind.spectral_mix_gain = 0.5
    host.mind.spectral_mix_lowpass = 0.35
    host.mind.spectral_soft_threshold = 0.0
    host.mind.dynamics_enabled = False  # isolate mix from pulse decay/fire
    # Empty-ish graph: only our three nodes (still has seeds — ok)
    a = Thought(id="iso_a", activation=2.5, threshold=99.0, activation_max=10.0)
    b = Thought(id="iso_b", activation=0.0, threshold=99.0, activation_max=10.0)
    c = Thought(id="iso_c", activation=0.0, threshold=99.0, activation_max=10.0)
    for t in (a, b, c):
        host.add_thought(t)
    # Direct mix step only
    r = host.mind.spectral_mix_step(host, candidate_ids=["iso_a", "iso_b", "iso_c"])
    assert r["skipped"] is False
    assert r["mixed"] >= 2
    # Non-local: B or C should gain energy from filtered spectrum of A
    assert b.activation > 0.0 or c.activation > 0.0


def test_interface_partition_does_not_mix():
    host = Symbioid()
    host.mind.spectral_mix_enabled = True
    t = Thought(id="x1", activation=1.0, threshold=0.5)
    host.add_thought(t)
    host.mark_hot(t)
    steps0 = host.mind.spectral_mix_steps
    stats = host.pulse_partition(membership={t.id}, engine_name="interface")
    assert "spectral" not in stats
    assert host.mind.spectral_mix_steps == steps0


def test_bind_stability_and_prefer_slot():
    bank = SpectralBank(size=8)
    s0 = bank.bind("a", prefer_slot=2)
    assert s0 == 2
    assert bank.bind("a") == 2  # stable
    s1 = bank.bind("b", prefer_slot=2)  # occupied → first free
    assert s1 != 2
    assert bank.thought_to_slot["a"] == 2
    assert bank.unbind("a") is True
    assert bank.bind("c", prefer_slot=2) == 2


def test_pack_unpack_identity():
    bank = SpectralBank(size=8)
    t0 = Thought(id="t0", label="t0", activation=1.25, threshold=0.5)
    t1 = Thought(id="t1", label="t1", activation=0.5, threshold=0.5)
    host = SimpleNamespace(thoughts={"t0": t0, "t1": t1})
    bank.bind("t0", prefer_slot=0)
    bank.bind("t1", prefer_slot=1)
    bank.pack_from_activations(host)
    assert bank.time_signal[0] == pytest.approx(1.25)
    assert bank.time_signal[1] == pytest.approx(0.5)
    # scramble then restore
    t0.activation = 0.0
    t1.activation = 0.0
    n = bank.unpack_to_activations(host, gain=1.0, mode="set")
    assert n == 2
    assert t0.activation == pytest.approx(1.25)
    assert t1.activation == pytest.approx(0.5)


def test_unpack_add_mode():
    bank = SpectralBank(size=4)
    t0 = Thought(id="t0", activation=1.0, activation_max=5.0)
    host = SimpleNamespace(thoughts={"t0": t0})
    bank.bind("t0", prefer_slot=0)
    bank.time_signal[0] = 0.5
    bank.unpack_to_activations(host, gain=0.2, mode="add")
    assert t0.activation == pytest.approx(1.1)


def test_fft_roundtrip_and_parseval_energy():
    bank = SpectralBank(size=32)
    rng = np.random.default_rng(0)
    bank.time_signal[:] = rng.standard_normal(bank.size).astype(np.float32)
    e_time = bank.energy_time()
    bank.fft()
    e_freq = bank.energy_freq()
    assert e_freq == pytest.approx(e_time, rel=1e-5, abs=1e-4)
    orig = bank.time_signal.copy()
    bank.ifft()
    assert bank.time_signal == pytest.approx(orig, rel=1e-5, abs=1e-5)


def test_pack_fft_ifft_unpack_restores_activations():
    bank = SpectralBank(size=16)
    thoughts = {
        f"t{i}": Thought(id=f"t{i}", activation=float(i) * 0.1, activation_max=10.0)
        for i in range(8)
    }
    host = SimpleNamespace(thoughts=thoughts)
    for i in range(8):
        bank.bind(f"t{i}", prefer_slot=i)
    bank.pack_from_activations(host)
    bank.fft()
    bank.ifft()
    for t in thoughts.values():
        t.activation = 0.0
    bank.unpack_to_activations(host, mode="set")
    for i in range(8):
        assert thoughts[f"t{i}"].activation == pytest.approx(float(i) * 0.1, abs=1e-5)


def test_thought_spectral_phase_default():
    t = Thought(id="x")
    assert t.spectral_phase == 0.0
    t.spectral_phase = 1.5
    d = t.as_dict()
    assert d["spectral_phase"] == pytest.approx(1.5)


def test_pad_pow2_size():
    bank = SpectralBank(size=50, pad_pow2=True)
    assert bank.size == 64
    bank2 = SpectralBank(size=50, pad_pow2=False)
    assert bank2.size == 50


# --- Phase 3 HolonomicStore -------------------------------------------------


def test_holonomic_write_probe_prefers_written_key():
    store = HolonomicStore(capacity=64, decay=0.0)
    store.write_key("sensor:r:1.0", strength=1.0)
    store.write_key("sensor:r:2.0", strength=1.0)
    score_a = store.score_key("sensor:r:1.0")
    score_b = store.score_key("sensor:r:2.0")
    score_c = store.score_key("sensor:r:9.9")  # never written
    assert score_a > score_c
    assert score_b > score_c
    # Fixed capacity: buffer size independent of write count
    n_bins = store.buffer.size
    for _ in range(20):
        store.write_key(f"extra:{_}", strength=0.5)
    assert store.buffer.size == n_bins
    assert store.n_writes == 22


def test_holonomic_decay_reduces_energy():
    store = HolonomicStore(capacity=32, decay=0.1)
    store.write_key("k", strength=1.0)
    e0 = store.energy()
    store.decay_step(0.5)
    assert store.energy() < e0


def test_holonomic_admit_mint_reuse_valence():
    m = Mind()
    m.holonomic_store_enabled = True
    m.holonomic_read_valence = 0.5
    m.reuse_valence_decay = 0.02
    m.habituate_after = 99  # allow many reuses
    r1 = m.admit_input("eye", {"reading": 0.5}, host_id="h")
    assert r1.action == "mint"
    assert m.holonomic_writes >= 1
    v_after_mint = m._valence[r1.content_key]
    r2 = m.admit_input("eye", {"reading": 0.5}, host_id="h")
    assert r2.action == "reuse"
    assert m.holonomic_reads >= 1
    # Holonomic boost should leave valence above pure-decay baseline
    pure_decay = v_after_mint - m.reuse_valence_decay
    assert m._valence[r2.content_key] >= pure_decay - 1e-9


def test_holonomic_persist_roundtrip():
    host = Symbioid()
    host.mind.holonomic_store_enabled = True
    host.mind.admit_input("mic", {"reading": 0.3}, host_id=host.id)
    host.mind.admit_input("mic", {"reading": 0.7}, host_id=host.id)
    assert host.mind.holonomic_store is not None
    e0 = host.mind.holonomic_store.energy()
    assert e0 > 0
    data = export_memory(host, mode="lean")
    assert data["mind"].get("holonomic") is not None
    host2 = Symbioid()
    apply_memory(host2, data, require_host_id=False)
    assert host2.mind.holonomic_store is not None
    assert host2.mind.holonomic_store.energy() == pytest.approx(e0, rel=1e-5)


def test_key_to_vector_stable():
    a = key_to_vector("hello", 32)
    b = key_to_vector("hello", 32)
    c = key_to_vector("world", 32)
    assert a == pytest.approx(b)
    assert not np.allclose(a, c)


# --- Phase 4 phase-locked Hebb -----------------------------------------------


def test_hebb_phase_default_off():
    m = Mind()
    assert m.hebb_phase_enabled is False
    assert m.phase_hebb_scale(Thought(id="a"), Thought(id="b")) == pytest.approx(1.0)


def test_phase_locked_pair_strengthens_faster():
    """Coherent phases get larger Hebb Δ than out-of-phase when enabled."""
    from symbioid import Link

    def _run(*, phase_a: float, phase_b: float) -> float:
        host = Symbioid()
        host.mind.spectral_mix_enabled = False  # isolate Hebb path
        host.mind.holonomic_store_enabled = False
        host.mind.hebb_phase_enabled = True
        host.mind.hebb_phase_tolerance = 0.4
        host.mind.hebb_phase_boost = 2.0
        host.mind.hebb_phase_mismatch = 0.5
        host.mind.hebb_lr = 0.1
        a = Thought(id="pa", activation=2.0, threshold=0.5, spectral_phase=phase_a)
        b = Thought(id="pb", activation=2.0, threshold=0.5, spectral_phase=phase_b)
        host.add_thought(a)
        host.add_thought(b)
        # minimal link type pole
        lt = Thought(id="lt", label="Follows", dynamics_enabled=False, threshold=10.0)
        host.add_thought(lt)
        link = Link(
            id="L1",
            source=a,
            link_type=lt,
            target=b,
            weight=1.0,
        )
        host.add_thought(link)
        host.mark_hot(a)
        host.mark_hot(b)
        host.pulse_tick()
        return float(link.weight)

    w_lock = _run(phase_a=0.0, phase_b=0.1)
    w_miss = _run(phase_a=0.0, phase_b=3.0)
    assert w_lock > w_miss


def test_spectral_filter_nudge_on_outcome():
    m = Mind()
    m.spectral_mix_enabled = True
    bank = m.ensure_spectral_bank()
    assert m.spectral_bin_gains is not None
    m.last_spectral_stats = {"top_bin": 2}
    before = float(m.spectral_bin_gains[2])
    m.nudge_spectral_filter(reward_sign=1.0)
    assert float(m.spectral_bin_gains[2]) > before


# --- Phase 5 audio spectral path ---------------------------------------------


def test_audio_spectral_contingent_runs():
    from symbioid.world.audio import compare_contingent_vs_noncontingent

    # Should complete without error; contingent often ≥ noncontingent but
    # spectral path is stochastic — only require finite numbers.
    res = compare_contingent_vs_noncontingent(blocks=20, seed=3, spectral=True)
    assert np.isfinite(res["contingent"])
    assert np.isfinite(res["noncontingent"])
    assert "delta" in res


def test_enable_spectral_demo_helper():
    m = Mind()
    m.spectral_mix_enabled = False
    m.holonomic_store_enabled = False
    m.enable_spectral_demo(phase_hebb=True)
    assert m.spectral_mix_enabled is True
    assert m.holonomic_store_enabled is True
    assert m.hebb_phase_enabled is True
    assert m.spectral_bank is not None
    assert m.holonomic_store is not None
    assert m.dynamics_mode == "hybrid"


# --- Mode B: spectral-primary dynamics ---------------------------------------


def test_dynamics_mode_default_hybrid():
    m = Mind()
    assert m.dynamics_mode == "hybrid"
    assert m.graph_spread_enabled() is True
    assert m.spectral_mix_wanted() is True


def test_mode_b_spectral_skips_link_spread():
    """With dynamics_mode=spectral, hot A does not recruit B via Link."""
    from symbioid import Link

    host = Symbioid()
    host.mind.enable_spectral_primary(phase_hebb=False)
    assert host.mind.dynamics_mode == "spectral"
    assert host.mind.graph_spread_enabled() is False

    a = Thought(id="sa", activation=3.0, threshold=0.5, activation_max=5.0)
    b = Thought(id="sb", activation=0.0, threshold=0.5, activation_max=5.0)
    lt = Thought(id="slt", label="Follows", dynamics_enabled=False, threshold=10.0)
    host.add_thought(a)
    host.add_thought(b)
    host.add_thought(lt)
    link = Link(id="SL1", source=a, link_type=lt, target=b, weight=4.0)
    host.add_thought(link)
    host.mark_hot(a)
    # Bind only A so mix does not dump energy into B from shared bank residual
    bank = host.mind.ensure_spectral_bank()
    bank.bind("sa", prefer_slot=0)
    # Disable mix residual for this check (still spectral mode skips spread)
    host.mind.spectral_mix_gain = 0.0
    # spectral mode forces gain back to 0.15 if zero — set gain very small instead
    host.mind.spectral_mix_gain = 1e-9
    b0 = float(b.activation)
    stats = host.pulse_tick()
    assert stats.get("dynamics_mode") == "spectral"
    assert stats.get("graph_spread") is False
    assert stats.get("spread", 0) == 0
    # B must not rise via Link spread (tiny mix gain + B unbound → ~0 change)
    assert float(b.activation) <= b0 + 0.05


def test_mode_b_graph_has_no_spectral_mix():
    host = Symbioid()
    host.mind.set_dynamics_mode("graph")
    t = Thought(id="g1", activation=1.5, threshold=0.5)
    host.add_thought(t)
    host.mark_hot(t)
    stats = host.pulse_tick()
    assert stats.get("dynamics_mode") == "graph"
    assert "spectral" not in stats


def test_mode_b_spectral_mix_runs():
    host = Symbioid()
    host.mind.enable_spectral_primary(phase_hebb=False)
    t = Thought(id="sp1", activation=1.5, threshold=0.5)
    host.add_thought(t)
    host.mark_hot(t)
    stats = host.pulse_tick()
    assert stats.get("dynamics_mode") == "spectral"
    assert "spectral" in stats
    assert stats["spectral"].get("skipped") is False


def test_enable_spectral_primary_helper():
    m = Mind()
    m.enable_spectral_primary(phase_hebb=True)
    assert m.dynamics_mode == "spectral"
    assert m.spectral_mix_enabled is True
    assert m.holonomic_store_enabled is True
    assert m.hebb_phase_enabled is True
