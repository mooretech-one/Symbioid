"""Phase 2 slice A: vector CPU pulse equivalence + basic stats."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbioid import Link, Symbioid, Thought, __version__


def _ring_graph(n_poles: int = 40, links_per: int = 2) -> Symbioid:
    s = Symbioid(install_constitution=False)
    s.mind.dynamics_enabled = True
    s.mind.spectral_mix_enabled = False
    s.mind.holonomic_store_enabled = False
    s.mind.hebb_enabled = True
    poles = []
    for i in range(n_poles):
        t = Thought(label=f"p{i}", activation=0.4 + (i % 5) * 0.25, threshold=1.0)
        s.add_thought(t)
        poles.append(t)
    lt = poles[0]
    for i, t in enumerate(poles):
        for j in range(1, links_per + 1):
            tgt = poles[(i + j) % n_poles]
            s.add_thought(Link(source=t, target=tgt, link_type=lt, weight=1.0 + 0.01 * i))
    # Heat half
    for t in poles[::2]:
        t.receive(1.2)
        s.mark_hot(t)
    return s


def _snapshot(s: Symbioid) -> dict:
    acts = {}
    refrac = {}
    weights = {}
    just = {}
    for tid, t in s.thoughts.items():
        acts[tid] = float(t.activation)
        refrac[tid] = int(t.refractory_ticks)
        just[tid] = bool(t.just_fired)
        if isinstance(t, Link):
            weights[tid] = float(t.weight)
    return {
        "act": acts,
        "refrac": refrac,
        "just": just,
        "w": weights,
        "hot": set(s._hot_ids),
        "cycle": int(s.pulse_cycle),
    }


def _clone_state(src: Symbioid) -> Symbioid:
    """Rebuild an equivalent graph for independent pulse backends."""
    dst = Symbioid(install_constitution=False)
    dst.mind.dynamics_enabled = True
    dst.mind.spectral_mix_enabled = False
    dst.mind.holonomic_store_enabled = False
    dst.mind.hebb_enabled = bool(src.mind.hebb_enabled)
    dst.mind.hebb_lr = float(src.mind.hebb_lr)
    dst.mind.propagate_gain = float(src.mind.propagate_gain)
    # Map old id → new Thought by label for poles; rebuild links
    by_label: dict[str, Thought] = {}
    # First pass: non-links
    for t in src.thoughts.values():
        if isinstance(t, Link):
            continue
        nt = Thought(
            label=t.label,
            activation=float(t.activation),
            resting=float(t.resting),
            threshold=float(t.threshold),
            activation_max=float(t.activation_max),
            decay_rate=float(t.decay_rate),
            refractory_ticks=int(t.refractory_ticks),
            default_refractory=int(t.default_refractory),
            dynamics_enabled=bool(t.dynamics_enabled),
        )
        # Force same id so hot sets align? add_thought uses factory id.
        # Better: store by label and compare by label.
        dst.add_thought(nt)
        by_label[str(t.label)] = nt
        if t.id in src._hot_ids:
            dst.mark_hot(nt)
    # Links
    for t in src.thoughts.values():
        if not isinstance(t, Link):
            continue
        src_l = str(t.source.label)
        tgt_l = str(t.target.label)
        lt_l = str(t.link_type.label)
        nl = Link(
            source=by_label[src_l],
            target=by_label[tgt_l],
            link_type=by_label[lt_l],
            weight=float(t.weight),
            is_port=bool(t.is_port),
        )
        dst.add_thought(nl)
    return dst


def test_version_at_least_049():
    parts = [int(x) for x in __version__.split(".")]
    assert parts >= [0, 0, 49]


def test_vector_backend_flag():
    s = Symbioid(install_constitution=False)
    assert s.mind.normalize_dynamics_backend() == "object"
    assert s.mind.set_dynamics_backend("vector") == "vector"
    assert s.mind.normalize_dynamics_backend("nope") == "object"


def test_vector_pulse_stats_and_backend_tag():
    s = _ring_graph(30)
    s.mind.set_dynamics_backend("vector")
    st = s.pulse_tick()
    assert st.get("dynamics_backend") == "vector"
    assert "fired" in st and "spread" in st and "hot" in st
    assert st["cycle"] >= 1


def test_object_vs_vector_activation_equivalence():
    """Same topology, seed, N pulses → activations match within atol."""
    n_poles = 48
    pulses = 12
    base = _ring_graph(n_poles)
    base.mind.hebb_enabled = True
    base.mind.spectral_mix_enabled = False

    # Object run
    so = _clone_state(base)
    so.mind.set_dynamics_backend("object")
    so.mind.hebb_enabled = True
    so.mind.spectral_mix_enabled = False
    for _ in range(pulses):
        so.pulse_tick()
    snap_o = {str(t.label): float(t.activation) for t in so.thoughts.values() if not isinstance(t, Link)}

    # Vector run from same base
    sv = _clone_state(base)
    sv.mind.set_dynamics_backend("vector")
    sv.mind.hebb_enabled = True
    sv.mind.spectral_mix_enabled = False
    for _ in range(pulses):
        st = sv.pulse_tick()
        assert st.get("dynamics_backend") == "vector"
    snap_v = {str(t.label): float(t.activation) for t in sv.thoughts.values() if not isinstance(t, Link)}

    assert set(snap_o) == set(snap_v)
    for lab in snap_o:
        assert abs(snap_o[lab] - snap_v[lab]) < 1e-4, (
            f"{lab}: object={snap_o[lab]} vector={snap_v[lab]}"
        )


def test_object_vs_vector_weights_and_fired_counts():
    base = _ring_graph(36)
    so = _clone_state(base)
    so.mind.set_dynamics_backend("object")
    so.mind.spectral_mix_enabled = False
    fired_o = 0
    for _ in range(8):
        fired_o += so.pulse_tick()["fired"]

    sv = _clone_state(base)
    sv.mind.set_dynamics_backend("vector")
    sv.mind.spectral_mix_enabled = False
    fired_v = 0
    for _ in range(8):
        fired_v += sv.pulse_tick()["fired"]

    assert fired_o == fired_v
    # Link weights by (src_label, tgt_label)
    def wmap(s: Symbioid) -> dict:
        m = {}
        for t in s.thoughts.values():
            if isinstance(t, Link):
                m[(str(t.source.label), str(t.target.label))] = float(t.weight)
        return m

    wo, wv = wmap(so), wmap(sv)
    assert set(wo) == set(wv)
    for k in wo:
        assert abs(wo[k] - wv[k]) < 1e-5


def test_vector_falls_back_with_membership():
    s = _ring_graph(20)
    s.mind.set_dynamics_backend("vector")
    ids = list(s.thoughts.keys())[:5]
    st = s.pulse_partition(membership=set(ids), engine_name="global")
    # Fallback to object path — no vector tag
    assert st.get("dynamics_backend") != "vector"
    assert "cycle" in st


def test_vector_no_spread_in_spectral_mode():
    s = _ring_graph(24)
    s.mind.set_dynamics_backend("vector")
    s.mind.set_dynamics_mode("spectral")
    s.mind.spectral_mix_enabled = False  # avoid FFT noise in this unit test
    # Force graph_spread off via spectral mode
    st = s.pulse_tick()
    assert st.get("graph_spread") is False
    assert st.get("spread", 0) == 0


def test_vector_csr_mode_when_hot_fraction_high():
    """Phase 2B: dense CSR path engages when hot/N and min_hot thresholds met."""
    s = _ring_graph(80, links_per=2)
    s.mind.set_dynamics_backend("vector")
    s.mind.spectral_mix_enabled = False
    s.mind.vector_csr_hot_fraction = 0.3
    s.mind.vector_csr_min_hot = 20
    # Heat most poles
    for t in list(s.thoughts.values()):
        if not isinstance(t, Link):
            t.receive(1.5)
            s.mark_hot(t)
    st = s.pulse_tick()
    assert st.get("dynamics_backend") == "vector"
    assert st.get("vector_mode") == "csr"
    assert st["fired"] >= 0


def test_object_vs_vector_csr_activations():
    """CSR path stays close to object over a few pulses."""
    n_poles = 60
    base = _ring_graph(n_poles, links_per=2)
    for t in list(base.thoughts.values()):
        if not isinstance(t, Link):
            t.receive(1.4)
            base.mark_hot(t)

    so = _clone_state(base)
    so.mind.set_dynamics_backend("object")
    so.mind.spectral_mix_enabled = False
    for _ in range(6):
        so.pulse_tick()
    snap_o = {
        str(t.label): float(t.activation)
        for t in so.thoughts.values()
        if not isinstance(t, Link)
    }

    sv = _clone_state(base)
    sv.mind.set_dynamics_backend("vector")
    sv.mind.spectral_mix_enabled = False
    sv.mind.vector_csr_hot_fraction = 0.2
    sv.mind.vector_csr_min_hot = 10
    for _ in range(6):
        st = sv.pulse_tick()
        assert st.get("vector_mode") in ("csr", "hotset")
    snap_v = {
        str(t.label): float(t.activation)
        for t in sv.thoughts.values()
        if not isinstance(t, Link)
    }
    for lab in snap_o:
        assert abs(snap_o[lab] - snap_v[lab]) < 1e-3, lab
