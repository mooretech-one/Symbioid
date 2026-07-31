"""
Vectorized full-graph pulse (Phase 2 slice A) — CPU numpy.

Hot-set-centric: dense arrays over currently hot Thoughts (decay/fire),
spread/Hebb via existing adjacency index (same edges as object path).

Eligible when Mind.dynamics_backend == "vector" and pulse is unrestricted
(membership is None, no synapse_filter, unlimited energy, owner_only False).
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from symbioid.Core.Link import Link
from symbioid.Core.Thought import Thought


def can_use_vector_pulse(
    *,
    membership: Any,
    synapse_filter: Any,
    owner_only: bool,
    energy_budget: Any,
) -> bool:
    if membership is not None:
        return False
    if synapse_filter is not None:
        return False
    if owner_only:
        return False
    if energy_budget is not None:
        return False
    return True


def pulse_partition_vector(
    host: Any,
    *,
    engine_name: str = "global",
    hebb: Optional[bool] = None,
) -> dict:
    """
    Full-graph vector pulse. Caller holds ``host.graph_lock``.
    """
    mind = host.mind
    if mind is not None and not getattr(mind, "dynamics_enabled", True):
        return {
            "cycle": host.pulse_cycle,
            "hot": 0,
            "fired": 0,
            "spread": 0,
            "hebb": 0,
            "engine": engine_name,
            "energy_used": 0.0,
            "energy_left": 0.0,
            "energy_capped": 0,
            "dynamics_mode": getattr(mind, "dynamics_mode", "hybrid"),
            "dynamics_backend": "vector",
        }

    dyn_mode = (
        mind.normalize_dynamics_mode()
        if mind is not None and hasattr(mind, "normalize_dynamics_mode")
        else "hybrid"
    )
    graph_spread = (
        mind.graph_spread_enabled()
        if mind is not None and hasattr(mind, "graph_spread_enabled")
        else True
    )

    gain = float(getattr(mind, "propagate_gain", 0.6) if mind else 0.6)
    hebb_on = (
        bool(hebb)
        if hebb is not None
        else bool(getattr(mind, "hebb_enabled", True) if mind else True)
    )
    if not graph_spread:
        hebb_on = False
    hebb_lr = float(getattr(mind, "hebb_lr", 0.08) if mind else 0.08)
    co_scale = float(getattr(mind, "hebb_co_fire_scale", 1.0) if mind else 1.0)
    pre_post = float(getattr(mind, "hebb_pre_post_scale", 0.35) if mind else 0.35)
    w_min = float(getattr(mind, "weight_min", 0.05) if mind else 0.05)
    w_max = float(getattr(mind, "weight_max", 4.0) if mind else 4.0)

    host.pulse_cycle += 1
    cycle = int(host.pulse_cycle)

    if getattr(host, "_out_index_dirty", False):
        host._rebuild_out_index_unlocked()

    thoughts: dict[str, Thought] = host.thoughts
    hot_ids = {tid for tid in host._hot_ids if tid in thoughts}
    host._hot_ids = set(hot_ids)

    if not hot_ids:
        host.last_pulse_fired = 0
        host.last_pulse_hot = 0
        host.last_hebb_updates = 0
        return {
            "cycle": cycle,
            "hot": 0,
            "fired": 0,
            "spread": 0,
            "hebb": 0,
            "engine": engine_name,
            "energy_used": 0.0,
            "energy_left": 0.0,
            "energy_capped": 0,
            "dynamics_mode": dyn_mode,
            "graph_spread": bool(graph_spread),
            "dynamics_backend": "vector",
        }

    # Stable list of hot Thoughts
    hot_list = [thoughts[tid] for tid in hot_ids]
    h = len(hot_list)
    act = np.empty(h, dtype=np.float64)
    resting = np.empty(h, dtype=np.float64)
    thresh = np.empty(h, dtype=np.float64)
    amax = np.empty(h, dtype=np.float64)
    decay = np.empty(h, dtype=np.float64)
    refrac = np.empty(h, dtype=np.int32)
    dyn = np.empty(h, dtype=np.bool_)
    for i, t in enumerate(hot_list):
        act[i] = float(t.activation)
        resting[i] = float(t.resting)
        thresh[i] = float(t.threshold)
        amax[i] = float(t.activation_max)
        decay[i] = max(0.0, min(1.0, float(t.decay_rate)))
        refrac[i] = int(t.refractory_ticks)
        dyn[i] = bool(t.dynamics_enabled)

    # --- 1) Decay (vectorized) ---
    en = dyn
    a_new = act.copy()
    a_new[en] = resting[en] + (1.0 - decay[en]) * (act[en] - resting[en])
    near = np.abs(a_new - resting) < 1e-9
    a_new[near] = resting[near]
    act[:] = a_new
    rf = refrac.copy()
    rf_en = en & (rf > 0)
    rf[rf_en] = rf[rf_en] - 1
    refrac[:] = rf

    # Write decay back before is_hot / fire (object mutates in place)
    for i, t in enumerate(hot_list):
        t.just_fired = False
        if dyn[i]:
            t.activation = float(act[i])
            t.refractory_ticks = int(refrac[i])
        else:
            t.activation = float(act[i])
            t.refractory_ticks = int(refrac[i])

    still_hot: set[str] = set()
    for t in hot_list:
        if t.is_hot():
            still_hot.add(t.id)

    # --- 2) Fire ---
    firers: list[Thought] = []
    # candidates = still_hot then remaining original hot (object order non-deterministic)
    candidates = list(still_hot) + [i for i in hot_ids if i not in still_hot]
    for tid in candidates:
        t = thoughts.get(tid)
        if t is None:
            continue
        if t.try_fire(cycle=cycle):
            firers.append(t)
            still_hot.add(t.id)

    firer_ids = {f.id for f in firers}

    # --- 3) Spread + Hebb via adjacency (same as object) ---
    spread = 0
    hebb_n = 0
    if graph_spread:
        for firer in firers:
            strength = max(float(firer.threshold), float(firer.activation))
            for link in host._outgoing_links_unlocked(firer.id):
                tgt = link.target
                if tgt is None:
                    continue
                w = float(getattr(link, "weight", 1.0))
                amt = strength * w * gain
                if amt != 0:
                    tgt.receive(amt)
                    still_hot.add(tgt.id)
                    host._register_in_store_unlocked(tgt)
                    host._hot_ids.add(tgt.id)
                    spread += 1

                if not hebb_on or not hasattr(link, "adjust_weight"):
                    continue
                if tgt.id in firer_ids:
                    delta = hebb_lr * co_scale
                elif float(tgt.activation) >= 0.5 * float(tgt.threshold):
                    delta = hebb_lr * pre_post
                else:
                    continue
                if mind is not None and hasattr(mind, "phase_hebb_scale"):
                    delta *= float(mind.phase_hebb_scale(firer, tgt))
                link.adjust_weight(delta, w_min=w_min, w_max=w_max)
                hebb_n += 1

    host._hot_ids = {i for i in still_hot if i in thoughts}

    for hid in host._hot_ids:
        ht = thoughts.get(hid)
        if ht is not None:
            ht.last_hot_cycle = cycle
    for firer in firers:
        firer.last_hot_cycle = cycle

    host.last_pulse_fired = len(firers)
    host.last_pulse_hot = len(host._hot_ids)
    host.last_hebb_updates = hebb_n
    if mind is not None and hebb_n:
        mind.hebb_updates = int(getattr(mind, "hebb_updates", 0)) + hebb_n

    return {
        "cycle": cycle,
        "hot": host.last_pulse_hot,
        "fired": host.last_pulse_fired,
        "spread": spread,
        "hebb": hebb_n,
        "engine": engine_name,
        "energy_used": 0.0,
        "energy_left": 0.0,
        "energy_capped": 0,
        "dynamics_mode": dyn_mode,
        "graph_spread": bool(graph_spread),
        "dynamics_backend": "vector",
    }
