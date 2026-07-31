"""
Vectorized full-graph pulse (Phase 2A/2B) — CPU numpy.

2A: hot-set dense decay + adjacency spread/Hebb.
2B: when hot/N is high, resident CSR + full activation vector + scatter-add.

Eligible when Mind.dynamics_backend == "vector" and pulse is unrestricted.
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


def _ensure_csr(host: Any) -> dict[str, Any]:
    """
    Build or reuse CSR-like adjacency: per-source target indices, weights, Link refs.

    Cached on host until thoughts size changes or out-index dirty.
    """
    thoughts: dict = host.thoughts
    n = len(thoughts)
    dirty = bool(getattr(host, "_out_index_dirty", False))
    cache = getattr(host, "_csr_cache", None)
    if (
        cache is not None
        and not dirty
        and int(cache.get("n", -1)) == n
        and cache.get("id_set") == set(thoughts.keys())
    ):
        return cache

    if dirty and hasattr(host, "_rebuild_out_index_unlocked"):
        host._rebuild_out_index_unlocked()

    ids = list(thoughts.keys())
    id_to_i = {tid: i for i, tid in enumerate(ids)}
    out_tgt: list[list[int]] = [[] for _ in range(n)]
    out_w: list[list[float]] = [[] for _ in range(n)]
    out_link: list[list[Link]] = [[] for _ in range(n)]

    for t in thoughts.values():
        if not isinstance(t, Link) or getattr(t, "is_port", False):
            continue
        src = t.source
        tgt = t.target
        if src is None or tgt is None:
            continue
        si = id_to_i.get(src.id)
        ti = id_to_i.get(tgt.id)
        if si is None or ti is None:
            continue
        out_tgt[si].append(ti)
        out_w[si].append(float(getattr(t, "weight", 1.0)))
        out_link[si].append(t)

    cache = {
        "n": n,
        "ids": ids,
        "id_to_i": id_to_i,
        "out_tgt": out_tgt,
        "out_w": out_w,
        "out_link": out_link,
        "id_set": set(ids),
    }
    host._csr_cache = cache
    return cache


def pulse_partition_vector(
    host: Any,
    *,
    engine_name: str = "global",
    hebb: Optional[bool] = None,
) -> dict:
    """Full-graph vector pulse. Caller holds ``host.graph_lock``."""
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
            "vector_mode": "idle",
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
    hot_frac_thr = float(getattr(mind, "vector_csr_hot_fraction", 0.35) if mind else 0.35)
    min_hot_csr = int(getattr(mind, "vector_csr_min_hot", 256) if mind else 256)

    host.pulse_cycle += 1
    cycle = int(host.pulse_cycle)

    if getattr(host, "_out_index_dirty", False):
        host._rebuild_out_index_unlocked()
        host._csr_cache = None

    thoughts: dict[str, Thought] = host.thoughts
    n_all = len(thoughts)
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
            "vector_mode": "empty",
        }

    hot_frac = len(hot_ids) / float(max(1, n_all))
    use_csr = (
        n_all >= 64
        and len(hot_ids) >= min_hot_csr
        and hot_frac >= hot_frac_thr
    )
    if use_csr:
        return _pulse_csr_dense(
            host,
            cycle=cycle,
            engine_name=engine_name,
            dyn_mode=dyn_mode,
            graph_spread=graph_spread,
            gain=gain,
            hebb_on=hebb_on,
            hebb_lr=hebb_lr,
            co_scale=co_scale,
            pre_post=pre_post,
            w_min=w_min,
            w_max=w_max,
            hot_ids=hot_ids,
            mind=mind,
        )
    return _pulse_hotset(
        host,
        cycle=cycle,
        engine_name=engine_name,
        dyn_mode=dyn_mode,
        graph_spread=graph_spread,
        gain=gain,
        hebb_on=hebb_on,
        hebb_lr=hebb_lr,
        co_scale=co_scale,
        pre_post=pre_post,
        w_min=w_min,
        w_max=w_max,
        hot_ids=hot_ids,
        mind=mind,
    )


def _pulse_csr_dense(
    host: Any,
    *,
    cycle: int,
    engine_name: str,
    dyn_mode: str,
    graph_spread: bool,
    gain: float,
    hebb_on: bool,
    hebb_lr: float,
    co_scale: float,
    pre_post: float,
    w_min: float,
    w_max: float,
    hot_ids: set[str],
    mind: Any,
) -> dict:
    """Phase 2B: full activation vector + CSR scatter for high hot fraction."""
    thoughts: dict[str, Thought] = host.thoughts
    csr = _ensure_csr(host)
    ids: list[str] = csr["ids"]
    id_to_i: dict[str, int] = csr["id_to_i"]
    out_tgt: list[list[int]] = csr["out_tgt"]
    out_w: list[list[float]] = csr["out_w"]
    out_link: list[list[Link]] = csr["out_link"]
    n = len(ids)
    objs = [thoughts[tid] for tid in ids]

    act = np.empty(n, dtype=np.float64)
    resting = np.empty(n, dtype=np.float64)
    thresh = np.empty(n, dtype=np.float64)
    amax = np.empty(n, dtype=np.float64)
    decay = np.empty(n, dtype=np.float64)
    refrac = np.empty(n, dtype=np.int32)
    dyn = np.empty(n, dtype=np.bool_)
    for i, t in enumerate(objs):
        act[i] = float(t.activation)
        resting[i] = float(t.resting)
        thresh[i] = float(t.threshold)
        amax[i] = float(t.activation_max)
        decay[i] = max(0.0, min(1.0, float(t.decay_rate)))
        refrac[i] = int(t.refractory_ticks)
        dyn[i] = bool(t.dynamics_enabled)

    hot_mask = np.zeros(n, dtype=np.bool_)
    for tid in hot_ids:
        i = id_to_i.get(tid)
        if i is not None:
            hot_mask[i] = True
    hot_idx = np.flatnonzero(hot_mask)

    # Decay hot
    if hot_idx.size:
        en = dyn[hot_idx]
        a = act[hot_idx]
        r = resting[hot_idx]
        d = decay[hot_idx]
        a_new = a.copy()
        a_new[en] = r[en] + (1.0 - d[en]) * (a[en] - r[en])
        near = np.abs(a_new - r) < 1e-9
        a_new[near] = r[near]
        act[hot_idx] = a_new
        rf = refrac[hot_idx].copy()
        rf_en = en & (rf > 0)
        rf[rf_en] -= 1
        refrac[hot_idx] = rf

    just_fired = np.zeros(n, dtype=np.bool_)
    # still_hot after decay
    eps = 1e-6
    still = hot_mask & dyn & ((np.abs(act - resting) > eps) | (refrac > 0))

    # Fire among original hot
    fire_cand = hot_mask & dyn & (refrac == 0) & (act >= thresh)
    firer_idx = np.flatnonzero(fire_cand)
    firer_ids: set[str] = set()
    for i in firer_idx:
        ii = int(i)
        t = objs[ii]
        dr = max(0, int(getattr(t, "default_refractory", 2)))
        just_fired[ii] = True
        refrac[ii] = dr
        firer_ids.add(ids[ii])
        still[ii] = True

    spread = 0
    hebb_n = 0
    dirty = still.copy()

    if graph_spread and firer_idx.size:
        for i in firer_idx:
            si = int(i)
            tgts = out_tgt[si]
            if not tgts:
                continue
            strength = max(float(thresh[si]), float(act[si]))
            ws = out_w[si]
            links = out_link[si]
            # numpy scatter for this firer's edges
            ti_arr = np.asarray(tgts, dtype=np.int32)
            w_arr = np.asarray(ws, dtype=np.float64)
            amts = strength * w_arr * gain
            # skip dyn-disabled targets
            for j, ti in enumerate(ti_arr):
                ti = int(ti)
                if not dyn[ti] or amts[j] == 0.0:
                    continue
                act[ti] = min(float(amax[ti]), max(0.0, float(act[ti]) + float(amts[j])))
                still[ti] = True
                dirty[ti] = True
                spread += 1

            if hebb_on:
                for j, ti in enumerate(ti_arr):
                    ti = int(ti)
                    link = links[j]
                    tgt_id = ids[ti]
                    if tgt_id in firer_ids:
                        delta = hebb_lr * co_scale
                    elif float(act[ti]) >= 0.5 * float(thresh[ti]):
                        delta = hebb_lr * pre_post
                    else:
                        continue
                    if mind is not None and hasattr(mind, "phase_hebb_scale"):
                        delta *= float(mind.phase_hebb_scale(objs[si], objs[ti]))
                    link.adjust_weight(delta, w_min=w_min, w_max=w_max)
                    # keep CSR weight in sync for later pulses
                    out_w[si][j] = float(link.weight)
                    hebb_n += 1

    # Write back dirty + original hot
    write_mask = hot_mask | dirty
    for i in np.flatnonzero(write_mask):
        ii = int(i)
        t = objs[ii]
        t.activation = float(act[ii])
        t.refractory_ticks = int(refrac[ii])
        t.just_fired = bool(just_fired[ii])

    host._hot_ids = {ids[int(i)] for i in np.flatnonzero(still) if ids[int(i)] in thoughts}

    for tid in host._hot_ids:
        ht = thoughts.get(tid)
        if ht is not None:
            ht.last_hot_cycle = cycle
    for i in firer_idx:
        objs[int(i)].last_hot_cycle = cycle

    host.last_pulse_fired = int(len(firer_ids))
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
        "vector_mode": "csr",
    }


def _pulse_hotset(
    host: Any,
    *,
    cycle: int,
    engine_name: str,
    dyn_mode: str,
    graph_spread: bool,
    gain: float,
    hebb_on: bool,
    hebb_lr: float,
    co_scale: float,
    pre_post: float,
    w_min: float,
    w_max: float,
    hot_ids: set[str],
    mind: Any,
) -> dict:
    """Phase 2A: hot-set numpy decay + adjacency spread."""
    thoughts: dict[str, Thought] = host.thoughts
    hot_list = [thoughts[tid] for tid in hot_ids]
    h = len(hot_list)
    act = np.empty(h, dtype=np.float64)
    resting = np.empty(h, dtype=np.float64)
    thresh = np.empty(h, dtype=np.float64)
    decay = np.empty(h, dtype=np.float64)
    refrac = np.empty(h, dtype=np.int32)
    dyn = np.empty(h, dtype=np.bool_)
    for i, t in enumerate(hot_list):
        act[i] = float(t.activation)
        resting[i] = float(t.resting)
        thresh[i] = float(t.threshold)
        decay[i] = max(0.0, min(1.0, float(t.decay_rate)))
        refrac[i] = int(t.refractory_ticks)
        dyn[i] = bool(t.dynamics_enabled)

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

    for i, t in enumerate(hot_list):
        t.just_fired = False
        t.activation = float(act[i])
        t.refractory_ticks = int(refrac[i])

    still_hot: set[str] = set()
    for t in hot_list:
        if t.is_hot():
            still_hot.add(t.id)

    firers: list[Thought] = []
    candidates = list(still_hot) + [i for i in hot_ids if i not in still_hot]
    for tid in candidates:
        t = thoughts.get(tid)
        if t is None:
            continue
        if t.try_fire(cycle=cycle):
            firers.append(t)
            still_hot.add(t.id)

    firer_ids = {f.id for f in firers}
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
        "vector_mode": "hotset",
    }
