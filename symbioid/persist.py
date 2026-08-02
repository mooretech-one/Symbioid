"""Agent memory persistence — Thoughts + Mind only (no game/world/coach).

**Lean snapshot (default, v2):** store Mind registries + poles they name +
Links that touch those poles (and link_type poles). Skip unregistered
scaffolding, seeds, laws (rebuilt on host construct).

**Full snapshot:** entire host.thoughts (legacy bulk dump).

Never serialize world state, scores, ciphers, or coach models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from symbioid.Core.Link import Link
from symbioid.Core.Symbioid import Symbioid
from symbioid.Core.Thought import Thought

FORMAT = "symbioid-memory"
VERSION = 2  # lean default; still loads version 1 full dumps

# Top-level keys allowed in a memory file (agent cognition only)
ALLOWED_TOP_KEYS = frozenset(
    {
        "format",
        "version",
        "saved_at",
        "host_id",
        "engines_mode",
        "pulse_cycle",
        "snapshot",  # "lean" | "full"
        "mind",
        "thoughts",
    }
)

# Forbidden keys if present anywhere at top level (game bleed)
FORBIDDEN_TOP_KEYS = frozenset(
    {
        "world",
        "coach",
        "cipher",
        "highscores",
        "score",
        "board",
        "rng",
        "bytes_tried",
        "game",
        "paddle",
    }
)

PathLike = Union[str, Path]
SnapshotMode = str  # "lean" | "full"


def default_memory_dir() -> Path:
    return Path.home() / ".local" / "share" / "symbioid"


def default_memory_path(name: str) -> Path:
    """e.g. name='tetris_memory.json' under ~/.local/share/symbioid/."""
    return default_memory_dir() / name


def _thought_record(t: Thought) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "kind": "Link" if isinstance(t, Link) else "Thought",
        "id": t.id,
        "label": t.label,
        "transient": bool(getattr(t, "transient", False)),
        "threshold": float(getattr(t, "threshold", 1.0)),
        "activation": float(getattr(t, "activation", 0.0)),
        "resting": float(getattr(t, "resting", 0.0)),
        "decay_rate": float(getattr(t, "decay_rate", 0.15)),
        "activation_max": float(getattr(t, "activation_max", 3.0)),
        "dynamics_enabled": bool(getattr(t, "dynamics_enabled", True)),
        "engine_owner": getattr(t, "engine_owner", None),
        "last_hot_cycle": int(getattr(t, "last_hot_cycle", -1)),
        "last_fired_cycle": int(getattr(t, "last_fired_cycle", -1)),
        "default_refractory": int(getattr(t, "default_refractory", 2)),
        "export_activation": float(getattr(t, "export_activation", 0.0) or 0.0),
    }
    if isinstance(t, Link):
        rec["weight"] = float(getattr(t, "weight", 1.0))
        rec["is_port"] = bool(getattr(t, "is_port", False))
        rec["source_id"] = t.source.id if t.source is not None else None
        rec["link_type_id"] = t.link_type.id if t.link_type is not None else None
        rec["target_id"] = t.target.id if t.target is not None else None
    return rec


def _export_mind(mind: Any) -> dict[str, Any]:
    with mind._lock:
        observations = {
            ck: {"id": t.id, "label": t.label}
            for ck, t in mind._observations.items()
        }
        actions = {
            ck: {
                "id": t.id,
                "label": t.label,
                "token": mind._action_tokens.get(ck),
            }
            for ck, t in mind._actions.items()
        }
        holo = None
        store = getattr(mind, "holonomic_store", None)
        if store is not None and hasattr(store, "to_serializable"):
            holo = store.to_serializable()
        return {
            "recognition_enabled": bool(getattr(mind, "recognition_enabled", True)),
            "forget_cold_enabled": bool(getattr(mind, "forget_cold_enabled", True)),
            "forget_cold_cycles": int(getattr(mind, "forget_cold_cycles", 64)),
            "forget_transient_only": bool(getattr(mind, "forget_transient_only", False)),
            "habituate_after": int(getattr(mind, "habituate_after", 2)),
            "hebb_enabled": bool(getattr(mind, "hebb_enabled", True)),
            "dynamics_enabled": bool(getattr(mind, "dynamics_enabled", True)),
            "dynamics_mode": str(getattr(mind, "dynamics_mode", "hybrid") or "hybrid"),
            "spectral_mix_enabled": bool(getattr(mind, "spectral_mix_enabled", True)),
            "holonomic_store_enabled": bool(
                getattr(mind, "holonomic_store_enabled", True)
            ),
            "observations": observations,
            "actions": actions,
            "valence": {k: float(v) for k, v in mind._valence.items()},
            "thought_to_key": dict(mind._thought_to_key),
            "follows": dict(mind._follows),
            "integrates": dict(mind._integrates),
            "holonomic": holo,
            "tft": (
                mind.tft_export()
                if hasattr(mind, "tft_export")
                else None
            ),
            "warm_start_actions": bool(getattr(mind, "warm_start_actions", True)),
            "warm_start_prior": float(getattr(mind, "warm_start_prior", 0.12) or 0.0),
            "stats": {
                "admits_mint": int(mind.admits_mint),
                "admits_reuse": int(mind.admits_reuse),
                "admits_skip": int(mind.admits_skip),
                "follows_mint": int(mind.follows_mint),
                "integrates_mint": int(mind.integrates_mint),
                "actions_mint": int(mind.actions_mint),
                "outcomes_recorded": int(mind.outcomes_recorded),
                "hebb_updates": int(mind.hebb_updates),
                "forgets_cold": int(mind.forgets_cold),
                "holonomic_writes": int(getattr(mind, "holonomic_writes", 0)),
                "holonomic_reads": int(getattr(mind, "holonomic_reads", 0)),
            },
        }


def _mind_core_ids(mind: Any) -> set[str]:
    """Thought ids named by Mind registries (learning substrate)."""
    ids: set[str] = set()
    with mind._lock:
        for t in mind._observations.values():
            if t is not None:
                ids.add(t.id)
        for t in mind._actions.values():
            if t is not None:
                ids.add(t.id)
        ids.update(mind._thought_to_key.keys())
    return ids


def _lean_keep_ids(host: Symbioid) -> set[str]:
    """
    Core poles + Links that touch them + their link_type poles.

    Does not include seeds, laws, unregistered formation scaffolding.
    """
    core = _mind_core_ids(host.mind)
    keep: set[str] = set(core)
    # Iteratively add incident Links and their link_type poles
    changed = True
    while changed:
        changed = False
        for t in host.thoughts.values():
            if not isinstance(t, Link):
                continue
            if t.id in keep:
                continue
            src = t.source.id if t.source is not None else None
            tgt = t.target.id if t.target is not None else None
            # Include link if either endpoint is core/keep (learning edge)
            if (src and src in keep) or (tgt and tgt in keep):
                keep.add(t.id)
                if src:
                    keep.add(src)
                if tgt:
                    keep.add(tgt)
                if t.link_type is not None:
                    keep.add(t.link_type.id)
                changed = True
    return keep


def _select_thoughts(
    host: Symbioid,
    *,
    mode: SnapshotMode,
) -> list[dict[str, Any]]:
    if mode == "full":
        return [_thought_record(t) for t in host.thoughts.values()]
    # lean (default)
    keep = _lean_keep_ids(host)
    out: list[dict[str, Any]] = []
    for tid in keep:
        t = host.thoughts.get(tid)
        if t is not None:
            out.append(_thought_record(t))
    return out


def export_memory(
    host: Symbioid,
    *,
    mode: SnapshotMode = "lean",
) -> dict[str, Any]:
    """
    Build an agent-only memory dict (Thoughts + Mind; no game fields).

    mode:
      lean — Mind registries + poles + incident Links (default)
      full — entire host.thoughts (bulk / debug)
    """
    m = (mode or "lean").lower()
    if m not in ("lean", "full"):
        m = "lean"
    with host.graph_lock:
        thoughts = _select_thoughts(host, mode=m)
        data = {
            "format": FORMAT,
            "version": VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "host_id": host.id,
            "engines_mode": getattr(host, "engines_mode", "legacy") or "legacy",
            "pulse_cycle": int(getattr(host, "pulse_cycle", 0) or 0),
            "snapshot": m,
            "mind": _export_mind(host.mind),
            "thoughts": thoughts,
        }
    return data


def _make_thought(rec: dict[str, Any], *, reset_activation: bool) -> Thought:
    act = 0.0 if reset_activation else float(rec.get("activation", 0.0) or 0.0)
    rest = float(rec.get("resting", 0.0) or 0.0)
    if reset_activation:
        act = rest
    return Thought(
        id=str(rec["id"]),
        label=rec.get("label"),
        transient=bool(rec.get("transient", False)),
        threshold=float(rec.get("threshold", 1.0)),
        activation=act,
        resting=rest,
        decay_rate=float(rec.get("decay_rate", 0.15)),
        activation_max=float(rec.get("activation_max", 3.0)),
        dynamics_enabled=bool(rec.get("dynamics_enabled", True)),
        engine_owner=rec.get("engine_owner"),
        last_hot_cycle=-1 if reset_activation else int(rec.get("last_hot_cycle", -1)),
        last_fired_cycle=int(rec.get("last_fired_cycle", -1)),
        default_refractory=int(rec.get("default_refractory", 2)),
        export_activation=0.0
        if reset_activation
        else float(rec.get("export_activation", 0.0) or 0.0),
        refractory_ticks=0,
    )


def _validate_payload(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("memory payload must be a dict")
    if data.get("format") != FORMAT:
        raise ValueError(f"unknown format: {data.get('format')!r}")
    ver = int(data.get("version", 0) or 0)
    if ver not in (1, VERSION):
        raise ValueError(f"unsupported memory version: {ver}")
    bad = set(data.keys()) & FORBIDDEN_TOP_KEYS
    if bad:
        raise ValueError(f"game fields not allowed in memory: {sorted(bad)}")


def apply_memory(
    host: Symbioid,
    data: dict[str, Any],
    *,
    reset_activation: bool = True,
    require_host_id: bool = True,
) -> None:
    """
    Inject agent memory into an already-built host (sensors/actuators present).

    **Merge** snapshot poles/Links into the live graph (keep seed, laws,
    awareness from the current build). Restores Mind registries including
    Action command keys.

    v1 full dumps and v2 lean dumps both merge; lean simply has fewer poles.
    """
    _validate_payload(data)
    if require_host_id and data.get("host_id") != host.id:
        raise ValueError(
            f"host_id mismatch: file={data.get('host_id')!r} host={host.id!r}"
        )

    thought_recs = list(data.get("thoughts") or [])
    node_recs = [r for r in thought_recs if r.get("kind") != "Link"]
    link_recs = [r for r in thought_recs if r.get("kind") == "Link"]

    built: dict[str, Thought] = {}
    for rec in node_recs:
        tid = str(rec.get("id") or "")
        if not tid:
            continue
        built[tid] = _make_thought(rec, reset_activation=reset_activation)

    for rec in link_recs:
        tid = str(rec.get("id") or "")
        sid = rec.get("source_id")
        lid = rec.get("link_type_id")
        tgt = rec.get("target_id")
        if not tid or not sid or not lid or not tgt:
            continue
        for need_id in (sid, lid, tgt):
            if need_id not in built:
                # may already be on host; placeholder until merge
                existing = host.thoughts.get(str(need_id))
                if existing is not None and not isinstance(existing, Link):
                    built[str(need_id)] = existing
                else:
                    built[str(need_id)] = Thought(
                        id=str(need_id),
                        threshold=10.0,
                        dynamics_enabled=False,
                    )
        try:
            link = Link(
                id=tid,
                label=rec.get("label"),
                source=built[str(sid)],
                link_type=built[str(lid)],
                target=built[str(tgt)],
                weight=float(rec.get("weight", 1.0)),
                is_port=bool(rec.get("is_port", False)),
                threshold=float(rec.get("threshold", 10.0)),
                transient=bool(rec.get("transient", False)),
                dynamics_enabled=bool(rec.get("dynamics_enabled", False)),
                activation=0.0
                if reset_activation
                else float(rec.get("activation", 0.0) or 0.0),
                resting=float(rec.get("resting", 0.0) or 0.0),
            )
        except (TypeError, ValueError, KeyError):
            continue
        built[tid] = link

    with host.graph_lock:
        # Merge: keep current shell (seed/laws/awareness), overlay snapshot
        merged = dict(host.thoughts)
        for tid, t in built.items():
            merged[tid] = t
        # Re-wire Links to use merged pole objects (same ids)
        for tid, t in list(merged.items()):
            if not isinstance(t, Link):
                continue
            src = merged.get(t.source.id)
            lt = merged.get(t.link_type.id)
            tgt = merged.get(t.target.id)
            if src is None or lt is None or tgt is None:
                continue
            if src is t.source and lt is t.link_type and tgt is t.target:
                continue
            try:
                merged[tid] = Link(
                    id=t.id,
                    label=t.label,
                    source=src,
                    link_type=lt,
                    target=tgt,
                    weight=float(t.weight),
                    is_port=bool(getattr(t, "is_port", False)),
                    threshold=float(t.threshold),
                    transient=bool(t.transient),
                    dynamics_enabled=bool(t.dynamics_enabled),
                    activation=float(t.activation),
                    resting=float(t.resting),
                )
            except (TypeError, ValueError):
                pass

        host.thoughts = merged
        host._hot_ids = set()
        # Keep host pulse_cycle at 0 after lean load so cold-forget age is fresh,
        # or restore if present (both ok). Prefer restore for continuity of stats.
        if "pulse_cycle" in data:
            host.pulse_cycle = int(data.get("pulse_cycle", 0) or 0)
        if "engines_mode" in data and data["engines_mode"] in (
            "legacy",
            "hybrid",
            "spike",
        ):
            host.engines_mode = str(data["engines_mode"])

        sys_id = f"{host.id}:system"
        env_id = f"{host.id}:environment"
        agent_id = f"{host.id}:agent"
        if sys_id in host.thoughts:
            host.system = host.thoughts[sys_id]
        if env_id in host.thoughts:
            host.environment = host.thoughts[env_id]
        if agent_id in host.thoughts:
            host.agent = host.thoughts[agent_id]

        for law in getattr(host, "laws", None) or []:
            link = getattr(law, "link", None)
            if link is not None and link.id in host.thoughts:
                law.link = host.thoughts[link.id]  # type: ignore[assignment]

    # Mind registries
    mind = host.mind
    md = data.get("mind") or {}
    with mind._lock:
        if "recognition_enabled" in md:
            mind.recognition_enabled = bool(md["recognition_enabled"])
        if "forget_cold_enabled" in md:
            mind.forget_cold_enabled = bool(md["forget_cold_enabled"])
        if "forget_cold_cycles" in md:
            mind.forget_cold_cycles = int(md["forget_cold_cycles"])
        if "forget_transient_only" in md:
            mind.forget_transient_only = bool(md["forget_transient_only"])
        if "habituate_after" in md:
            mind.habituate_after = int(md["habituate_after"])
        if "hebb_enabled" in md:
            mind.hebb_enabled = bool(md["hebb_enabled"])
        if "dynamics_enabled" in md:
            mind.dynamics_enabled = bool(md["dynamics_enabled"])
        if "dynamics_mode" in md and hasattr(mind, "set_dynamics_mode"):
            try:
                mind.set_dynamics_mode(str(md["dynamics_mode"]))
            except Exception:  # noqa: BLE001
                mind.dynamics_mode = "hybrid"
        if "spectral_mix_enabled" in md:
            mind.spectral_mix_enabled = bool(md["spectral_mix_enabled"])
        if "holonomic_store_enabled" in md:
            mind.holonomic_store_enabled = bool(md["holonomic_store_enabled"])
        if "warm_start_actions" in md:
            mind.warm_start_actions = bool(md["warm_start_actions"])
        if "warm_start_prior" in md:
            try:
                mind.warm_start_prior = float(md["warm_start_prior"])
            except (TypeError, ValueError):
                pass
        if "tft" in md and hasattr(mind, "tft_import"):
            mind.tft_import(md.get("tft"))

        mind._observations.clear()
        mind._actions.clear()
        mind._action_tokens.clear()
        mind._valence.clear()
        mind._thought_to_key.clear()
        mind._follows.clear()
        mind._integrates.clear()

        for ck, meta in (md.get("observations") or {}).items():
            oid = (meta or {}).get("id")
            if not oid:
                continue
            th = host.thoughts.get(str(oid))
            if th is None:
                th = Thought(
                    id=str(oid),
                    label=(meta or {}).get("label"),
                    transient=True,
                )
                host.thoughts[th.id] = th
            mind._observations[str(ck)] = th
            mind._thought_to_key[th.id] = str(ck)

        for ck, meta in (md.get("actions") or {}).items():
            oid = (meta or {}).get("id")
            if not oid:
                continue
            th = host.thoughts.get(str(oid))
            if th is None:
                th = Thought(
                    id=str(oid),
                    label=(meta or {}).get("label"),
                    transient=False,
                )
                host.thoughts[th.id] = th
            mind._actions[str(ck)] = th
            tok = (meta or {}).get("token")
            if tok:
                mind._action_tokens[str(ck)] = str(tok)
            mind._thought_to_key[th.id] = str(ck)

        for k, v in (md.get("valence") or {}).items():
            try:
                mind._valence[str(k)] = float(v)
            except (TypeError, ValueError):
                continue

        for tid, ck in (md.get("thought_to_key") or {}).items():
            mind._thought_to_key[str(tid)] = str(ck)

        mind._follows = {str(k): str(v) for k, v in (md.get("follows") or {}).items()}
        mind._integrates = {
            str(k): str(v) for k, v in (md.get("integrates") or {}).items()
        }

        stats = md.get("stats") or {}
        for attr in (
            "admits_mint",
            "admits_reuse",
            "admits_skip",
            "follows_mint",
            "integrates_mint",
            "actions_mint",
            "outcomes_recorded",
            "hebb_updates",
            "forgets_cold",
            "holonomic_writes",
            "holonomic_reads",
        ):
            if attr in stats:
                setattr(mind, attr, int(stats[attr]))

        # Phase 3: restore interference buffer if present
        holo = md.get("holonomic")
        if holo and isinstance(holo, dict):
            from symbioid.Core.spectral import HolonomicStore

            mind.holonomic_store = HolonomicStore.from_serializable(holo)


def save_memory(
    host: Symbioid,
    path: PathLike,
    *,
    mode: SnapshotMode = "lean",
) -> Path:
    """Write agent memory JSON. Creates parent dirs. Returns path used."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = export_memory(host, mode=mode)
    for k in FORBIDDEN_TOP_KEYS:
        data.pop(k, None)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(p)
    return p


def load_memory(path: PathLike) -> Optional[dict[str, Any]]:
    """Load and validate memory JSON. Returns None if missing/invalid."""
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _validate_payload(data)
        return data
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None


def try_load_into(
    host: Symbioid,
    path: PathLike,
    *,
    reset_activation: bool = True,
) -> bool:
    """Load file into host if possible. Returns True on success."""
    data = load_memory(path)
    if data is None:
        return False
    try:
        apply_memory(host, data, reset_activation=reset_activation)
        return True
    except ValueError:
        return False
