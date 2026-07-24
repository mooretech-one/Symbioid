"""Innerface process — binding spiking engine (Phase 2) + legacy automata."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from symbioid.Core.SpikingEngine import SpikingEngine
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import (
    complete_follows_set,
    complete_formation,
    complete_integrate_set,
    emit_six_set,
    is_scaffold_thought,
    six_set_poles,
)
from symbioid.Core.ids import _new_id


@dataclass
class Innerface(SpikingEngine):
    """
    Inner process (~3): Feeling / Reflection / Maker — Self, Mind, Thought.

    **legacy:** complete formations, Follows, Integrates, depth-fold, prune
    on every inbox drain (automata-primary).

    **hybrid/spike (Phase 2):** SpikingEngine — port-in from Interface exports,
    membership pulse, co-fire consolidators (Follows/Integrates), sparse
    depth-fold/prune every ``consolidate_every`` ticks.
    """

    id: str = field(default_factory=lambda: _new_id("inner-"))
    label: Optional[str] = "Innerface"
    engine_name: str = "innerface"
    # Import charge from Interface export_activation
    port_gain: float = 0.55
    # Run depth-fold + prune at most every N engine ticks (hybrid)
    consolidate_every: int = 5
    # H2: soft caps for depth-fold (integrate-of-integrates).
    # Global: never keep more than this many active integrates total.
    max_active_integrates: int = 24
    # Per awareness channel (sensor/actuator terminator): cascade folds until
    # each channel has ≤ this many active integrates (multi-step halving).
    max_active_integrates_per_channel: int = 3
    # If True, every batch runs depth fold until per-channel + global caps hold
    # (not only when already over the global soft cap).
    eager_depth_fold: bool = True
    formation_ticks: int = field(default=0, init=False, repr=False)
    completed_formations: dict[str, dict[str, Thought]] = field(
        default_factory=dict, init=False, repr=False
    )
    completed_syncs: dict[str, dict[str, Thought]] = field(
        default_factory=dict, init=False, repr=False
    )
    completed_integrates: dict[str, dict[str, Thought]] = field(
        default_factory=dict, init=False, repr=False
    )
    sets_emitted: int = field(default=0, init=False, repr=False)
    # set_id → kind ("sense" | "sync" | "integrate") — not yet superseded
    active_ids: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # observation Thought id → sense formation_id
    _obs_to_formation: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # observation Thought id → sensor_id (integration channel / terminator)
    _obs_channel: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # integrate_id → sensor_id channel (same-channel depth fold only)
    _integrate_channel: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    # sensor_id → most recent Observation (for temporal last-two integrate)
    _last_obs_by_sensor: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    # frozenset key already integrated (obs pair or depth parent integrate ids)
    _integrated_pairs: set[frozenset[str]] = field(default_factory=set, init=False, repr=False)
    integrate_count: int = field(default=0, init=False, repr=False)
    depth_fold_count: int = field(default=0, init=False, repr=False)
    integrate_blocked_cross_channel: int = field(default=0, init=False, repr=False)
    # Thoughts removed after being superseded by Integrate / depth-fold
    thoughts_pruned: int = field(default=0, init=False, repr=False)
    thoughts_forgotten: int = field(default=0, init=False, repr=False)
    # Auto-GC after integrate (can disable for debugging)
    auto_prune: bool = True
    # Phase 2 metrics
    co_fire_consolidations: int = field(default=0, init=False, repr=False)
    port_imports: int = field(default=0, init=False, repr=False)
    engine_ticks: int = field(default=0, init=False, repr=False)

    def process(self) -> Optional[threading.Thread]:
        thread = super().process()
        return thread

    def _engine_mode(self) -> str:
        host = self.host
        if host is None:
            return "legacy"
        mode = getattr(host, "engines_mode", "legacy") or "legacy"
        return mode if mode in ("legacy", "hybrid", "spike") else "legacy"

    def _register_members(self, *thoughts: Optional[Thought]) -> None:
        for t in thoughts:
            if t is None:
                continue
            self.add_member(t.id)
            # Dual membership: keep interface owner if set; else claim inner
            if t.engine_owner is None:
                t.engine_owner = self.engine_name

    @property
    def active_set_count(self) -> int:
        """Number of six-sets still active (not superseded by integration)."""
        with self._local_lock:
            return len(self.active_ids)

    def active_set_summary(self) -> dict[str, int]:
        """Counts of active six-sets by kind."""
        with self._local_lock:
            out: dict[str, int] = {}
            for kind in self.active_ids.values():
                out[kind] = out.get(kind, 0) + 1
            return out

    def _merge_store(self, store: dict[str, Thought]) -> None:
        host = self.host
        if host is None:
            return
        with host.graph_lock:
            for tid, thought in store.items():
                host.thoughts[tid] = thought

    def _emit_completed_set(self, kind: str, store: dict[str, Thought]) -> None:
        """Only process-loop output: print six-set labels when completed."""
        with self._local_lock:
            self.sets_emitted += 1
            n = self.sets_emitted
        emit_six_set(kind, store, index=n)

    def _activate(self, set_id: str, kind: str) -> None:
        with self._local_lock:
            self.active_ids[set_id] = kind

    def _deactivate(self, set_id: str) -> None:
        with self._local_lock:
            self.active_ids.pop(set_id, None)

    def accept_formation(
        self,
        handoff: dict[str, Any],
        *,
        integrate_temporal: bool = True,
    ) -> dict[str, Thought]:
        """
        Complete a formation_handoff (Rodin 4,8,7,5): Sensor -Perceives→ Observation
        and reciprocal PerceivedBy. Merges into host graph; emits six-set line.
        Optionally integrates last-two Observations for this sensor (temporal halving).

        Mind reuse: if ``reused`` and the sense six-set already exists on the host,
        short-circuit — no new Thoughts, no emit, no temporal integrate.
        """
        fid = handoff["formation_id"]
        partial = handoff.get("partial") or {}
        obs = partial.get("observation")
        sensor_id = str(handoff.get("sensor_id") or "")
        sensor_label = handoff.get("sensor_label")
        reused = bool(handoff.get("reused") or handoff.get("mind_action") == "reuse")

        # Short-circuit pure recognition reuse (same content-addressed poles)
        if reused:
            with self._local_lock:
                existing = self.completed_formations.get(fid)
            if existing is not None:
                if isinstance(obs, Thought) and sensor_id:
                    with self._local_lock:
                        self._last_obs_by_sensor[sensor_id] = obs
                        self._obs_to_formation[obs.id] = fid
                        self._obs_channel[obs.id] = sensor_id
                return existing
            # First reuse after mint but links already on host graph
            host = self.host
            if host is not None:
                with host.graph_lock:
                    link_id = f"{fid}:sensor_perceives_obs"
                    if link_id in host.thoughts and isinstance(obs, Thought):
                        with self._local_lock:
                            self.formation_ticks += 1
                            if sensor_id:
                                self._last_obs_by_sensor[sensor_id] = obs
                                self._obs_to_formation[obs.id] = fid
                                self._obs_channel[obs.id] = sensor_id
                        # Return a minimal store from host poles without minting
                        store = {
                            k: host.thoughts[k]
                            for k in (
                                f"{fid}:perceives",
                                f"{fid}:perceived_by",
                                f"{fid}:sensor_perceives_obs",
                                f"{fid}:obs_perceived_by_sensor",
                            )
                            if k in host.thoughts
                        }
                        if isinstance(partial.get("sensor"), Thought):
                            store[partial["sensor"].id] = partial["sensor"]
                        store[obs.id] = obs
                        with self._local_lock:
                            self.completed_formations[fid] = store
                        return store

        store = complete_formation(handoff)
        self._merge_store(store)
        with self._local_lock:
            self.completed_formations[fid] = store
            self.formation_ticks += 1
        self._activate(fid, "sense")
        self._emit_completed_set("sense", store)
        self._stimulate_poles(store, kind="sense")
        # Membership for binding engine
        poles = six_set_poles(store)
        self._register_members(*poles)

        if isinstance(obs, Thought) and sensor_id:
            with self._local_lock:
                prev = self._last_obs_by_sensor.get(sensor_id)
                self._last_obs_by_sensor[sensor_id] = obs
                self._obs_to_formation[obs.id] = fid
                self._obs_channel[obs.id] = sensor_id
            # Outerfaces: Interface Observation may be Feedback for Belief formation
            host = self.host
            if host is not None and host.outerface is not None:
                host.outerface.post(
                    {
                        "kind": "interface_observation",
                        "observation": obs,
                        "sensor_id": sensor_id,
                        "sensor_label": sensor_label,
                        "formation_id": fid,
                    }
                )
            # Temporal integrate: skip when pure reuse of same poles
            if (
                integrate_temporal
                and prev is not None
                and not reused
                and prev.id != obs.id
            ):
                self.integrate_pair(
                    prev,
                    obs,
                    reason="temporal",
                    with_labels=bool(handoff.get("with_labels", True)),
                    channel=sensor_id,
                )
                self.maybe_depth_fold(
                    with_labels=bool(handoff.get("with_labels", True))
                )
        return store

    def synchronize_observations(
        self,
        observations: list[Thought],
        *,
        tick: Optional[int] = None,
        with_labels: bool = True,
        integrate_pairs: bool = True,
    ) -> list[dict[str, Thought]]:
        """
        Build lateral Follows/FollowedBy six-sets between Observations
        (ordered by label). Optionally Rodin-halve each pair via Integrates.

        Mind.admit_follows: co-occurrence is content-addressed (undirected pair
        of Observation keys). mint → new stable sync; reuse → no new scaffold;
        skip → habituated, no handoff / no re-integrate.
        """
        if len(observations) < 2:
            return []
        ordered = sorted(observations, key=lambda t: (t.label or t.id, t.id))
        host = self.host
        host_id = host.id if host is not None else "host"
        tick_part = tick if tick is not None else self.cycles
        mind = getattr(host, "mind", None) if host is not None else None
        created: list[dict[str, Thought]] = []
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]

            admit = None
            if mind is not None:
                admit = mind.admit_follows(a, b, host_id=host_id)
                if admit.action == "skip":
                    continue
                sync_id = admit.formation_id or (
                    f"{host_id}:sync:t{tick_part}:{a.id}:{b.id}"
                )
            else:
                sync_id = f"{host_id}:sync:t{tick_part}:{a.id[-8:]}:{b.id[-8:]}"

            reused = bool(admit is not None and admit.action == "reuse")
            with self._local_lock:
                existing = self.completed_syncs.get(sync_id)

            if reused and existing is not None:
                # Association known — no new scaffolding; optional first integrate
                created.append(existing)
                if integrate_pairs:
                    self.integrate_pair(
                        a,
                        b,
                        reason="follows",
                        tick=tick_part,
                        with_labels=with_labels,
                        deactivate_sync_id=sync_id,
                    )
                continue

            store = complete_follows_set(
                a, b, sync_id=sync_id, with_labels=with_labels
            )
            self._merge_store(store)
            with self._local_lock:
                self.completed_syncs[sync_id] = store
            self._activate(sync_id, "sync")
            self._emit_completed_set("sync", store)
            self._stimulate_poles(store, kind="sync")
            created.append(store)
            # Cross-sensor Follows is allowed (co-occurrence), but Integrate is
            # terminated by awareness ("Symbioid has Ear" vs "has Eye").
            if integrate_pairs:
                self.integrate_pair(
                    a,
                    b,
                    reason="follows",
                    tick=tick_part,
                    with_labels=with_labels,
                    deactivate_sync_id=sync_id,
                )
        return created

    def _channel_for_obs(self, obs: Thought) -> Optional[str]:
        with self._local_lock:
            return self._obs_channel.get(obs.id)

    def _same_integration_channel(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        channel: Optional[str] = None,
    ) -> bool:
        """
        Awareness terminators: Observations from different Sensors/Actuators
        must not Integrate across channels.
        """
        if channel is not None:
            return True  # caller asserts single channel
        ca = self._channel_for_obs(observation_a)
        cb = self._channel_for_obs(observation_b)
        if ca is None or cb is None:
            return True  # unknown lineage — allow (depth uses explicit channel)
        if ca == cb:
            return True
        host = self.host
        if host is not None and host.integration_terminators:
            # Both ends are known terminator channels and differ → stop
            if ca in host.integration_terminators or cb in host.integration_terminators:
                return False
        return ca == cb

    def _supersede_sources_for_integrate(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        deactivate_sync_id: Optional[str],
        deactivate_integrate_ids: Optional[list[str]],
        integrate_id: str,
    ) -> None:
        """Deactivate sense/sync/parent integrates; mark this integrate active."""
        for obs in (observation_a, observation_b):
            fid = self._obs_to_formation.get(obs.id)
            if fid:
                self.active_ids.pop(fid, None)
        if deactivate_sync_id:
            self.active_ids.pop(deactivate_sync_id, None)
        if deactivate_integrate_ids:
            for iid in deactivate_integrate_ids:
                self.active_ids.pop(iid, None)
        self.active_ids[integrate_id] = "integrate"

    def integrate_pair(
        self,
        observation_a: Thought,
        observation_b: Thought,
        *,
        reason: str = "pair",
        tick: Optional[int] = None,
        with_labels: bool = True,
        deactivate_sync_id: Optional[str] = None,
        deactivate_integrate_ids: Optional[list[str]] = None,
        pair_key: Optional[frozenset[str]] = None,
        depth_fold: bool = False,
        channel: Optional[str] = None,
    ) -> Optional[dict[str, Thought]]:
        """
        H1: integrate two Observations into an Integrates/IntegratedBy six-set.
        H2: depth_fold supersedes parent integrate ids (integrate-of-integrates).
        Awareness terminators block cross-Sensor/Actuator Integrate.

        Mind.admit_integrates: content key = undirected pole pair + channel
        (+ depth parent ids). mint → stable integrate_id; reuse/skip avoid remint.
        """
        if observation_a.id == observation_b.id and not deactivate_integrate_ids:
            return None
        # Terminator: do not Integrate across different sensor/actuator channels
        if not depth_fold and not self._same_integration_channel(
            observation_a, observation_b, channel=channel
        ):
            with self._local_lock:
                self.integrate_blocked_cross_channel += 1
            return None

        host = self.host
        host_id = host.id if host is not None else "host"

        # Resolve channel for this integrate (terminator lineage)
        ch = channel
        if ch is None:
            ca = self._channel_for_obs(observation_a)
            cb = self._channel_for_obs(observation_b)
            if ca and ca == cb:
                ch = ca
        if ch is None and deactivate_integrate_ids:
            with self._local_lock:
                chs = {
                    self._integrate_channel[i]
                    for i in deactivate_integrate_ids
                    if i in self._integrate_channel
                }
            if len(chs) == 1:
                ch = next(iter(chs))
        ch_key = ch or "_"

        depth_parents: Optional[tuple[str, str]] = None
        if depth_fold and deactivate_integrate_ids and len(deactivate_integrate_ids) >= 2:
            depth_parents = (
                deactivate_integrate_ids[0],
                deactivate_integrate_ids[1],
            )

        mind = getattr(host, "mind", None) if host is not None else None
        if mind is not None:
            admit = mind.admit_integrates(
                observation_a,
                observation_b,
                host_id=host_id,
                channel=ch_key,
                depth_parents=depth_parents,
            )
            integrate_id = admit.formation_id or ""
            # Prefer content key so depth/g-fold callers cannot remint via unique pair_key
            key = frozenset({admit.content_key})

            if admit.action in ("skip", "reuse") and integrate_id:
                existing: Optional[dict[str, Thought]] = None
                with self._local_lock:
                    existing = self.completed_integrates.get(integrate_id)
                    if existing is not None:
                        self._supersede_sources_for_integrate(
                            observation_a,
                            observation_b,
                            deactivate_sync_id=deactivate_sync_id,
                            deactivate_integrate_ids=deactivate_integrate_ids,
                            integrate_id=integrate_id,
                        )
                        self._integrated_pairs.add(key)
                if existing is not None:
                    self.maybe_gc()
                    return existing
                if admit.action == "skip":
                    return None
            # mint or reuse-without-store: build with stable integrate_id
        else:
            key = pair_key or frozenset({observation_a.id, observation_b.id})
            tick_part = tick if tick is not None else self.cycles
            with self._local_lock:
                seq = self.integrate_count
            integrate_id = (
                f"{host_id}:int:{reason}:t{tick_part}:n{seq}:"
                f"{observation_a.id}:{observation_b.id}"
            )
            if len(integrate_id) > 180:
                integrate_id = (
                    f"{host_id}:int:{reason}:t{tick_part}:n{seq}:"
                    f"{hash(observation_a.id) & 0xFFFFFFFF:x}:"
                    f"{hash(observation_b.id) & 0xFFFFFFFF:x}"
                )

        with self._local_lock:
            if key in self._integrated_pairs:
                existing = self.completed_integrates.get(integrate_id)
                if existing is not None:
                    return existing
                # Same key already in flight without store — do not double-add
            else:
                self._integrated_pairs.add(key)

        try:
            store = complete_integrate_set(
                observation_a,
                observation_b,
                integrate_id=integrate_id,
                with_labels=with_labels,
            )
        except (ValueError, RuntimeError) as exc:
            self.last_error = f"integrate: {exc}"
            with self._local_lock:
                self._integrated_pairs.discard(key)
            return None

        self._merge_store(store)
        with self._local_lock:
            self.completed_integrates[integrate_id] = store
            self.integrate_count += 1
            if ch is not None:
                self._integrate_channel[integrate_id] = ch
            if depth_fold:
                self.depth_fold_count += 1
            self._supersede_sources_for_integrate(
                observation_a,
                observation_b,
                deactivate_sync_id=deactivate_sync_id,
                deactivate_integrate_ids=deactivate_integrate_ids,
                integrate_id=integrate_id,
            )

        emit_kind = "depth" if depth_fold else "integrate"
        self._emit_completed_set(emit_kind, store)
        self._stimulate_poles(store, kind="integrate")

        self.maybe_gc()

        if host is not None and host.outerface is not None:
            host.outerface.post({"kind": "active_integrates_changed"})

        return store

    def _stimulate_poles(self, store: dict[str, Thought], *, kind: str = "sense") -> None:
        """Raise activation on six-set poles after structure is completed."""
        host = self.host
        if host is None:
            return
        mind = getattr(host, "mind", None)
        if mind is None or not getattr(mind, "dynamics_enabled", True):
            return
        if kind == "integrate":
            amt = float(mind.integrate_pole_stimulus)
        else:
            amt = float(mind.formation_pole_stimulus)
        for t in six_set_poles(store):
            host.stimulate(t, amt)

    def _store_for_set_id(self, set_id: str) -> Optional[dict[str, Thought]]:
        with self._local_lock:
            if set_id in self.completed_formations:
                return self.completed_formations[set_id]
            if set_id in self.completed_syncs:
                return self.completed_syncs[set_id]
            if set_id in self.completed_integrates:
                return self.completed_integrates[set_id]
        return None

    def _set_valence(self, set_id: str) -> float:
        """Mean Mind valence of poles in a completed six-set (0 if unknown)."""
        host = self.host
        mind = getattr(host, "mind", None) if host is not None else None
        if mind is None:
            return 0.0
        store = self._store_for_set_id(set_id)
        if not store:
            return 0.0
        vals = [mind.valence_of(thought_id=tid) for tid in store]
        return sum(vals) / len(vals) if vals else 0.0

    def _lowest_valence_set_id(self, set_ids: list[str]) -> Optional[str]:
        if not set_ids:
            return None
        return min(set_ids, key=lambda sid: (self._set_valence(sid), sid))

    def _protected_thought_ids(self) -> set[str]:
        """
        Thoughts that must not be GC'd: seeds, laws, awareness, active six-sets,
        live sensor/actuator poles, last observations, Outerface beliefs.
        """
        host = self.host
        protected: set[str] = set()
        if host is None:
            return protected

        with host.graph_lock:
            twin = host.twin_seed_thoughts()
            if twin:
                protected.update(twin.keys())
            if getattr(host, "agent", None) is not None:
                protected.add(host.agent.id)
            if getattr(host, "system", None) is not None:
                protected.add(host.system.id)
            if getattr(host, "environment", None) is not None:
                protected.add(host.environment.id)
            for law in getattr(host, "laws", None) or []:
                link = law.link
                protected.add(link.id)
                protected.add(link.source.id)
                protected.add(link.link_type.id)
                protected.add(link.target.id)
            for store in (host.awareness_sets or {}).values():
                protected.update(store.keys())
            # Stable sensor / actuator grounding poles
            for sen in host.sensors:
                protected.add(f"{host.id}:sensor:{sen.id}")
            for act in host.actuators:
                protected.add(f"{host.id}:actuator:{act.id}")
                protected.add(f"{host.id}:sensor:{act.id}")  # awareness may use same pattern
            # Mind content-addressed Observations (recognition registry)
            mind = getattr(host, "mind", None)
            if mind is not None:
                protected.update(mind.registered_observation_ids())

        with self._local_lock:
            # All thoughts still in an *active* six-set
            for sid in list(self.active_ids.keys()):
                store = (
                    self.completed_formations.get(sid)
                    or self.completed_syncs.get(sid)
                    or self.completed_integrates.get(sid)
                )
                if store:
                    protected.update(store.keys())
            # Latest observation poles (may feed next temporal integrate)
            for obs in self._last_obs_by_sensor.values():
                protected.add(obs.id)

        # Outerface beliefs remain live expectations
        of = getattr(host, "outerface", None)
        if of is not None:
            with of._local_lock:
                for store in (getattr(of, "beliefs", None) or {}).values():
                    if isinstance(store, dict):
                        protected.update(store.keys())
                for bid in getattr(of, "active_belief_ids", None) or set():
                    store = (getattr(of, "beliefs", None) or {}).get(bid)
                    if isinstance(store, dict):
                        protected.update(store.keys())

        return protected

    def prune_inactive_thoughts(self) -> int:
        """
        Remove host Thoughts that belonged only to *inactive* six-sets after
        those sets were properly Integrated / depth-folded (Rodin halving of
        doubled sense/sync structures).

        Ghost scaffolding (Links + relation-type Thoughts from Rodin double/halve)
        is **always** removed from inactive sets. Poles stay only if still
        protected (active set, Mind registry, last obs, twin, laws, awareness).

        Returns number of Thoughts removed from ``host.thoughts``.
        """
        host = self.host
        if host is None:
            return 0

        protected = self._protected_thought_ids()

        # Candidates: thoughts that appear in at least one *inactive* completed set
        inactive_ids: set[str] = set()
        # Scaffold ghosts: force-remove even if a pole from the same store is protected
        scaffold_ids: set[str] = set()
        with self._local_lock:
            active = set(self.active_ids.keys())
            for sid, store in (
                list(self.completed_formations.items())
                + list(self.completed_syncs.items())
                + list(self.completed_integrates.items())
            ):
                if sid in active:
                    continue
                inactive_ids.update(store.keys())
                for tid, thought in store.items():
                    if is_scaffold_thought(thought):
                        scaffold_ids.add(tid)

        # Poles / shared nodes: only if not protected
        removable = (inactive_ids - protected) | (
            # Scaffold of inactive sets must leave the live graph (ghost GC).
            # Never delete if still part of an *active* six-set (in protected).
            scaffold_ids - protected
        )
        if not removable:
            return 0

        removed = 0
        with host.graph_lock:
            for tid in removable:
                if tid in host.thoughts:
                    del host.thoughts[tid]
                    removed += 1

        if removed:
            with self._local_lock:
                self.thoughts_pruned += removed
        # Archives (completed_formations / syncs / integrates) keep metadata for
        # ticks/stats; live graph is host.thoughts only.

        return removed

    def forget_cold_thoughts(self) -> int:
        """
        Activation-based forgetting: remove unprotected Thoughts that have been
        cold (near resting, no refractory) for ``Mind.forget_cold_cycles`` host
        pulse cycles since ``last_hot_cycle``.

        Default on (``Mind.forget_cold_enabled``). Only forgets Thoughts that
        were hot at least once (``last_hot_cycle >= 0``). With
        ``forget_transient_only=True``, only ``transient=True`` poles
        (default is False — any unprotected Thought).

        Reuses ``_protected_thought_ids`` (seeds, laws, Mind registry, active
        six-sets, last obs, beliefs). Also drops Port Links whose endpoint was
        forgotten, and orphan Links whose source/target is missing.

        Returns number of Thoughts removed from ``host.thoughts``.
        """
        host = self.host
        if host is None:
            return 0
        mind = getattr(host, "mind", None)
        if mind is None or not getattr(mind, "forget_cold_enabled", False):
            return 0

        eps = float(getattr(mind, "forget_activation_eps", 1e-6) or 1e-6)
        need = max(1, int(getattr(mind, "forget_cold_cycles", 64) or 64))
        transient_only = bool(getattr(mind, "forget_transient_only", True))
        max_n = max(1, int(getattr(mind, "forget_max_per_pass", 64) or 64))
        cycle = int(host.pulse_cycle)
        protected = self._protected_thought_ids()

        from symbioid.Core.Link import Link

        candidates: list[str] = []
        with host.graph_lock:
            for tid, t in list(host.thoughts.items()):
                if tid in protected:
                    continue
                if not getattr(t, "dynamics_enabled", True):
                    continue
                if isinstance(t, Link) and getattr(t, "is_port", False):
                    continue  # Port Links cleaned when endpoint forgotten
                if transient_only and not getattr(t, "transient", False):
                    continue
                last_hot = int(getattr(t, "last_hot_cycle", -1))
                if last_hot < 0:
                    continue  # never hot — leave to structure prune
                if cycle - last_hot < need:
                    continue
                if int(getattr(t, "refractory_ticks", 0) or 0) > 0:
                    continue
                act = float(t.activation)
                rest = float(getattr(t, "resting", 0.0) or 0.0)
                if abs(act - rest) > eps:
                    continue
                candidates.append(tid)
                if len(candidates) >= max_n:
                    break

            if not candidates:
                return 0

            remove: set[str] = set(candidates)
            # Port Links and orphan Links whose ends are gone
            for tid, t in list(host.thoughts.items()):
                if not isinstance(t, Link):
                    continue
                if tid in remove or tid in protected:
                    continue
                src_id = t.source.id if t.source is not None else None
                tgt_id = t.target.id if t.target is not None else None
                if getattr(t, "is_port", False):
                    if src_id in remove or tgt_id in remove:
                        remove.add(tid)
                    continue
                # orphan: either end already missing or being forgotten
                src_gone = src_id is None or src_id not in host.thoughts or src_id in remove
                tgt_gone = tgt_id is None or tgt_id not in host.thoughts or tgt_id in remove
                if src_gone or tgt_gone:
                    # only drop if at least one end is in this forget batch
                    # (avoid sweeping unrelated orphans mid-game)
                    if (src_id in remove) or (tgt_id in remove):
                        remove.add(tid)

            removed = 0
            for tid in remove:
                if tid in host.thoughts:
                    del host.thoughts[tid]
                    removed += 1
                host._hot_ids.discard(tid)

            # Drop from engine memberships
            for eng in (host.interface, host.innerface, host.outerface):
                mem = getattr(eng, "membership", None)
                if mem is None:
                    continue
                for tid in remove:
                    mem.discard(tid)

        if removed:
            with self._local_lock:
                self.thoughts_forgotten += removed
            mind.forgets_cold = int(getattr(mind, "forgets_cold", 0)) + removed
        return removed

    def maybe_gc(self) -> int:
        """Structure prune + cold-forget (when enabled). Returns total removed."""
        n = 0
        if self.auto_prune:
            n += self.prune_inactive_thoughts()
        n += self.forget_cold_thoughts()
        return n

    def _active_integrates_by_channel(self) -> dict[str, list[str]]:
        """Channel → ordered list of active integrate ids (oldest first)."""
        with self._local_lock:
            by_ch: dict[str, list[str]] = {}
            for sid, kind in self.active_ids.items():
                if kind != "integrate":
                    continue
                ch = self._integrate_channel.get(sid, "_")
                by_ch.setdefault(ch, []).append(sid)
            return by_ch

    def maybe_depth_fold(self, *, with_labels: bool = True) -> int:
        """
        H2 multi-step Rodin halving (integrate-of-integrates).

        When ``eager_depth_fold`` is True (default), repeatedly fold within each
        awareness channel until that channel has ≤ ``max_active_integrates_per_channel``
        active integrates, and overall ≤ ``max_active_integrates``.

        When eager is False, only fold while the *global* active-integrate count
        exceeds ``max_active_integrates`` (legacy soft-cap behaviour).

        Same-channel only (awareness terminators). Returns folds performed.
        """
        folds = 0
        deactivated = 0  # sets dropped or superseded (ghost candidates)
        global_limit = max(1, int(self.max_active_integrates))
        ch_limit = max(1, int(self.max_active_integrates_per_channel))

        for _ in range(256):
            by_ch = self._active_integrates_by_channel()
            total = sum(len(ids) for ids in by_ch.values())

            # Choose a channel that still needs compression
            pair: Optional[tuple[str, str, str]] = None
            if self.eager_depth_fold:
                # Prefer channels over their per-channel cap
                over = [
                    (ch, ids)
                    for ch, ids in by_ch.items()
                    if len(ids) > ch_limit
                ]
                if over:
                    # Fold the fullest channel first
                    over.sort(key=lambda t: len(t[1]), reverse=True)
                    ch, ids = over[0]
                    if len(ids) >= 2:
                        pair = (ids[0], ids[1], ch)
                elif total > global_limit:
                    # All channels within per-channel cap but global still high
                    for ch, ids in by_ch.items():
                        if len(ids) >= 2:
                            pair = (ids[0], ids[1], ch)
                            break
                else:
                    break  # both caps satisfied
            else:
                # Legacy: only act when global over soft cap
                if total <= global_limit:
                    break
                for ch, ids in by_ch.items():
                    if len(ids) >= 2:
                        pair = (ids[0], ids[1], ch)
                        break

            if pair is None:
                # Need to reduce but no same-channel pair — drop lowest-valence excess
                with self._local_lock:
                    int_ids = [
                        sid
                        for sid, kind in self.active_ids.items()
                        if kind == "integrate"
                    ]
                    if len(int_ids) > global_limit and int_ids:
                        drop = self._lowest_valence_set_id(int_ids)
                        self.active_ids.pop(drop or int_ids[0], None)
                        deactivated += 1
                    elif self.eager_depth_fold:
                        for ch, ids in by_ch.items():
                            if len(ids) > ch_limit and ids:
                                drop = self._lowest_valence_set_id(ids)
                                self.active_ids.pop(drop or ids[0], None)
                                deactivated += 1
                                break
                break

            a_id, b_id, ch = pair
            with self._local_lock:
                store_a = self.completed_integrates.get(a_id)
                store_b = self.completed_integrates.get(b_id)
            if not store_a or not store_b:
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                    self.active_ids.pop(b_id, None)
                    deactivated += 2
                continue

            poles_a = six_set_poles(store_a)
            poles_b = six_set_poles(store_b)
            if not poles_a or not poles_b:
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                    deactivated += 1
                continue

            rep_a = poles_a[0]
            rep_b = poles_b[0]
            if rep_a.id == rep_b.id and len(poles_b) > 1:
                rep_b = poles_b[1]
            if rep_a.id == rep_b.id and len(poles_a) > 1:
                rep_a = poles_a[1]

            pair_key = frozenset({f"depth:{a_id}", f"depth:{b_id}", f"g{folds}"})
            # Defer prune until the whole cascade finishes
            prev_auto = self.auto_prune
            self.auto_prune = False
            try:
                result = self.integrate_pair(
                    rep_a,
                    rep_b,
                    reason="depth",
                    with_labels=with_labels,
                    deactivate_integrate_ids=[a_id, b_id],
                    pair_key=pair_key,
                    depth_fold=True,
                    channel=ch if ch != "_" else None,
                )
            finally:
                self.auto_prune = prev_auto
            if result is None:
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                    deactivated += 1
                continue
            folds += 1
            deactivated += 2  # parent integrates superseded

        # Always GC ghost scaffolding after any supersede / excess drop
        if folds or deactivated:
            self.maybe_gc()
        return folds

    def accept_formation_batch(self, batch: dict[str, Any]) -> list[dict[str, Thought]]:
        """
        Complete several sensor formations then lateral-sync + integrate.
        Batch shape: {"kind": "formation_batch", "handoffs": [...], "tick": n}

        Multi-sensor: temporal integrate deferred until after Follows so lateral
        pairs (Follows → Integrates) take priority; unpaired sensors still get
        temporal last-two integrate afterward.
        """
        handoffs = list(batch.get("handoffs") or [])
        multi = len(handoffs) > 1
        stores: list[dict[str, Thought]] = []
        observations: list[Thought] = []
        pending_temporal: list[tuple[Thought, Thought, bool]] = []

        for h in handoffs:
            if not isinstance(h, dict):
                continue
            # Defer temporal integrate when batch has multiple sensors
            sensor_id = str(h.get("sensor_id") or "")
            partial = h.get("partial") or {}
            obs = partial.get("observation")
            prev: Optional[Thought] = None
            if isinstance(obs, Thought) and sensor_id:
                with self._local_lock:
                    prev = self._last_obs_by_sensor.get(sensor_id)
            store = self.accept_formation(h, integrate_temporal=not multi)
            stores.append(store)
            if isinstance(obs, Thought):
                observations.append(obs)
                if multi and prev is not None:
                    pending_temporal.append(
                        (prev, obs, bool(h.get("with_labels", True)))
                    )

        with_labels = True
        if handoffs and isinstance(handoffs[0], dict):
            with_labels = bool(handoffs[0].get("with_labels", True))

        if len(observations) >= 2:
            self.synchronize_observations(
                observations,
                tick=batch.get("tick"),
                with_labels=with_labels,
                integrate_pairs=True,
            )
        else:
            # Single-sensor batch: temporal already done in accept_formation
            pass

        # Temporal last-two for multi-sensor (after lateral integrate; pair keys skip dups)
        for prev, obs, wl in pending_temporal:
            self.integrate_pair(
                prev,
                obs,
                reason="temporal",
                tick=batch.get("tick"),
                with_labels=wl,
            )
        # Final depth plateau pass (H2)
        self.maybe_depth_fold(with_labels=with_labels)
        return stores

    def _drain_and_accept_formations(self) -> None:
        """Sparse structure lock-in: process Interface mint handoffs."""
        messages = self._drain_inbox()
        host = self.host
        if host is None or not host.mind.enabled:
            return

        by_tick: dict[Any, list[dict[str, Any]]] = {}
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            try:
                if msg.get("kind") == "formation_batch":
                    self.accept_formation_batch(msg)
                elif msg.get("kind") == "formation_handoff":
                    tkey = msg.get("tick", "_")
                    by_tick.setdefault(tkey, []).append(msg)
            except (ValueError, RuntimeError, TypeError) as exc:
                self.last_error = f"formation: {exc}"

        for tkey, group in by_tick.items():
            if len(group) > 1:
                self.accept_formation_batch(
                    {"kind": "formation_batch", "handoffs": group, "tick": tkey}
                )
            else:
                try:
                    self.accept_formation(group[0])
                except (ValueError, RuntimeError, TypeError) as exc:
                    self.last_error = f"formation: {exc}"

    def _pull_structure_pending(self) -> None:
        """
        Phase 4 spike: pull structure_pending from Interface (no inbox).
        hybrid still uses inbox drain.
        """
        host = self.host
        if host is None:
            return
        iface = getattr(host, "interface", None)
        if iface is None:
            return
        pending = list(getattr(iface, "structure_pending", None) or [])
        if not pending:
            return
        with iface._local_lock:
            iface.structure_pending.clear()
        if len(pending) > 1:
            self.accept_formation_batch(
                {
                    "kind": "formation_batch",
                    "handoffs": pending,
                    "tick": host.pulse_cycle,
                }
            )
        else:
            try:
                self.accept_formation(pending[0])
            except (ValueError, RuntimeError, TypeError) as exc:
                self.last_error = f"structure_pending: {exc}"

    def pre_ports(self) -> None:
        """Import Interface export charge + sparse structure consolidators."""
        host = self.host
        if host is None:
            return
        mode = self._engine_mode()
        if mode == "spike":
            # No inbox automata — structure only from Interface.structure_pending
            self._pull_structure_pending()
            # Drop any leftover inbox without full batch automata
            _ = self._drain_inbox()
        else:
            # hybrid: inbox mint handoffs
            self._drain_and_accept_formations()

        # Port-in: Phase 5 queue first, then last_export_ids fallback
        gain = float(self.port_gain)
        mind = host.mind
        if mind is not None:
            gain = float(getattr(mind, "port_gain", gain))
        packets = host.drain_port("interface", "innerface")
        if packets:
            n = host.apply_port_packets(packets, gain=gain, hebb=True)
            for pkt in packets:
                t = host.thoughts.get(pkt.thought_id)
                if t is not None:
                    self._register_members(t)
            with self._local_lock:
                self.port_imports += n
        else:
            # Compat: direct last_export_ids path (no queue yet)
            iface = getattr(host, "interface", None)
            if iface is not None:
                export_ids = list(getattr(iface, "last_export_ids", None) or [])
                for tid in export_ids:
                    t = host.thoughts.get(tid)
                    if t is None:
                        continue
                    amt = float(getattr(t, "export_activation", 0.0) or 0.0) * gain
                    if amt == 0:
                        continue
                    host.stimulate(t, amt)
                    self._register_members(t)
                    with self._local_lock:
                        self.port_imports += 1

        # Membership: last observations + active six-set poles
        with self._local_lock:
            for obs in self._last_obs_by_sensor.values():
                self._register_members(obs)
            for sid in list(self.active_ids.keys()):
                store = (
                    self.completed_formations.get(sid)
                    or self.completed_syncs.get(sid)
                    or self.completed_integrates.get(sid)
                )
                if store:
                    self._register_members(*six_set_poles(store))

    def pulse(self) -> dict[str, int]:
        """Membership pulse for binding engine (hybrid/spike)."""
        host = self.host
        if host is None:
            return {"cycle": 0, "hot": 0, "fired": 0, "spread": 0, "hebb": 0}
        if not getattr(host.mind, "dynamics_enabled", True):
            return {
                "cycle": host.pulse_cycle,
                "hot": 0,
                "fired": 0,
                "spread": 0,
                "hebb": 0,
            }
        self.use_membership = True
        if not self.membership:
            self.last_pulse_stats = {
                "cycle": host.pulse_cycle,
                "hot": 0,
                "fired": 0,
                "spread": 0,
                "hebb": 0,
                "engine": self.engine_name,
            }
            self.last_export_ids = []
            return self.last_pulse_stats
        return super().pulse()

    def consolidate_co_fires(self) -> int:
        """
        Replace bulk synchronize automata: Observations that fired this pulse
        get Follows (+ channel Integrates via existing helpers).
        """
        host = self.host
        if host is None:
            return 0
        cycle = host.pulse_cycle
        fired_obs: list[Thought] = []
        with host.graph_lock:
            for tid in list(self.membership):
                t = host.thoughts.get(tid)
                if t is None:
                    continue
                if int(getattr(t, "last_fired_cycle", -1)) != int(cycle):
                    continue
                # Prefer true Observations (channel map or transient)
                with self._local_lock:
                    is_obs = tid in self._obs_channel or t.transient
                if is_obs or (t.label and ":" in str(t.label)):
                    fired_obs.append(t)

        if len(fired_obs) < 2:
            # Single-channel temporal: if two last_obs different and both hot
            with self._local_lock:
                by_ch = list(self._last_obs_by_sensor.items())
            for sid, obs in by_ch:
                if int(getattr(obs, "last_fired_cycle", -1)) == int(cycle):
                    # try temporal integrate with previous if any — handled by accept
                    pass
            return 0

        # Dedup by id
        uniq: dict[str, Thought] = {t.id: t for t in fired_obs}
        obs_list = list(uniq.values())
        before = len(self.completed_syncs)
        self.synchronize_observations(
            obs_list,
            tick=cycle,
            with_labels=host.with_labels if host else True,
            integrate_pairs=True,
        )
        # Same-channel pairs among fired (temporal style)
        by_channel: dict[str, list[Thought]] = {}
        for t in obs_list:
            ch = self._channel_for_obs(t) or "_"
            by_channel.setdefault(ch, []).append(t)
        for ch, group in by_channel.items():
            if len(group) < 2 or ch == "_":
                continue
            # integrate all pairs same channel (admit_integrates dedupes)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    self.integrate_pair(
                        group[i],
                        group[j],
                        reason="cofire",
                        channel=ch if ch != "_" else None,
                        with_labels=host.with_labels if host else True,
                    )

        n = max(0, len(self.completed_syncs) - before)
        if n or len(obs_list) >= 2:
            with self._local_lock:
                self.co_fire_consolidations += 1
        return n

    def post_ports(self) -> None:
        """Co-fire consolidators + sparse depth-fold/prune."""
        host = self.host
        if host is None:
            return
        self.consolidate_co_fires()

        # Export for Outerface (Phase 3 + Phase 5 port queue)
        for tid in self.last_export_ids:
            t = host.thoughts.get(tid)
            if t is not None:
                t.export_activation = float(t.activation)
        if self.last_export_ids:
            host.export_port_packets(
                src_engine=self.engine_name,
                dst_engine="outerface",
                thought_ids=list(self.last_export_ids),
                cycle=host.pulse_cycle,
            )

        with self._local_lock:
            self.engine_ticks += 1
            ticks = self.engine_ticks
        every = max(1, int(self.consolidate_every))
        if ticks % every == 0:
            self.maybe_depth_fold(with_labels=host.with_labels if host else True)
            self.maybe_gc()

    def _process_body_legacy(self) -> None:
        """Original automata path: drain inbox and form/sync/integrate fully."""
        self._drain_and_accept_formations()

    def _process_body(self) -> None:
        """legacy automata vs hybrid/spike binding engine."""
        host = self.host
        if host is None or not host.mind.enabled:
            return
        if self._engine_mode() == "legacy":
            self._process_body_legacy()
            return
        self.use_membership = True
        self.pre_ports()
        self.pulse()
        self.post_ports()
