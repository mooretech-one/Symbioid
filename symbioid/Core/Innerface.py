"""Innerface process — complete formations, Follows sync, Rodin-halving integrate."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from symbioid.Core.Process import Process
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import (
    complete_follows_set,
    complete_formation,
    complete_integrate_set,
    emit_six_set,
    six_set_poles,
)
from symbioid.Core.ids import _new_id


@dataclass
class Innerface(Process):
    """
    Inner process (~3): Feeling / Reflection / Maker — Self, Mind, Thought.

    Completes Rodin formation (stages 4→8→7→5) after Interface hands off
    Sensor (Source) + Observation (Target). Builds lateral Follows /
    FollowedBy six-sets, then Rodin-halving Integrates pairs (Follows pairs
    and/or last-two Observations per sensor). H2: depth-fold active integrates
    until count ≤ max_active_integrates (plateau).
    """

    id: str = field(default_factory=lambda: _new_id("inner-"))
    label: Optional[str] = "Innerface"
    # H2: soft cap — only depth-fold when active integrates exceed this.
    # Default keeps many concurrent patterns (humans hold many beliefs/patterns).
    max_active_integrates: int = 24
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
    # Auto-GC after integrate (can disable for debugging)
    auto_prune: bool = True

    def process(self) -> Optional[threading.Thread]:
        thread = super().process()
        return thread

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
        """
        store = complete_formation(handoff)
        self._merge_store(store)
        fid = handoff["formation_id"]
        with self._local_lock:
            self.completed_formations[fid] = store
            self.formation_ticks += 1
        self._activate(fid, "sense")
        self._emit_completed_set("sense", store)

        partial = handoff.get("partial") or {}
        obs = partial.get("observation")
        sensor_id = str(handoff.get("sensor_id") or "")
        sensor_label = handoff.get("sensor_label")
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
            # Temporal integrate stays inside one sensor channel (terminator)
            if integrate_temporal and prev is not None:
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
        """
        if len(observations) < 2:
            return []
        ordered = sorted(observations, key=lambda t: (t.label or t.id, t.id))
        host = self.host
        host_id = host.id if host is not None else "host"
        tick_part = tick if tick is not None else self.cycles
        created: list[dict[str, Thought]] = []
        for i in range(len(ordered) - 1):
            a, b = ordered[i], ordered[i + 1]
            sync_id = f"{host_id}:sync:t{tick_part}:{a.id[-8:]}:{b.id[-8:]}"
            store = complete_follows_set(
                a, b, sync_id=sync_id, with_labels=with_labels
            )
            self._merge_store(store)
            with self._local_lock:
                self.completed_syncs[sync_id] = store
            self._activate(sync_id, "sync")
            self._emit_completed_set("sync", store)
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

        key = pair_key or frozenset({observation_a.id, observation_b.id})
        with self._local_lock:
            if key in self._integrated_pairs:
                return None
            self._integrated_pairs.add(key)

        host = self.host
        host_id = host.id if host is not None else "host"
        tick_part = tick if tick is not None else self.cycles
        integrate_id = (
            f"{host_id}:int:{reason}:t{tick_part}:"
            f"{observation_a.id[-8:]}:{observation_b.id[-8:]}"
        )
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

        self._merge_store(store)
        with self._local_lock:
            self.completed_integrates[integrate_id] = store
            self.integrate_count += 1
            if ch is not None:
                self._integrate_channel[integrate_id] = ch
            if depth_fold:
                self.depth_fold_count += 1
            # Supersede source sense formations
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

        emit_kind = "depth" if depth_fold else "integrate"
        self._emit_completed_set(emit_kind, store)

        # Drop scaffolding from six-sets no longer active (now integrated)
        if self.auto_prune:
            self.prune_inactive_thoughts()

        # Notify Outerface that active integrates changed (H3 harvest)
        if host is not None and host.outerface is not None:
            host.outerface.post({"kind": "active_integrates_changed"})

        # Depth plateau is applied at end of batch / accept (not mid-stream)
        return store

    def _store_for_set_id(self, set_id: str) -> Optional[dict[str, Thought]]:
        with self._local_lock:
            if set_id in self.completed_formations:
                return self.completed_formations[set_id]
            if set_id in self.completed_syncs:
                return self.completed_syncs[set_id]
            if set_id in self.completed_integrates:
                return self.completed_integrates[set_id]
        return None

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
        those sets were properly Integrated / depth-folded.

        Keeps: twin seed, constitution, awareness terminators, sensor poles,
        observations still poles of active integrates, Outerface beliefs,
        and any Thought still referenced by an active six-set.

        Returns number of Thoughts removed from ``host.thoughts``.
        """
        host = self.host
        if host is None:
            return 0

        protected = self._protected_thought_ids()

        # Candidates: thoughts that appear in at least one *inactive* completed set
        inactive_ids: set[str] = set()
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

        removable = inactive_ids - protected
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

    def maybe_depth_fold(self, *, with_labels: bool = True) -> int:
        """
        H2: while active integrate count > max_active_integrates, fold the two
        oldest active integrates into one depth integrate (plateau).
        Guarantees progress: if a fold cannot be built, drop the oldest active.
        Returns number of depth folds performed this call.
        """
        folds = 0
        for _ in range(128):
            with self._local_lock:
                int_ids = [
                    sid for sid, kind in self.active_ids.items() if kind == "integrate"
                ]
                limit = max(1, int(self.max_active_integrates))
            if len(int_ids) <= limit:
                break
            if len(int_ids) < 2:
                # Single extra integrate above limit shouldn't happen if limit>=1
                with self._local_lock:
                    if int_ids:
                        self.active_ids.pop(int_ids[0], None)
                break

            # Only depth-fold within the same awareness channel (terminator)
            with self._local_lock:
                by_ch: dict[str, list[str]] = {}
                for sid in int_ids:
                    ch = self._integrate_channel.get(sid, "_")
                    by_ch.setdefault(ch, []).append(sid)
                pair: Optional[tuple[str, str, str]] = None
                for ch, ids in by_ch.items():
                    if len(ids) >= 2:
                        pair = (ids[0], ids[1], ch)
                        break
            if pair is None:
                # No same-channel pair to fold — stop (do not cross terminators)
                break
            a_id, b_id, ch = pair
            with self._local_lock:
                store_a = self.completed_integrates.get(a_id)
                store_b = self.completed_integrates.get(b_id)
            if not store_a or not store_b:
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                    self.active_ids.pop(b_id, None)
                continue

            poles_a = six_set_poles(store_a)
            poles_b = six_set_poles(store_b)
            if not poles_a or not poles_b:
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                continue

            # Prefer distinct representative poles (avoid self-pair)
            rep_a = poles_a[0]
            rep_b = poles_b[0]
            if rep_a.id == rep_b.id and len(poles_b) > 1:
                rep_b = poles_b[1]
            if rep_a.id == rep_b.id and len(poles_a) > 1:
                rep_a = poles_a[1]

            # Unique key per fold attempt (parent integrate ids + generation)
            pair_key = frozenset({f"depth:{a_id}", f"depth:{b_id}", f"g{folds}"})
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
            if result is None:
                # Guarantee progress toward plateau
                with self._local_lock:
                    self.active_ids.pop(a_id, None)
                continue
            folds += 1
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

    def _process_body(self) -> None:
        """Drain inbox; complete formations, Follows sync, and integrates."""
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
