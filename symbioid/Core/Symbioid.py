"""Symbioid host — twin System ⋈ Environment with aspects and faces."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterator, Optional, Set

from symbioid.Core.Actuator import Actuator
from symbioid.Core.Body import Body
from symbioid.Core.Energy import Energy
from symbioid.Core.Innerface import Innerface
from symbioid.Core.Interface import Interface
from symbioid.Core.Law import Law, constitutional_seed
from symbioid.Core.Link import Link
from symbioid.Core.Mind import Mind
from symbioid.Core.Outerface import Outerface
from symbioid.Core.Sensor import Sensor
from symbioid.Core.SpikingEngine import PortPacket
from symbioid.Core.System import System
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import complete_awareness_set, emit_six_set, ensure_sensor_thought
from symbioid.Core.ids import _new_id
from symbioid.Core.seed import is_minimal_symbioid_shape, minimal_seed
from symbioid.Core.thought_layers import assert_mind_not_thought

# legacy = single global pulse (current faces); hybrid/spike reserved for engine migration
ENGINES_MODES = ("legacy", "hybrid", "spike")
# Phase 5 port channels (src>dst)
PORT_I_N = "interface>innerface"
PORT_N_O = "innerface>outerface"


@dataclass
class Symbioid(System):
    """
    Siamese twin System ⋈ Environment; also a System (not a Thought).

    Contained aspects (Antelligence):
      body, mind, energy, sensors[], actuators[], thoughts{} (list via thought_list),
      innerface, interface, outerface

    Structural twin seed (six Thoughts) lives in `thoughts` when seed_minimal=True.
    Constitution (Asimov-shaped laws) installs when install_constitution=True
    (default): STABLE Law Links for Outerface gating — not the same as twin seed.

    engines_mode:
      legacy — faces as today; pulse_tick = full-graph partition
      hybrid — faces become SpikingEngines (phased migration)
      spike  — automata demoted; engines primary (later phases)

    Phase 5: port_queues decouple I→N→O; pulse_partition energy_budget;
    cross-engine Hebb only on Port Links (is_port=True).

    Architecture MVP: Mind≠Thought enforced; Thought layers Structure/Pattern/Feeling;
    act_from_graph / think_tick are core Outerface agency (not demo-only).
    """

    id: str = field(default_factory=lambda: _new_id("sym-"))
    seed_minimal: bool = True
    with_labels: bool = True
    mirror_in_environment: bool = False
    install_constitution: bool = True
    engines_mode: str = "legacy"
    # When True, pulse_tick spends host.energy (nested budget) if energy_budget not set
    energy_enforced: bool = False

    body: Body = field(default_factory=Body)
    mind: Mind = field(default_factory=Mind)
    energy: Energy = field(default_factory=Energy)
    sensors: list[Sensor] = field(default_factory=list)
    actuators: list[Actuator] = field(default_factory=list)
    innerface: Innerface = field(default_factory=Innerface)
    interface: Interface = field(default_factory=Interface)
    outerface: Outerface = field(default_factory=Outerface)
    # aspect_id → awareness six-set (Agent Has Sensor/Actuator) — integration terminators
    awareness_sets: dict[str, dict[str, Thought]] = field(
        default_factory=dict, init=False, repr=False
    )
    # aspect ids that bound integration channels (sensors + actuators)
    integration_terminators: set[str] = field(default_factory=set, init=False, repr=False)

    system: Thought = field(init=False, repr=False)
    environment: Thought = field(init=False, repr=False)
    thoughts: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    env_thoughts: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    laws: list[Law] = field(default_factory=list, init=False, repr=False)
    agent: Thought = field(init=False, repr=False)
    graph_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)
    # Dynamics: ids with non-rest activation / refractory (optimized pulse_tick)
    _hot_ids: set[str] = field(default_factory=set, init=False, repr=False)
    # source_id → set of non-Port Link ids (spread adjacency; maintained on add/remove)
    _out_by_source: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _out_index_dirty: bool = field(default=True, init=False, repr=False)
    pulse_cycle: int = field(default=0, init=False, repr=False)
    last_pulse_fired: int = field(default=0, init=False, repr=False)
    last_pulse_hot: int = field(default=0, init=False, repr=False)
    last_hebb_updates: int = field(default=0, init=False, repr=False)
    # Phase 5: channel key → FIFO of PortPacket (thread-safe under graph_lock)
    port_queues: dict[str, deque] = field(default_factory=dict, init=False, repr=False)
    last_port_hebb: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.body.label is None:
            self.body.label = "Body"
        if self.mind.label is None:
            self.mind.label = "Mind"
        if self.energy.label is None:
            self.energy.label = "Energy"
        # Architecture MVP: Mind is substrate, never Thought content
        assert_mind_not_thought(self.mind)
        self.innerface.host = self
        self.interface.host = self
        self.outerface.host = self
        if self.seed_minimal:
            self.seed_self_description()
        else:
            self.system = Thought(
                id=f"{self.id}:system",
                label="System" if self.with_labels else None,
                threshold=10.0,
            )
            self.environment = Thought(
                id=f"{self.id}:environment",
                label="Environment" if self.with_labels else None,
                threshold=10.0,
            )
            self.thoughts = {
                self.system.id: self.system,
                self.environment.id: self.environment,
            }
        self.agent = Thought(
            id=f"{self.id}:agent",
            label="Agent" if self.with_labels else None,
            threshold=10.0,
        )
        self.thoughts[self.agent.id] = self.agent
        if self.install_constitution:
            self.install_laws()
        self.rebuild_out_index()

    def start_processes(self) -> list[threading.Thread]:
        """Start Innerface, Interface, Outerface workers (each calls process())."""
        threads: list[threading.Thread] = []
        for proc in (self.innerface, self.interface, self.outerface):
            t = proc.process()
            if t is not None:
                threads.append(t)
        return threads

    def stop_processes(self, timeout: float = 1.0) -> None:
        """Stop all face processes (order: Outer → Inter → Inner to drain agency first)."""
        for proc in (self.outerface, self.interface, self.innerface):
            proc.stop(timeout=timeout)

    def mark_hot(self, thought: Thought) -> None:
        """Register Thought for dynamics processing after receive."""
        if thought is None or not getattr(thought, "dynamics_enabled", True):
            return
        with self.graph_lock:
            self._hot_ids.add(thought.id)
            thought.last_hot_cycle = int(self.pulse_cycle)

    def stimulate(self, thought: Thought, amount: float) -> None:
        """receive + mark hot (thread-safe)."""
        if thought is None or amount == 0:
            return
        mind = self.mind
        if mind is not None and not getattr(mind, "dynamics_enabled", True):
            return
        thought.receive(amount)
        self.mark_hot(thought)

    def pulse_tick(self) -> dict:
        """
        Full-graph dynamics tick (legacy global engine).

        Implemented as ``pulse_partition(membership=None)`` so future face
        engines share the same core path.
        """
        return self.pulse_partition(membership=None, engine_name="global")

    def run_engines(self) -> dict[str, dict]:
        """
        Serial I → N → O engine ticks (hybrid/spike).

        Phase 5: engines exchange via port_queues (not only last_export_ids).
        Order remains causal I then N then O; queues make concurrent face
        threads safe when used with Process workers.

        In legacy mode, runs each face's process body (still valid) and a
        global pulse_tick once if dynamics on.
        Returns per-engine last_pulse_stats where available.
        """
        mode = getattr(self, "engines_mode", "legacy") or "legacy"
        out: dict[str, dict] = {}
        if mode == "legacy":
            self.interface._process_body()
            self.innerface._process_body()
            self.outerface._process_body()
            if getattr(self.mind, "dynamics_enabled", True):
                out["global"] = self.pulse_tick()
            return out

        # hybrid / spike: each face body is already engine-shaped
        self.interface._process_body()
        out["interface"] = dict(getattr(self.interface, "last_pulse_stats", {}) or {})
        self.innerface._process_body()
        out["innerface"] = dict(getattr(self.innerface, "last_pulse_stats", {}) or {})
        self.outerface._process_body()
        out["outerface"] = dict(getattr(self.outerface, "last_pulse_stats", {}) or {})
        return out

    # --- Phase 5: port queues -------------------------------------------------

    @staticmethod
    def port_channel(src_engine: str, dst_engine: str) -> str:
        return f"{src_engine}>{dst_engine}"

    def push_port(
        self,
        src_engine: str,
        dst_engine: str,
        packet: PortPacket,
    ) -> None:
        """Enqueue a PortPacket (drops oldest if over Mind.port_queue_max)."""
        key = packet.channel or self.port_channel(src_engine, dst_engine)
        mind = self.mind
        cap = int(getattr(mind, "port_queue_max", 256) if mind else 256)
        with self.graph_lock:
            q = self.port_queues.setdefault(key, deque())
            q.append(packet)
            while cap > 0 and len(q) > cap:
                q.popleft()

    def drain_port(
        self,
        src_engine: str,
        dst_engine: str,
        *,
        max_n: Optional[int] = None,
    ) -> list[PortPacket]:
        """Pop all (or up to max_n) packets from channel."""
        key = self.port_channel(src_engine, dst_engine)
        out: list[PortPacket] = []
        with self.graph_lock:
            q = self.port_queues.get(key)
            if not q:
                return out
            n = len(q) if max_n is None else min(len(q), int(max_n))
            for _ in range(n):
                out.append(q.popleft())
        return out

    def port_queue_len(self, src_engine: str, dst_engine: str) -> int:
        key = self.port_channel(src_engine, dst_engine)
        with self.graph_lock:
            q = self.port_queues.get(key)
            return len(q) if q else 0

    def ensure_port_link(
        self,
        thought: Thought,
        *,
        channel: str,
    ) -> Link:
        """
        Ensure a Port Link for cross-engine transfer gain / Hebb.

        Port Links do not participate in pulse spread (is_port=True).
        Self-edge: source=target=thought; weight multiplies port_gain on import.
        """
        link_id = f"{self.id}:port:{channel}:{thought.id}"
        with self.graph_lock:
            existing = self.thoughts.get(link_id)
            if isinstance(existing, Link) and getattr(existing, "is_port", False):
                return existing
            type_id = f"{self.id}:port-type"
            port_type = self.thoughts.get(type_id)
            if port_type is None:
                port_type = Thought(
                    id=type_id,
                    label="Port" if self.with_labels else None,
                    threshold=10.0,
                    dynamics_enabled=False,
                )
                self._register_in_store_unlocked(port_type)
            link = Link(
                id=link_id,
                label=f"Port[{channel}]" if self.with_labels else None,
                source=thought,
                link_type=port_type,
                target=thought,
                weight=1.0,
                threshold=10.0,
                is_port=True,
                dynamics_enabled=False,
            )
            self._register_in_store_unlocked(link)
            return link

    def export_port_packets(
        self,
        *,
        src_engine: str,
        dst_engine: str,
        thought_ids: list[str],
        cycle: int | None = None,
    ) -> int:
        """
        Snapshot export_activation and push PortPackets for firers.
        Returns number of packets enqueued.
        """
        if not thought_ids:
            return 0
        ch = self.port_channel(src_engine, dst_engine)
        cyc = int(cycle if cycle is not None else self.pulse_cycle)
        n = 0
        for tid in thought_ids:
            t = self.thoughts.get(tid)
            if t is None:
                continue
            act = float(getattr(t, "export_activation", 0.0) or 0.0)
            if act == 0.0:
                act = float(t.activation)
            if act == 0.0:
                continue
            t.export_activation = act
            self.push_port(
                src_engine,
                dst_engine,
                PortPacket(
                    thought_id=tid,
                    activation=act,
                    source_engine=src_engine,
                    cycle=cyc,
                    channel=ch,
                ),
            )
            n += 1
        return n

    def apply_port_packets(
        self,
        packets: list[PortPacket],
        *,
        gain: float | None = None,
        hebb: bool = True,
    ) -> int:
        """
        Stimulate Thoughts from PortPackets; optional Port-Link Hebb.
        Returns number of successful imports.
        """
        if not packets:
            return 0
        mind = self.mind
        g = float(
            gain
            if gain is not None
            else (getattr(mind, "port_gain", 0.55) if mind else 0.55)
        )
        hebb_on = bool(hebb) and bool(
            getattr(mind, "hebb_enabled", True) if mind else True
        )
        port_lr = float(getattr(mind, "port_hebb_lr", 0.05) if mind else 0.05)
        w_min = float(getattr(mind, "weight_min", 0.05) if mind else 0.05)
        w_max = float(getattr(mind, "weight_max", 4.0) if mind else 4.0)
        imported = 0
        hebb_n = 0
        for pkt in packets:
            t = self.thoughts.get(pkt.thought_id)
            if t is None:
                continue
            ch = pkt.channel or self.port_channel(pkt.source_engine, "unknown")
            link = self.ensure_port_link(t, channel=ch)
            w = float(getattr(link, "weight", 1.0))
            amt = float(pkt.activation) * g * w
            if amt == 0:
                continue
            self.stimulate(t, amt)
            imported += 1
            if hebb_on and hasattr(link, "adjust_weight"):
                # Successful transfer → slight Port-channel Hebb
                link.adjust_weight(port_lr, w_min=w_min, w_max=w_max)
                hebb_n += 1
        self.last_port_hebb = hebb_n
        if mind is not None and hebb_n:
            mind.hebb_updates = int(getattr(mind, "hebb_updates", 0)) + hebb_n
        return imported

    def pulse_partition(
        self,
        membership: Optional[Set[str]] = None,
        *,
        synapse_filter: Optional[Callable[[Link], bool]] = None,
        engine_name: str = "global",
        hebb: Optional[bool] = None,
        owner_only: bool = False,
        energy_budget: Optional[float] = None,
    ) -> dict:
        """
        Masked spiking tick: decay → fire → (optional) one-hop spread → Hebb.

        Mind.dynamics_mode:
          graph    — spread + Hebb; no FFT mix
          hybrid   — spread + Hebb + optional FFT residual (default)
          spectral — no Link spread/Hebb; FFT mix is the associative path (Mode B)

        membership:
          None  — all Thoughts (legacy pulse_tick)
          set   — only those ids may decay/fire; spread only from those firers
                  (targets may lie outside membership and still receive)

        synapse_filter:
          if set, only Links passing the predicate are used for spread/Hebb.

        owner_only:
          if True, skip try_fire unless thought.engine_owner in (None, engine_name).

        energy_budget:
          Phase 5 — max energy units for this partition (None = unlimited).
          Fire costs Mind.energy_fire_cost; spread costs energy_spread_cost.

        Hebb (Phase 5, when membership set and Mind.port_hebb_cross_only):
          non-Port Links Hebb only if target is in membership;
          Port Links never spread; their Hebb is via apply_port_packets.
        """
        mind = self.mind
        if mind is not None and not getattr(mind, "dynamics_enabled", True):
            return {
                "cycle": self.pulse_cycle,
                "hot": 0,
                "fired": 0,
                "spread": 0,
                "hebb": 0,
                "engine": engine_name,
                "energy_used": 0.0,
                "energy_left": 0.0,
                "energy_capped": 0,
                "dynamics_mode": getattr(mind, "dynamics_mode", "hybrid"),
            }

        # Phase 2: vector CPU pulse (full-graph unrestricted only)
        backend = (
            mind.normalize_dynamics_backend()
            if mind is not None and hasattr(mind, "normalize_dynamics_backend")
            else "object"
        )
        if backend == "vector":
            from symbioid.Core.vector_pulse import (
                can_use_vector_pulse,
                pulse_partition_vector,
            )

            if can_use_vector_pulse(
                membership=membership,
                synapse_filter=synapse_filter,
                owner_only=owner_only,
                energy_budget=energy_budget,
            ):
                with self.graph_lock:
                    stats_v = pulse_partition_vector(
                        self, engine_name=engine_name, hebb=hebb
                    )
                # Spectral residual mix (same as object path tail)
                want_mix = (
                    mind.spectral_mix_wanted()
                    if mind is not None and hasattr(mind, "spectral_mix_wanted")
                    else bool(
                        getattr(mind, "spectral_mix_enabled", False) if mind else False
                    )
                )
                if mind is not None and want_mix and engine_name in ("innerface", "global"):
                    dyn_mode_v = (
                        mind.normalize_dynamics_mode()
                        if hasattr(mind, "normalize_dynamics_mode")
                        else "hybrid"
                    )
                    cands = list(self._hot_ids)
                    if dyn_mode_v == "spectral" and float(
                        getattr(mind, "spectral_mix_gain", 0.0) or 0.0
                    ) == 0.0:
                        mind.spectral_mix_gain = 0.15
                    mix = mind.spectral_mix_step(self, candidate_ids=cands or None)
                    stats_v = dict(stats_v)
                    stats_v["spectral"] = mix
                return stats_v
            # else fall through to object path (masked / energy / filter)

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
        # Mode B spectral: no Link Hebb even if hebb_enabled
        if not graph_spread:
            hebb_on = False
        hebb_lr = float(getattr(mind, "hebb_lr", 0.08) if mind else 0.08)
        co_scale = float(getattr(mind, "hebb_co_fire_scale", 1.0) if mind else 1.0)
        pre_post = float(getattr(mind, "hebb_pre_post_scale", 0.35) if mind else 0.35)
        w_min = float(getattr(mind, "weight_min", 0.05) if mind else 0.05)
        w_max = float(getattr(mind, "weight_max", 4.0) if mind else 4.0)
        fire_cost = float(getattr(mind, "energy_fire_cost", 1.0) if mind else 1.0)
        spread_cost = float(getattr(mind, "energy_spread_cost", 0.25) if mind else 0.25)
        cross_only = bool(
            getattr(mind, "port_hebb_cross_only", True) if mind else True
        )
        energy_left = float(energy_budget) if energy_budget is not None else float("inf")
        energy_start = energy_left
        energy_capped = 0

        with self.graph_lock:
            self.pulse_cycle += 1
            cycle = self.pulse_cycle

            # Adjacency index (lazy rebuild if external code mutated thoughts{})
            if self._out_index_dirty:
                self._rebuild_out_index_unlocked()

            # Eligible ids for decay/fire
            # Full-graph: trust _hot_ids only (O(hot)) — no all-Thoughts is_hot scan.
            if membership is None:
                eligible = None  # type: ignore[assignment]
                hot_ids = {tid for tid in self._hot_ids if tid in self.thoughts}
                if len(hot_ids) != len(self._hot_ids):
                    self._hot_ids = set(hot_ids)
            else:
                eligible = set(membership) & set(self.thoughts.keys())
                hot_ids = (set(self._hot_ids) & eligible) | {
                    tid
                    for tid in eligible
                    if (t := self.thoughts.get(tid)) is not None and t.is_hot()
                }

            def _in_eligible(tid: str) -> bool:
                return eligible is None or tid in eligible

            # 1) Decay only hot (∩ eligible when membership set)
            still_hot: set[str] = set()
            for tid in hot_ids:
                t = self.thoughts.get(tid)
                if t is None or not _in_eligible(tid):
                    continue
                t.decay_step()
                if t.is_hot():
                    still_hot.add(tid)

            # 2) Fire (energy-gated)
            firers: list[Thought] = []
            candidates = list(still_hot) + [i for i in hot_ids if i not in still_hot]
            for tid in candidates:
                if not _in_eligible(tid):
                    continue
                t = self.thoughts.get(tid)
                if t is None:
                    continue
                if owner_only:
                    owner = getattr(t, "engine_owner", None)
                    if owner is not None and owner != engine_name:
                        continue
                # Would-fire check before spending energy
                if t.refractory_ticks > 0:
                    continue
                if float(t.activation) < float(t.threshold):
                    continue
                if energy_left < fire_cost:
                    energy_capped += 1
                    continue
                if t.try_fire(cycle=cycle):
                    energy_left -= fire_cost
                    firers.append(t)
                    still_hot.add(t.id)
                    # Claim ownership only when partitions enforce owner_only
                    if owner_only and t.engine_owner is None:
                        t.engine_owner = engine_name

            firer_ids = {f.id for f in firers}

            # 3) One-hop spread + Hebb (skipped in dynamics_mode=spectral / Mode B)
            spread = 0
            hebb_n = 0
            if graph_spread:
                for firer in firers:
                    strength = max(float(firer.threshold), float(firer.activation))
                    for link in self._outgoing_links_unlocked(
                        firer.id, synapse_filter=synapse_filter
                    ):
                        tgt = link.target
                        if tgt is None:
                            continue
                        w = float(getattr(link, "weight", 1.0))
                        amt = strength * w * gain
                        if amt != 0:
                            if energy_left < spread_cost:
                                energy_capped += 1
                            else:
                                energy_left -= spread_cost
                                tgt.receive(amt)
                                still_hot.add(tgt.id)
                                self._register_in_store_unlocked(tgt)
                                # Targets that receive become globally hot
                                self._hot_ids.add(tgt.id)
                                spread += 1

                        if not hebb_on or not hasattr(link, "adjust_weight"):
                            continue
                        # Phase 5: under membership, non-Port Hebb only if target in set
                        if (
                            membership is not None
                            and cross_only
                            and tgt.id not in eligible  # type: ignore[operator]
                        ):
                            continue
                        if tgt.id in firer_ids:
                            delta = hebb_lr * co_scale
                        elif float(tgt.activation) >= 0.5 * float(tgt.threshold):
                            delta = hebb_lr * pre_post
                        else:
                            continue
                        # Phase 4: phase-locked Hebb scale (no-op when disabled)
                        if mind is not None and hasattr(mind, "phase_hebb_scale"):
                            delta *= float(mind.phase_hebb_scale(firer, tgt))
                        link.adjust_weight(delta, w_min=w_min, w_max=w_max)
                        hebb_n += 1

            # Hot set: retain non-eligible hot ids; update eligible from still_hot
            if membership is None:
                self._hot_ids = {i for i in still_hot if i in self.thoughts}
            else:
                retained = {i for i in self._hot_ids if i not in eligible}  # type: ignore[operator]
                self._hot_ids = retained | {i for i in still_hot if i in self.thoughts}

            # Cold-forget age: stamp last_hot_cycle on anything still hot this tick
            for hid in self._hot_ids:
                ht = self.thoughts.get(hid)
                if ht is not None:
                    ht.last_hot_cycle = cycle
            for firer in firers:
                firer.last_hot_cycle = cycle

            self.last_pulse_fired = len(firers)
            self.last_pulse_hot = len(self._hot_ids)
            self.last_hebb_updates = hebb_n
            if mind is not None and hebb_n:
                mind.hebb_updates = int(getattr(mind, "hebb_updates", 0)) + hebb_n

            energy_used = (
                0.0
                if energy_start == float("inf")
                else max(0.0, energy_start - energy_left)
            )
            energy_rem = 0.0 if energy_left == float("inf") else max(0.0, energy_left)

        stats: dict = {
            "cycle": cycle,
            "hot": self.last_pulse_hot,
            "fired": self.last_pulse_fired,
            "spread": spread,
            "hebb": self.last_hebb_updates,
            "engine": engine_name,
            "energy_used": energy_used,
            "energy_left": energy_rem,
            "energy_capped": energy_capped,
            "dynamics_mode": dyn_mode,
            "graph_spread": bool(graph_spread),
        }
        # FFT residual mix: hybrid when enabled; spectral Mode B always
        want_mix = (
            mind.spectral_mix_wanted()
            if mind is not None and hasattr(mind, "spectral_mix_wanted")
            else bool(getattr(mind, "spectral_mix_enabled", False) if mind else False)
        )
        if mind is not None and want_mix:
            if engine_name in ("innerface", "global"):
                cands = list(self._hot_ids)
                if membership is not None:
                    cands = list(set(cands) | set(membership))
                # Mode B: ensure mix is not soft-disabled by zero gain
                if dyn_mode == "spectral" and float(
                    getattr(mind, "spectral_mix_gain", 0.0) or 0.0
                ) == 0.0:
                    mind.spectral_mix_gain = 0.15
                mix = mind.spectral_mix_step(self, candidate_ids=cands or None)
                stats["spectral"] = mix
        return stats

    def reinforce_edge(
        self,
        source: Thought,
        target: Thought,
        *,
        delta: float | None = None,
    ) -> int:
        """
        Nudge weight on all Links source→target (and optionally used after outcomes).
        Returns number of Links updated.
        """
        mind = self.mind
        if mind is not None and not getattr(mind, "hebb_enabled", True):
            return 0
        if delta is None:
            delta = float(getattr(mind, "outcome_weight_lr", 0.05) if mind else 0.05)
        w_min = float(getattr(mind, "weight_min", 0.05) if mind else 0.05)
        w_max = float(getattr(mind, "weight_max", 4.0) if mind else 4.0)
        n = 0
        with self.graph_lock:
            if self._out_index_dirty:
                self._rebuild_out_index_unlocked()
            for link in self._outgoing_links_unlocked(source.id):
                if link.target is not None and link.target.id == target.id:
                    link.adjust_weight(float(delta), w_min=w_min, w_max=w_max)
                    n += 1
        if mind is not None and n:
            mind.hebb_updates = int(getattr(mind, "hebb_updates", 0)) + n
        return n

    def ensure_reciprocal_links(
        self,
        a: Thought,
        b: Thought,
        *,
        initial_weight: float = 1.0,
        rel_label: str = "Associates",
    ) -> tuple[Link, Link]:
        """
        Ensure A→B and B→A Links exist for policy / co-fire plasticity.
        Reuses existing edges if present.
        """
        mind = self.mind
        w_min = float(getattr(mind, "weight_min", 0.05) if mind else 0.05)
        w_max = float(getattr(mind, "weight_max", 4.0) if mind else 4.0)
        w0 = max(w_min, min(w_max, float(initial_weight)))

        def _find(src: Thought, tgt: Thought) -> Optional[Link]:
            for link in self._outgoing_links_unlocked(src.id):
                if link.target is not None and link.target.id == tgt.id:
                    return link
            return None

        with self.graph_lock:
            if self._out_index_dirty:
                self._rebuild_out_index_unlocked()
            self._register_in_store_unlocked(a)
            self._register_in_store_unlocked(b)
            ab = _find(a, b)
            ba = _find(b, a)
            if ab is None:
                lt = Thought(
                    id=f"{self.id}:rel:{a.id[-8:]}:{b.id[-8:]}:t",
                    label=rel_label if self.with_labels else None,
                    threshold=10.0,
                )
                ab = Link(
                    id=f"{self.id}:edge:{a.id[-8:]}:{b.id[-8:]}",
                    label=f"{rel_label}AB" if self.with_labels else None,
                    source=a,
                    link_type=lt,
                    target=b,
                    weight=w0,
                    threshold=10.0,
                )
                self._register_in_store_unlocked(lt)
                self._register_in_store_unlocked(ab)
            else:
                ab.weight = max(ab.weight, w0)
            if ba is None:
                lt2 = Thought(
                    id=f"{self.id}:rel:{b.id[-8:]}:{a.id[-8:]}:t",
                    label=rel_label if self.with_labels else None,
                    threshold=10.0,
                )
                ba = Link(
                    id=f"{self.id}:edge:{b.id[-8:]}:{a.id[-8:]}",
                    label=f"{rel_label}BA" if self.with_labels else None,
                    source=b,
                    link_type=lt2,
                    target=a,
                    weight=w0,
                    threshold=10.0,
                )
                self._register_in_store_unlocked(lt2)
                self._register_in_store_unlocked(ba)
            else:
                ba.weight = max(ba.weight, w0)
        return ab, ba

    def _index_link_unlocked(self, thought: Thought) -> None:
        if not isinstance(thought, Link) or getattr(thought, "is_port", False):
            return
        src = thought.source
        if src is None:
            return
        self._out_by_source.setdefault(src.id, set()).add(thought.id)

    def _unindex_link_unlocked(self, thought: Thought) -> None:
        if not isinstance(thought, Link):
            return
        src = thought.source
        if src is None:
            return
        bucket = self._out_by_source.get(src.id)
        if not bucket:
            return
        bucket.discard(thought.id)
        if not bucket:
            self._out_by_source.pop(src.id, None)

    def _register_in_store_unlocked(self, thought: Thought) -> None:
        """thoughts[id]= + adjacency index (caller holds graph_lock)."""
        old = self.thoughts.get(thought.id)
        if old is not None and old is not thought:
            self._unindex_link_unlocked(old)
        self.thoughts[thought.id] = thought
        self._index_link_unlocked(thought)
        if thought.is_hot():
            self._hot_ids.add(thought.id)

    def _remove_thought_unlocked(self, thought_id: str) -> Optional[Thought]:
        t = self.thoughts.pop(thought_id, None)
        if t is None:
            return None
        self._unindex_link_unlocked(t)
        self._hot_ids.discard(thought_id)
        return t

    def remove_thought(self, thought_id: str) -> Optional[Thought]:
        """Remove a Thought/Link and keep adjacency / hot sets coherent."""
        with self.graph_lock:
            return self._remove_thought_unlocked(thought_id)

    def _rebuild_out_index_unlocked(self) -> None:
        self._out_by_source.clear()
        for t in self.thoughts.values():
            self._index_link_unlocked(t)
        self._out_index_dirty = False

    def rebuild_out_index(self) -> None:
        """Full adjacency rebuild (after bulk store replace / seed)."""
        with self.graph_lock:
            self._rebuild_out_index_unlocked()

    def mark_out_index_dirty(self) -> None:
        """Call after external bulk mutations of ``thoughts`` without add/remove."""
        self._out_index_dirty = True

    def _outgoing_links_unlocked(
        self,
        source_id: str,
        *,
        synapse_filter: Optional[Callable[[Link], bool]] = None,
    ) -> list[Link]:
        """Non-port Links from source via adjacency index (caller holds lock)."""
        out: list[Link] = []
        bucket = self._out_by_source.get(source_id)
        if not bucket:
            return out
        stale: list[str] = []
        for lid in bucket:
            t = self.thoughts.get(lid)
            if not isinstance(t, Link) or getattr(t, "is_port", False):
                stale.append(lid)
                continue
            if synapse_filter is not None and not synapse_filter(t):
                continue
            out.append(t)
        for lid in stale:
            bucket.discard(lid)
        if not bucket and source_id in self._out_by_source:
            del self._out_by_source[source_id]
        return out

    def add_thought(self, thought: Thought) -> None:
        """Register a Thought/Link under graph_lock; maintain Link adjacency."""
        with self.graph_lock:
            self._register_in_store_unlocked(thought)

    @property
    def thought_list(self) -> list[Thought]:
        """list of Thoughts (structural store values)."""
        return list(self.thoughts.values())

    def seed_self_description(self) -> dict[str, Thought]:
        """Install the six-Thought minimal configuration on the System side."""
        prefix = f"{self.id}:"
        seed = minimal_seed(with_labels=self.with_labels, id_prefix=prefix)
        preserved = {
            tid: t
            for tid, t in self.thoughts.items()
            if tid == f"{self.id}:agent" or tid.startswith(f"{self.id}:const:")
        }
        self.thoughts = {**seed, **preserved}
        self.system = seed[f"{prefix}system"]
        self.environment = seed[f"{prefix}environment"]
        if self.mirror_in_environment:
            self.env_thoughts = minimal_seed(
                with_labels=self.with_labels, id_prefix=f"{prefix}env:"
            )
        else:
            self.env_thoughts = {}
        self._out_index_dirty = True
        return seed

    def install_laws(self) -> list[Law]:
        """
        Install Asimov-shaped constitution as STABLE Thoughts/Links.
        Distinct from twin seed: Outerface constraints, not self-description.
        Merges into `thoughts` and sets `laws` ordered by priority.
        """
        if not hasattr(self, "environment") or self.environment is None:
            raise RuntimeError("install_laws requires environment pole (seed or bare poles first)")
        if not hasattr(self, "agent"):
            self.agent = Thought(
                id=f"{self.id}:agent",
                label="Agent" if self.with_labels else None,
            )
        nodes, laws = constitutional_seed(
            agent=self.agent,
            environment=self.environment,
            id_prefix=f"{self.id}:const:",
            with_labels=self.with_labels,
        )
        for tid, t in nodes.items():
            self.thoughts[tid] = t
        self._out_index_dirty = True
        self.laws = sorted(laws, key=lambda law: law.priority)
        return self.laws

    def law_by_code(self, code: str) -> Optional[Law]:
        for law in self.laws:
            if law.code == code:
                return law
        return None

    def check_action(self, **kwargs) -> tuple[bool, str]:
        """Delegate to Outerface constitutional gate."""
        return self.outerface.check_action(self, **kwargs)

    def act_from_graph(
        self,
        *,
        domain: Optional[str] = None,
        poles: Optional[list[Thought]] = None,
    ) -> list[tuple[bool, str]]:
        """
        Core agency: Outerface fires from Mind graph recommendation.

        Prefer this over demo-local control paths when the host should act from
        minted Action poles + valence. Fail-open empty list when cold.
        """
        return self.outerface.propose_actions_from_graph(domain=domain, poles=poles)

    def think_tick(
        self,
        *,
        domain: Optional[str] = None,
        poles: Optional[list[Thought]] = None,
        run_agency: bool = True,
    ) -> dict:
        """
        One host cognitive step: pulse dynamics (+ optional energy), then agency.

        Returns a small report for tests/HUD:
          pulse, actions (list of (ok, reason)), energy_remaining
        """
        budget = None
        if self.energy_enforced and self.energy is not None:
            # Cap this tick to remaining host energy (falsifiable)
            budget = max(0.0, float(self.energy.remaining))
        if budget is not None:
            pulse = self.pulse_partition(
                membership=None, engine_name="global", energy_budget=budget
            )
        else:
            pulse = self.pulse_tick()
        # Map pulse energy spend into host Energy pool when enforced
        if self.energy_enforced and self.energy is not None:
            used = float(pulse.get("energy_used", 0.0) or 0.0)
            if used > 0:
                # pulse_partition already "used" virtual budget; sync host pool
                self.energy.spend(used)
            elif budget is not None and budget <= 0:
                # Already empty — pulse may have capped; count refuse
                self.energy.refuse_count += 1
        actions: list[tuple[bool, str]] = []
        if run_agency:
            actions = self.act_from_graph(domain=domain, poles=poles)
        return {
            "pulse": pulse,
            "actions": actions,
            "energy_remaining": float(self.energy.remaining) if self.energy else 0.0,
            "energy_refused": int(self.energy.refuse_count) if self.energy else 0,
        }

    def nest_energy(self, capacity: float, *, label: Optional[str] = None) -> Energy:
        """Allocate a nested Energy sub-budget from this host (falsifiable)."""
        return self.energy.nest(capacity, label=label)

    def is_minimal(self) -> bool:
        return is_minimal_symbioid_shape(
            self.thoughts,
            system_id=self.system.id,
            environment_id=self.environment.id,
        )

    def twin_seed_thoughts(self) -> dict[str, Thought]:
        """
        Six-seed twin poles only (O(1) id lookup — not a full-graph scan).

        Avoids O(|thoughts| × prefixes) startswith thrash during protect/prune.
        """
        p = f"{self.id}:"
        seed_ids = (
            f"{p}system",
            f"{p}environment",
            f"{p}exists_in",
            f"{p}exists_around",
            f"{p}sys_exists_in_env",
            f"{p}env_exists_around_sys",
        )
        out: dict[str, Thought] = {}
        for tid in seed_ids:
            t = self.thoughts.get(tid)
            if t is not None:
                out[tid] = t
        return out

    def formation_thoughts(self, formation_id: Optional[str] = None) -> dict[str, Thought]:
        """
        Thoughts from sensor Input formations.

        When ``formation_id`` is given, prefer the Innerface completed store
        (includes content-addressed Observation poles outside the ``form:`` id
        prefix). Falls back to id-prefix scan. Without id, returns all
        ``{host}:form:`` prefix Thoughts (scaffolding only).
        """
        form_prefix = f"{self.id}:form:"
        if formation_id is not None:
            store = self.innerface.completed_formations.get(formation_id)
            if store:
                # Exclude stable sensor grounding pole (shared across formations)
                sensor_id = f"{self.id}:sensor:"
                return {
                    tid: t
                    for tid, t in store.items()
                    if not tid.startswith(sensor_id)
                }
            p = f"{formation_id}:"
            return {tid: t for tid, t in self.thoughts.items() if tid.startswith(p)}
        return {tid: t for tid, t in self.thoughts.items() if tid.startswith(form_prefix)}

    def sync_thoughts(self, sync_id: Optional[str] = None) -> dict[str, Thought]:
        """Thoughts from lateral Follows/FollowedBy sync sets (`{id}:sync:`)."""
        sync_prefix = f"{self.id}:sync:"
        if sync_id is not None:
            p = f"{sync_id}:"
            return {tid: t for tid, t in self.thoughts.items() if tid.startswith(p)}
        return {tid: t for tid, t in self.thoughts.items() if tid.startswith(sync_prefix)}

    def links(self) -> list[Link]:
        return [t for t in self.thoughts.values() if isinstance(t, Link)]

    def nodes(self) -> list[Thought]:
        return [t for t in self.thoughts.values() if not isinstance(t, Link)]

    def add(self, thought: Thought) -> None:
        """Alias for add_thought."""
        self.add_thought(thought)

    def get(self, thought_id: str) -> Optional[Thought]:
        with self.graph_lock:
            return self.thoughts.get(thought_id)

    def install_aspect_awareness(
        self,
        aspect_id: str,
        aspect_label: Optional[str],
        *,
        kind: str = "Sensor",
        full: bool = True,
    ) -> dict[str, Thought]:
        """
        Install awareness six-set: Agent **Has** aspect (e.g. "Symbioid has Ear").

        Registers `aspect_id` as an **integration terminator** so Innerface
        does not merge Observations across different Sensors/Actuators.

        full=False: only register the terminator + lightweight aspect pole
        (no awareness six-set). Use for high-cardinality maps (e.g. 200 cells).
        """
        if not hasattr(self, "agent") or self.agent is None:
            self.agent = Thought(
                id=f"{self.id}:agent",
                label="Agent" if self.with_labels else None,
            )
            with self.graph_lock:
                self.thoughts[self.agent.id] = self.agent

        # Human-readable aspect pole ("Ear", "Eye", "Hand")
        aspect_name = aspect_label or aspect_id
        if kind == "Sensor":
            # Reuse stable sensor Thought ids used by Interface formation
            class _AspectProxy:
                def __init__(self, sid: str, lab: Optional[str]) -> None:
                    self.id = sid
                    self.label = lab

            aspect_pole = ensure_sensor_thought(
                _AspectProxy(aspect_id, aspect_name),  # type: ignore[arg-type]
                host_id=self.id,
                with_labels=self.with_labels,
            )
            # Prefer capitalised sensor label as display name
            if self.with_labels and aspect_name:
                aspect_pole.label = aspect_name[0].upper() + aspect_name[1:]
        else:
            aspect_pole = Thought(
                id=f"{self.id}:actuator:{aspect_id}",
                label=(aspect_name[0].upper() + aspect_name[1:]) if aspect_name else aspect_id,
            )

        with self.graph_lock:
            self.thoughts[aspect_pole.id] = aspect_pole
            self.integration_terminators.add(aspect_id)

        if not full:
            return {"aspect": aspect_pole}

        awareness_id = f"{self.id}:aware:{kind.lower()}:{aspect_id}"
        store = complete_awareness_set(
            self.agent,
            aspect_pole,
            awareness_id=awareness_id,
            with_labels=self.with_labels,
            aspect_kind=kind,
        )
        with self.graph_lock:
            for tid, t in store.items():
                self.thoughts[tid] = t
            self.awareness_sets[aspect_id] = store
        emit_six_set("awareness", store)
        return store

    def add_sensor(
        self,
        sensor: Optional[Sensor] = None,
        *,
        label: Optional[str] = None,
        awareness: bool = True,
    ) -> Sensor:
        """
        Register a Sensor. awareness=False skips full six-set (terminator only)
        — preferred for dense maps (Tetris cells).
        """
        s = sensor or Sensor(label=label or f"sensor-{len(self.sensors)}")
        with self.graph_lock:
            self.sensors.append(s)
        self.install_aspect_awareness(
            s.id, s.label, kind="Sensor", full=bool(awareness)
        )
        return s

    def add_actuator(
        self,
        actuator: Optional[Actuator] = None,
        *,
        label: Optional[str] = None,
        awareness: bool = True,
    ) -> Actuator:
        a = actuator or Actuator(label=label or f"actuator-{len(self.actuators)}")
        with self.graph_lock:
            self.actuators.append(a)
        self.install_aspect_awareness(
            a.id, a.label, kind="Actuator", full=bool(awareness)
        )
        return a

    def __iter__(self) -> Iterator[Thought]:
        return iter(self.thoughts.values())

    def __len__(self) -> int:
        return len(self.thoughts)

    def as_dict(self) -> dict:
        return {
            "kind": "Symbioid",
            "id": self.id,
            "label": self.label,
            "is_system": True,
            "is_thought": False,
            "minimal": self.is_minimal(),
            "body": {"id": self.body.id, "label": self.body.label},
            "mind": {"id": self.mind.id, "label": self.mind.label, "enabled": self.mind.enabled},
            "sensors": [{"id": s.id, "label": s.label, "direction": s.direction} for s in self.sensors],
            "actuators": [
                {"id": a.id, "label": a.label, "direction": a.direction} for a in self.actuators
            ],
            "thoughts": {tid: t.as_dict() for tid, t in self.thoughts.items()},
            "innerface": {
                "id": self.innerface.id,
                "label": self.innerface.label,
                "enabled": self.innerface.enabled,
            },
            "interface": {
                "id": self.interface.id,
                "label": self.interface.label,
                "enabled": self.interface.enabled,
            },
            "outerface": {
                "id": self.outerface.id,
                "label": self.outerface.label,
                "enabled": self.outerface.enabled,
            },
            "system_pole_id": self.system.id,
            "environment_pole_id": self.environment.id,
            "agent_id": self.agent.id,
            "laws": [law.as_dict() for law in self.laws],
            "env_thoughts": {tid: t.as_dict() for tid, t in self.env_thoughts.items()},
        }

    def __repr__(self) -> str:
        lab = f" label={self.label!r}" if self.label else ""
        return (
            f"Symbioid(id={self.id!r}{lab} thoughts={len(self.thoughts)} "
            f"laws={len(self.laws)} sensors={len(self.sensors)} "
            f"actuators={len(self.actuators)} mind={self.mind.enabled} "
            f"minimal={self.is_minimal()})"
        )
