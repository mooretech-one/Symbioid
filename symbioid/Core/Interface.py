"""Interface process — ingress spiking engine (Phase 1) + legacy automata."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from symbioid.Core.Sensor import Sensor
from symbioid.Core.SpikingEngine import SpikingEngine
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import begin_sensor_formation, ensure_sensor_thought
from symbioid.Core.ids import _new_id


@dataclass
class Interface(SpikingEngine):
    """
    Interface process (~6): Sensors/Actuators + I/O events.

    **legacy** (default): sample → admit → handoff every formation; full-graph
    ``pulse_tick`` at end of body.

    **hybrid / spike** (Phase 1 engine): sample → stimulate → membership pulse;
    handoff to Innerface only on **mint** (first structure lock-in), not every
    reuse/skip. Dynamics are the main work; automata are sparse.
    """

    id: str = field(default_factory=lambda: _new_id("iface-"))
    label: Optional[str] = "Interface"
    engine_name: str = "interface"
    continuous_inputs: bool = True
    io_events: int = field(default=0, init=False, repr=False)
    _formation_gen: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _sensors_started: set[str] = field(default_factory=set, init=False, repr=False)
    _sensor_thoughts: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    handoffs_sent: int = field(default=0, init=False, repr=False)
    inputs_sampled: int = field(default=0, init=False, repr=False)
    handoffs_skipped_mind: int = field(default=0, init=False, repr=False)
    # Phase 1: reuse/skip did not post to Innerface
    handoffs_suppressed_engine: int = field(default=0, init=False, repr=False)
    # Pending handoffs built in pre_ports, posted in post_ports (hybrid)
    _pending_handoffs: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    # Phase 4 spike: structure for Inner consolidator (no inbox)
    structure_pending: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    _pending_tick: int = field(default=0, init=False, repr=False)
    _pending_forward_msgs: list[Any] = field(default_factory=list, init=False, repr=False)

    def process(self) -> Optional[threading.Thread]:
        return super().process()

    def _engine_mode(self) -> str:
        host = self.host
        if host is None:
            return "legacy"
        mode = getattr(host, "engines_mode", "legacy") or "legacy"
        return mode if mode in ("legacy", "hybrid", "spike") else "legacy"

    def _sensor_source(self, sensor: Sensor) -> Thought:
        """Stable Source Thought for this Sensor on the host."""
        host = self.host
        host_id = host.id if host is not None else "host"
        with_labels = host.with_labels if host is not None else True
        with self._local_lock:
            existing = self._sensor_thoughts.get(sensor.id)
            if existing is not None:
                return existing
            thought = ensure_sensor_thought(
                sensor, host_id=host_id, with_labels=with_labels
            )
            self._sensor_thoughts[sensor.id] = thought
            return thought

    def _register_members(self, *thoughts: Optional[Thought]) -> None:
        for t in thoughts:
            if t is None:
                continue
            self.add_member(t.id)
            if t.engine_owner is None:
                t.engine_owner = self.engine_name

    def start_formation_for_sensor(
        self,
        sensor: Sensor,
        *,
        force: bool = False,
        sense: Optional[dict[str, Any]] = None,
        post_to_innerface: Optional[bool] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Rodin 1→2: Sensor (Source) + Observation (Target) for one Input.

        Consults host.mind.admit_input when recognition is enabled.
        Returns handoff or None if already started and not forced, or Mind skips.

        post_to_innerface: if False, queue for engine ``post_ports`` instead of
        returning only for the caller. Default True (external callers / demos
        post or receive the handoff themselves).
        """
        host = self.host
        if host is None:
            return None
        mode = self._engine_mode()
        if post_to_innerface is None:
            post_to_innerface = True

        with self._local_lock:
            if sensor.id in self._sensors_started and not force:
                return None
            gen = self._formation_gen.get(sensor.id, -1) + 1
            self._formation_gen[sensor.id] = gen
            self._sensors_started.add(sensor.id)

        sensor_thought = self._sensor_source(sensor)
        mind = getattr(host, "mind", None)
        admit = None
        if mind is not None:
            admit = mind.admit_input(
                sensor.id,
                sense,
                host_id=host.id,
                with_labels=host.with_labels,
            )
            if admit.action == "skip":
                # Still stimulate known pole if we have a prior observation
                if mind.dynamics_enabled and admit.observation:
                    host.stimulate(
                        admit.observation, float(mind.observation_stimulus) * 0.5
                    )
                    host.stimulate(sensor_thought, float(mind.sensor_stimulus) * 0.5)
                    self._register_members(sensor_thought, admit.observation)
                with self._local_lock:
                    self.handoffs_skipped_mind += 1
                    self.io_events += 1
                return None

        reused = bool(admit is not None and admit.action == "reuse")
        mind_action = admit.action if admit is not None else "mint"
        handoff = begin_sensor_formation(
            sensor,
            host_id=host.id,
            generation=gen,
            with_labels=host.with_labels,
            sense=sense,
            sensor_thought=sensor_thought,
            observation=admit.observation if admit is not None else None,
            formation_id=admit.formation_id if admit is not None else None,
            content_key=admit.content_key if admit is not None else None,
            reused=reused,
            mind_action=mind_action,
        )
        # Ensure host graph has sensor + observation poles early
        with host.graph_lock:
            host.thoughts[sensor_thought.id] = sensor_thought
            obs = handoff["partial"]["observation"]
            if isinstance(obs, Thought):
                host.thoughts[obs.id] = obs
        # Ingress Signal: raise activation on poles (Thought-as-neuron)
        if mind is not None and getattr(mind, "dynamics_enabled", True):
            host.stimulate(sensor_thought, float(mind.sensor_stimulus))
            obs = handoff["partial"].get("observation")
            if isinstance(obs, Thought):
                host.stimulate(obs, float(mind.observation_stimulus))
        self._register_members(
            sensor_thought,
            handoff["partial"].get("observation")
            if isinstance(handoff["partial"].get("observation"), Thought)
            else None,
        )

        # hybrid: handoff only on mint; spike: no inbox (structure_pending if engine path)
        should_handoff = True
        if mode == "spike":
            should_handoff = False
            with self._local_lock:
                self.handoffs_suppressed_engine += 1
                # Engine-owned structure only when not returning handoff to external
                if mind_action == "mint" and not post_to_innerface:
                    self.structure_pending.append(handoff)
        elif mode == "hybrid" and mind_action != "mint":
            should_handoff = False
            with self._local_lock:
                self.handoffs_suppressed_engine += 1

        with self._local_lock:
            if should_handoff:
                self.handoffs_sent += 1
            self.io_events += 1

        if not should_handoff:
            # spike mint + external post_to_innerface: return handoff for demo/tests
            if mode == "spike" and mind_action == "mint" and post_to_innerface:
                return handoff
            # hybrid: non-mint suppressed
            return None

        if post_to_innerface:
            return handoff
        # hybrid engine path: queue mint for post_ports
        with self._local_lock:
            self._pending_handoffs.append(handoff)
        return handoff

    def pre_ports(self) -> None:
        """Sample sensors and inject charge (engine path)."""
        host = self.host
        if host is None:
            return
        tick = self.cycles + 1
        self._pending_tick = tick
        self._pending_handoffs.clear()
        self._pending_forward_msgs.clear()
        messages = self._drain_inbox()

        force_ids: set[str] = set()
        inbox_sense: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if not isinstance(msg, dict) or not msg.get("sensor_id"):
                continue
            sid = str(msg["sensor_id"])
            if msg.get("force") or msg.get("sense") is not None or msg.get("kind") == "input":
                force_ids.add(sid)
                if msg.get("kind") == "input" or msg.get("sense") is not None:
                    inbox_sense[sid] = (
                        msg if msg.get("kind") == "input" else dict(msg.get("sense") or msg)
                    )

        with host.graph_lock:
            sensors_snapshot = list(host.sensors)
            world = {
                (a.label or a.id): float(getattr(a, "output", 0.0))
                for a in host.actuators
            }
            self.io_events += 1 + len(messages)

        for sensor in sensors_snapshot:
            sense: Optional[dict[str, Any]] = inbox_sense.get(sensor.id)
            if self.continuous_inputs or sensor.id in force_ids:
                if sense is None:
                    if not sensor.can_sample():
                        continue
                    sense = sensor.sample(tick=tick, world=world)
                    if sense is None:
                        continue
                    with self._local_lock:
                        self.inputs_sampled += 1
                self.start_formation_for_sensor(
                    sensor, force=True, sense=sense, post_to_innerface=False
                )
            else:
                if not sensor.can_sample() and sense is None:
                    continue
                self.start_formation_for_sensor(
                    sensor, force=False, sense=sense, post_to_innerface=False
                )

        for msg in messages:
            if isinstance(msg, dict) and msg.get("kind") in (
                "formation_handoff",
                "formation_batch",
            ):
                continue
            if isinstance(msg, dict) and msg.get("sensor_id") and (
                msg.get("force")
                or msg.get("sense") is not None
                or msg.get("kind") == "input"
            ):
                continue
            self._pending_forward_msgs.append({"from": "interface", "payload": msg})

    def pulse(self) -> dict[str, int]:
        """Masked interface pulse (hybrid/spike) or full-graph (legacy)."""
        host = self.host
        if host is None:
            return {"cycle": 0, "hot": 0, "fired": 0, "spread": 0, "hebb": 0}
        if not getattr(host.mind, "dynamics_enabled", True):
            return {"cycle": host.pulse_cycle, "hot": 0, "fired": 0, "spread": 0, "hebb": 0}

        mode = self._engine_mode()
        if mode == "legacy":
            stats = host.pulse_tick()
            self.last_pulse_stats = dict(stats)
            self.last_export_ids = []
            return stats

        # hybrid/spike: membership-only pulse
        self.use_membership = True
        if not self.membership:
            # Still advance nothing meaningful
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

    def post_ports(self) -> None:
        """Post sparse mint handoffs (hybrid only) + export fired Observations."""
        host = self.host
        if host is None:
            return
        mode = self._engine_mode()
        handoffs = list(self._pending_handoffs)
        self._pending_handoffs.clear()
        tick = self._pending_tick

        # hybrid: mint handoffs via inbox; spike: structure_pending only (no inbox)
        if mode == "hybrid" and host.innerface is not None and handoffs:
            if len(handoffs) > 1:
                host.innerface.post(
                    {
                        "kind": "formation_batch",
                        "handoffs": handoffs,
                        "tick": tick,
                    }
                )
            else:
                host.innerface.post(handoffs[0])

        if mode != "spike" and host.innerface is not None:
            for msg in self._pending_forward_msgs:
                host.innerface.post(msg)
        self._pending_forward_msgs.clear()

        # Snapshot export activations + Phase 5 port queue → Innerface
        for tid in self.last_export_ids:
            t = host.thoughts.get(tid)
            if t is not None:
                t.export_activation = float(t.activation)
        if mode in ("hybrid", "spike") and self.last_export_ids:
            host.export_port_packets(
                src_engine=self.engine_name,
                dst_engine="innerface",
                thought_ids=list(self.last_export_ids),
                cycle=host.pulse_cycle,
            )

    def _process_body_legacy(self) -> None:
        """Original automata path: handoff every formation + full pulse_tick."""
        tick = self.cycles + 1
        messages = self._drain_inbox()
        host = self.host
        if host is None:
            return

        force_ids: set[str] = set()
        inbox_sense: dict[str, dict[str, Any]] = {}
        for msg in messages:
            if not isinstance(msg, dict) or not msg.get("sensor_id"):
                continue
            sid = str(msg["sensor_id"])
            if msg.get("force") or msg.get("sense") is not None or msg.get("kind") == "input":
                force_ids.add(sid)
                if msg.get("kind") == "input" or msg.get("sense") is not None:
                    inbox_sense[sid] = (
                        msg if msg.get("kind") == "input" else dict(msg.get("sense") or msg)
                    )

        with host.graph_lock:
            sensors_snapshot = list(host.sensors)
            world = {
                (a.label or a.id): float(getattr(a, "output", 0.0))
                for a in host.actuators
            }
            self.io_events += 1 + len(messages)

        handoffs: list[dict[str, Any]] = []
        for sensor in sensors_snapshot:
            sense: Optional[dict[str, Any]] = inbox_sense.get(sensor.id)
            if self.continuous_inputs or sensor.id in force_ids:
                if sense is None:
                    if not sensor.can_sample():
                        continue
                    sense = sensor.sample(tick=tick, world=world)
                    if sense is None:
                        continue
                    with self._local_lock:
                        self.inputs_sampled += 1
                h = self.start_formation_for_sensor(
                    sensor, force=True, sense=sense, post_to_innerface=True
                )
            else:
                if not sensor.can_sample() and sense is None:
                    continue
                h = self.start_formation_for_sensor(
                    sensor, force=False, sense=sense, post_to_innerface=True
                )
            if h is not None:
                handoffs.append(h)

        if host.innerface is not None and handoffs:
            if len(handoffs) > 1:
                host.innerface.post(
                    {
                        "kind": "formation_batch",
                        "handoffs": handoffs,
                        "tick": tick,
                    }
                )
            else:
                host.innerface.post(handoffs[0])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("kind") in (
                    "formation_handoff",
                    "formation_batch",
                ):
                    continue
                if isinstance(msg, dict) and msg.get("sensor_id") and (
                    msg.get("force")
                    or msg.get("sense") is not None
                    or msg.get("kind") == "input"
                ):
                    continue
                host.innerface.post({"from": "interface", "payload": msg})

        # When True, demo main loop owns pulse_tick (avoid double full-graph work).
        if getattr(self, "skip_global_pulse", False):
            return
        if getattr(host.mind, "dynamics_enabled", True):
            host.pulse_tick()

    def _process_body(self) -> None:
        """legacy automata vs hybrid/spike engine tick."""
        if self._engine_mode() == "legacy":
            self._process_body_legacy()
            return
        # Engine: pre_ports (sample+stimulate) → membership pulse → post (sparse handoff)
        self.use_membership = True
        self.pre_ports()
        if not getattr(self, "skip_global_pulse", False):
            self.pulse()
        self.post_ports()
