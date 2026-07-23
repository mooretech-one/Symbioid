"""Interface process — sample Sensors, start Rodin 1→2, hand off to Innerface."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from symbioid.Core.Process import Process
from symbioid.Core.Sensor import Sensor
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import begin_sensor_formation, ensure_sensor_thought
from symbioid.Core.ids import _new_id


@dataclass
class Interface(Process):
    """
    Interface process (~6): Sensors/Actuators + I/O events.

    Each tick, every Sensor samples a new Input. Interface starts Rodin 1→2:
    Sensor (Source) + Observation (Target / Input value), then hands a batch
    to Innerface for Perceives/PerceivedBy completion and Follows sync.
    Quiet in the process loop; six-set lines emit only on Innerface completion.
    """

    id: str = field(default_factory=lambda: _new_id("iface-"))
    label: Optional[str] = "Interface"
    continuous_inputs: bool = True
    io_events: int = field(default=0, init=False, repr=False)
    _formation_gen: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _sensors_started: set[str] = field(default_factory=set, init=False, repr=False)
    _sensor_thoughts: dict[str, Thought] = field(default_factory=dict, init=False, repr=False)
    handoffs_sent: int = field(default=0, init=False, repr=False)
    inputs_sampled: int = field(default=0, init=False, repr=False)

    def process(self) -> Optional[threading.Thread]:
        thread = super().process()
        return thread

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

    def start_formation_for_sensor(
        self,
        sensor: Sensor,
        *,
        force: bool = False,
        sense: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Rodin 1→2: Sensor (Source) + Observation (Target) for one Input.
        Returns handoff or None if already started and not forced.
        """
        host = self.host
        if host is None:
            return None
        with self._local_lock:
            if sensor.id in self._sensors_started and not force:
                return None
            gen = self._formation_gen.get(sensor.id, -1) + 1
            self._formation_gen[sensor.id] = gen
            self._sensors_started.add(sensor.id)
        sensor_thought = self._sensor_source(sensor)
        handoff = begin_sensor_formation(
            sensor,
            host_id=host.id,
            generation=gen,
            with_labels=host.with_labels,
            sense=sense,
            sensor_thought=sensor_thought,
        )
        with self._local_lock:
            self.handoffs_sent += 1
            self.io_events += 1
        return handoff

    def _process_body(self) -> None:
        """I/O tick: sample each sensor → batch handoff to Innerface."""
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
            # Actuator world state for closed-loop Sensor.transfer readouts
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
                        continue  # max_samples reached — cease Inputs for this sensor
                    sense = sensor.sample(tick=tick, world=world)
                    if sense is None:
                        continue
                    with self._local_lock:
                        self.inputs_sampled += 1
                h = self.start_formation_for_sensor(sensor, force=True, sense=sense)
            else:
                if not sensor.can_sample() and sense is None:
                    continue
                h = self.start_formation_for_sensor(sensor, force=False, sense=sense)
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
