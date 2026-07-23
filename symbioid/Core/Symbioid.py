"""Symbioid host — twin System ⋈ Environment with aspects and faces."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional

from symbioid.Core.Actuator import Actuator
from symbioid.Core.Body import Body
from symbioid.Core.Innerface import Innerface
from symbioid.Core.Interface import Interface
from symbioid.Core.Law import Law, constitutional_seed
from symbioid.Core.Link import Link
from symbioid.Core.Mind import Mind
from symbioid.Core.Outerface import Outerface
from symbioid.Core.Sensor import Sensor
from symbioid.Core.System import System
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import complete_awareness_set, emit_six_set, ensure_sensor_thought
from symbioid.Core.ids import _new_id
from symbioid.Core.seed import is_minimal_symbioid_shape, minimal_seed


@dataclass
class Symbioid(System):
    """
    Siamese twin System ⋈ Environment; also a System (not a Thought).

    Contained aspects (Antelligence):
      body, mind, sensors[], actuators[], thoughts{} (list via thought_list),
      innerface, interface, outerface

    Structural twin seed (six Thoughts) lives in `thoughts` when seed_minimal=True.
    Constitution (Asimov-shaped laws) installs when install_constitution=True
    (default): STABLE Law Links for Outerface gating — not the same as twin seed.
    """

    id: str = field(default_factory=lambda: _new_id("sym-"))
    seed_minimal: bool = True
    with_labels: bool = True
    mirror_in_environment: bool = False
    install_constitution: bool = True

    body: Body = field(default_factory=Body)
    mind: Mind = field(default_factory=Mind)
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

    def __post_init__(self) -> None:
        if self.body.label is None:
            self.body.label = "Body"
        if self.mind.label is None:
            self.mind.label = "Mind"
        self.innerface.host = self
        self.interface.host = self
        self.outerface.host = self
        if self.seed_minimal:
            self.seed_self_description()
        else:
            self.system = Thought(
                id=f"{self.id}:system",
                label="System" if self.with_labels else None,
            )
            self.environment = Thought(
                id=f"{self.id}:environment",
                label="Environment" if self.with_labels else None,
            )
            self.thoughts = {
                self.system.id: self.system,
                self.environment.id: self.environment,
            }
        self.agent = Thought(
            id=f"{self.id}:agent",
            label="Agent" if self.with_labels else None,
        )
        self.thoughts[self.agent.id] = self.agent
        if self.install_constitution:
            self.install_laws()

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

    def add_thought(self, thought: Thought) -> None:
        """Register a Thought/Link under graph_lock."""
        with self.graph_lock:
            self.thoughts[thought.id] = thought

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

    def is_minimal(self) -> bool:
        return is_minimal_symbioid_shape(
            self.thoughts,
            system_id=self.system.id,
            environment_id=self.environment.id,
        )

    def twin_seed_thoughts(self) -> dict[str, Thought]:
        """Thoughts belonging to the six-seed (by id prefix), excluding constitution/formations."""
        prefix = f"{self.id}:"
        skip_prefixes = (
            f"{self.id}:const:",
            f"{self.id}:form:",
            f"{self.id}:sync:",
            f"{self.id}:sensor:",
            f"{self.id}:aware:",
            f"{self.id}:belief:",
            f"{self.id}:int:",
            f"{self.id}:actuator:",
        )
        return {
            tid: t
            for tid, t in self.thoughts.items()
            if tid.startswith(prefix)
            and not any(tid.startswith(p) for p in skip_prefixes)
            and tid != f"{self.id}:agent"
        }

    def formation_thoughts(self, formation_id: Optional[str] = None) -> dict[str, Thought]:
        """Thoughts from sensor Input formations (ids under `{id}:form:`)."""
        form_prefix = f"{self.id}:form:"
        if formation_id is not None:
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
    ) -> dict[str, Thought]:
        """
        Install awareness six-set: Agent **Has** aspect (e.g. "Symbioid has Ear").

        Registers `aspect_id` as an **integration terminator** so Innerface
        does not merge Observations across different Sensors/Actuators.
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
            self.integration_terminators.add(aspect_id)
        emit_six_set("awareness", store)
        return store

    def add_sensor(self, sensor: Optional[Sensor] = None, *, label: Optional[str] = None) -> Sensor:
        s = sensor or Sensor(label=label or f"sensor-{len(self.sensors)}")
        with self.graph_lock:
            self.sensors.append(s)
        self.install_aspect_awareness(s.id, s.label, kind="Sensor")
        return s

    def add_actuator(self, actuator: Optional[Actuator] = None, *, label: Optional[str] = None) -> Actuator:
        a = actuator or Actuator(label=label or f"actuator-{len(self.actuators)}")
        with self.graph_lock:
            self.actuators.append(a)
        self.install_aspect_awareness(a.id, a.label, kind="Actuator")
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
