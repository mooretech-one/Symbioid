"""SpikingEngine — Process whose tick is primarily pulse, not face automata.

Phase 0 scaffold: base class + hooks. Faces migrate in Phases 1–3.
Phase 5: PortPacket export, per-engine energy budget on pulse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Set

from symbioid.Core.Process import Process


@dataclass(frozen=True)
class PortPacket:
    """Activation packet transferred between engines via Symbioid port queues."""

    thought_id: str
    activation: float
    source_engine: str
    cycle: int = 0
    channel: str = ""  # e.g. "interface>innerface"


@dataclass
class SpikingEngine(Process):
    """
    Base for Interface / Innerface / Outerface as spiking engines.

    Default ``_process_body``:
      pre_ports → pulse (masked partition) → post_ports

    Subclasses override pre/post for inject, consolidate, actuate.
    ``engine_name`` is used for membership / owner tags.
    """

    engine_name: str = "engine"
    # If True, pulse uses only membership Thought ids; empty membership = no pulse
    use_membership: bool = False
    membership: Set[str] = field(default_factory=set)
    # Phase 5: override Mind energy budget for this engine (None → Mind default)
    energy_budget: Optional[float] = None
    # Export activation from last pulse (port to next engine; compat + queue fill)
    last_export_ids: list[str] = field(default_factory=list, init=False, repr=False)
    last_pulse_stats: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def pre_ports(self) -> None:
        """Ingress: sample sensors / import port charge. Override in faces."""

    def post_ports(self) -> None:
        """Egress: export fire list / consolidate / actuate. Override in faces."""

    def membership_ids(self) -> Optional[Set[str]]:
        """
        Thought ids this engine may decay/fire/spread-from.
        None → full graph (legacy global pulse).
        """
        if not self.use_membership:
            return None
        return set(self.membership)

    def _resolve_energy_budget(self) -> Optional[float]:
        """Engine override → Mind per-engine → Mind default; 0/None = unlimited."""
        if self.energy_budget is not None:
            b = float(self.energy_budget)
            return b if b > 0 else None
        host = self.host
        if host is None:
            return None
        mind = getattr(host, "mind", None)
        if mind is None:
            return None
        per = {
            "interface": float(getattr(mind, "energy_budget_interface", 0.0) or 0.0),
            "innerface": float(getattr(mind, "energy_budget_innerface", 0.0) or 0.0),
            "outerface": float(getattr(mind, "energy_budget_outerface", 0.0) or 0.0),
        }.get(self.engine_name, 0.0)
        if per > 0:
            return per
        default = float(getattr(mind, "energy_budget_default", 0.0) or 0.0)
        return default if default > 0 else None

    def pulse(self) -> dict[str, Any]:
        """Run masked pulse_partition on host (with optional energy budget)."""
        host = self.host
        if host is None:
            return {
                "cycle": 0,
                "hot": 0,
                "fired": 0,
                "spread": 0,
                "hebb": 0,
                "energy_used": 0.0,
                "energy_left": 0.0,
            }
        mem = self.membership_ids()
        stats = host.pulse_partition(
            membership=mem,
            engine_name=self.engine_name,
            energy_budget=self._resolve_energy_budget(),
        )
        self.last_pulse_stats = dict(stats)
        # Export ids that just fired (for port transfer) — scan hot set / membership only
        fired_ids: list[str] = []
        with host.graph_lock:
            if mem is None:
                scan = list(getattr(host, "_hot_ids", ()) or ())
            else:
                scan = list(mem)
            for tid in scan:
                t = host.thoughts.get(tid)
                if t is not None and getattr(t, "just_fired", False):
                    fired_ids.append(tid)
        self.last_export_ids = fired_ids
        return stats

    def add_member(self, thought_id: str) -> None:
        self.membership.add(thought_id)

    def _process_body(self) -> None:
        """Engine tick: ports sandwich a pulse."""
        self.pre_ports()
        self.pulse()
        self.post_ports()
