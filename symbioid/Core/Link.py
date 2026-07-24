"""Link = Thought with Source, LinkType, Target."""

from __future__ import annotations

from dataclasses import dataclass

from symbioid.Core.Thought import Thought


@dataclass(kw_only=True)
class Link(Thought):
    """Link is a Thought with Source, LinkType, Target — all Thoughts."""

    source: Thought
    link_type: Thought
    target: Thought
    # Pulse gain when source fires → target.receive(activation * weight * gain)
    # Plastic: raised by Hebbian co-fire / outcome (see Symbioid.pulse_tick)
    weight: float = 1.0
    # Phase 5: Port channel Links — no pulse spread; cross-engine Hebb + transfer gain
    is_port: bool = False

    def __post_init__(self) -> None:
        for name, comp in (
            ("source", self.source),
            ("link_type", self.link_type),
            ("target", self.target),
        ):
            if not isinstance(comp, Thought):
                raise TypeError(f"Link.{name} must be a Thought, got {type(comp)!r}")

    def adjust_weight(
        self,
        delta: float,
        *,
        w_min: float = 0.05,
        w_max: float = 4.0,
    ) -> float:
        """Clamp-update synaptic weight; returns new weight."""
        self.weight = max(float(w_min), min(float(w_max), float(self.weight) + float(delta)))
        return self.weight

    def components(self) -> tuple[Thought, Thought, Thought]:
        return self.source, self.link_type, self.target

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update(
            {
                "source_id": self.source.id,
                "link_type_id": self.link_type.id,
                "target_id": self.target.id,
                "weight": self.weight,
                "is_port": bool(self.is_port),
            }
        )
        return d

    def __repr__(self) -> str:
        lab = f" label={self.label!r}" if self.label else ""
        return (
            f"Link(id={self.id!r}{lab} "
            f"{self.source.id!r} -[{self.link_type.id!r}]-> {self.target.id!r})"
        )
