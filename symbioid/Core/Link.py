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

    def __post_init__(self) -> None:
        for name, comp in (
            ("source", self.source),
            ("link_type", self.link_type),
            ("target", self.target),
        ):
            if not isinstance(comp, Thought):
                raise TypeError(f"Link.{name} must be a Thought, got {type(comp)!r}")

    def components(self) -> tuple[Thought, Thought, Thought]:
        return self.source, self.link_type, self.target

    def as_dict(self) -> dict:
        d = super().as_dict()
        d.update(
            {
                "source_id": self.source.id,
                "link_type_id": self.link_type.id,
                "target_id": self.target.id,
            }
        )
        return d

    def __repr__(self) -> str:
        lab = f" label={self.label!r}" if self.label else ""
        return (
            f"Link(id={self.id!r}{lab} "
            f"{self.source.id!r} -[{self.link_type.id!r}]-> {self.target.id!r})"
        )
