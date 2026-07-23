"""Atomic Thought (Simon)."""

from __future__ import annotations

from dataclasses import dataclass

from symbioid.Core.System import System


@dataclass
class Thought(System):
    """Atomic Thought (Simon); also a System."""

    transient: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": type(self).__name__,
            "id": self.id,
            "label": self.label,
            "transient": self.transient,
        }
