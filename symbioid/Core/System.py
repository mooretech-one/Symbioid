"""Base System class."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from symbioid.Core.ids import _new_id


@dataclass
class System:
    """Base: anything that 'is a system'."""

    id: str = field(default_factory=_new_id)
    label: Optional[str] = None

    def __repr__(self) -> str:
        lab = f" label={self.label!r}" if self.label else ""
        return f"{type(self).__name__}(id={self.id!r}{lab})"
