"""Body aspect."""

from __future__ import annotations

from dataclasses import dataclass

from symbioid.Core.System import System


@dataclass
class Body(System):
    """Outer boundary of the Symbioid; contains the other aspects."""

    note: str = ""
