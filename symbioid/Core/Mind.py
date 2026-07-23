"""Mind aspect."""

from __future__ import annotations

from dataclasses import dataclass

from symbioid.Core.System import System


@dataclass
class Mind(System):
    """Processor-like substrate; not identical to Thoughts."""

    enabled: bool = True
