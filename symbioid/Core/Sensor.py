"""Sensor aspect — Input sampling (random or actuator-coupled feedback)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from symbioid.Core.System import System

# world map: actuator_label → current output value
TransferFn = Callable[[dict[str, float]], float]


@dataclass
class Sensor(System):
    """
    Observe environment and/or interior.

    Default: random reading in [value_min, value_max].
    For feedback tests: set `transfer` to a function of actuator outputs, e.g.
      ear.transfer = lambda w: math.sin(w["hand"])
      eye.transfer = lambda w: math.cos(w["hand"])
    """

    direction: str = "in"  # in | both
    value_min: float = 0.0
    value_max: float = 1.0
    max_samples: Optional[int] = None
    # Optional closed-loop readout from actuator world state
    transfer: Optional[TransferFn] = field(default=None, repr=False)
    sample_count: int = field(default=0, init=False, repr=False)
    last_value: Optional[float] = field(default=None, init=False, repr=False)

    def can_sample(self) -> bool:
        """False once max_samples is reached (if set)."""
        if self.max_samples is None:
            return True
        return self.sample_count < self.max_samples

    def sample(
        self,
        *,
        tick: int = 0,
        world: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Emit a fresh Input, or None if max_samples reached.

        If `transfer` is set and `world` is provided, reading = transfer(world)
        (e.g. sin/cos of hand output). Otherwise uniform random.
        """
        if not self.can_sample():
            return None
        self.sample_count += 1
        lab = self.label or self.id
        if self.transfer is not None and world is not None:
            reading = float(self.transfer(world))
        else:
            reading = random.uniform(self.value_min, self.value_max)
        self.last_value = reading
        return {
            "kind": "input",
            "sensor_id": self.id,
            "sensor_label": lab,
            "tick": tick,
            "sample": self.sample_count,
            "value": f"{lab}:{reading:.4f}",
            "reading": reading,
        }
