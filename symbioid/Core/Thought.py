"""Atomic Thought (Simon) — also carries activation dynamics (Thought-as-neuron)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from symbioid.Core.System import System
from symbioid.Core.thought_layers import (
    SIMON_ATOMIC_THOUGHT,
    ThoughtLayer,
    normalize_layer,
)


@dataclass
class Thought(System):
    """
    Atomic Thought (Simon MVP); also a System.

    Dual contract (Simon):
      - **Structure** — id / label / links (long-term graph identity)
      - **Signal** — activation / threshold / decay (short-term energy)

    Layer (Structure / Pattern / Feeling) reduces which cognitive surface this
    Thought primarily serves — see ``thought_layers`` and SevenSphere mapping.
    Mind is **not** a Thought (processor aspect lives on Symbioid.mind).
    """

    transient: bool = False
    # Structure | Pattern | Feeling (Thought MVP layers)
    layer: ThoughtLayer = ThoughtLayer.PATTERN
    # --- Thought-as-neuron dynamics ---
    activation: float = 0.0
    resting: float = 0.0
    threshold: float = 1.0
    activation_max: float = 3.0
    decay_rate: float = 0.15  # fraction of (activation - resting) lost per tick
    refractory_ticks: int = 0
    default_refractory: int = 2
    last_fired_cycle: int = -1
    # Host pulse_cycle when last hot / stimulated (cold-forget age; -1 = never)
    last_hot_cycle: int = -1
    # Structural seeds may set False or very high threshold
    dynamics_enabled: bool = True
    # Multi-engine: which SpikingEngine may try_fire this Thought (None = any)
    engine_owner: Optional[str] = None
    # Port export (activation snapshot for next engine)
    export_activation: float = 0.0
    # Set True for one tick after successful try_fire (consumers / HUD)
    just_fired: bool = field(default=False, init=False, repr=False)
    # Spectral substrate (Phase 1+): phase for optional phase-locked Hebb (radians)
    spectral_phase: float = 0.0

    def __post_init__(self) -> None:
        self.layer = normalize_layer(self.layer)

    def set_layer(self, layer: Union[ThoughtLayer, str]) -> "Thought":
        """Assign cognitive layer; returns self for chaining."""
        self.layer = normalize_layer(layer)
        return self

    def is_structure(self) -> bool:
        return self.layer is ThoughtLayer.STRUCTURE

    def is_pattern(self) -> bool:
        return self.layer is ThoughtLayer.PATTERN

    def is_feeling(self) -> bool:
        return self.layer is ThoughtLayer.FEELING

    @staticmethod
    def simon_contract() -> dict:
        """Documented Simon Atomic Thought dual (structure + signal)."""
        return dict(SIMON_ATOMIC_THOUGHT)

    def receive(self, amount: float) -> None:
        """Add stimulus; clamp. Does not fire (pulse_tick decides)."""
        if not self.dynamics_enabled or amount == 0:
            return
        a = float(self.activation) + float(amount)
        self.activation = max(0.0, min(float(self.activation_max), a))

    def try_fire(self, *, cycle: int) -> bool:
        """
        Fire if not refractory and activation >= threshold.
        Starts refractory; sets just_fired. Does not spread (host does).
        """
        self.just_fired = False
        if not self.dynamics_enabled:
            return False
        if self.refractory_ticks > 0:
            return False
        if float(self.activation) < float(self.threshold):
            return False
        self.just_fired = True
        self.last_fired_cycle = int(cycle)
        self.refractory_ticks = max(0, int(self.default_refractory))
        return True

    def decay_step(self) -> None:
        """Leak activation toward resting; decrement refractory; clear just_fired."""
        self.just_fired = False
        if not self.dynamics_enabled:
            return
        r = float(self.resting)
        d = max(0.0, min(1.0, float(self.decay_rate)))
        self.activation = r + (1.0 - d) * (float(self.activation) - r)
        if abs(self.activation - r) < 1e-9:
            self.activation = r
        if self.refractory_ticks > 0:
            self.refractory_ticks -= 1

    def is_hot(self, eps: float = 1e-6) -> bool:
        """True if dynamics tick should process this Thought."""
        if not self.dynamics_enabled:
            return False
        return (
            abs(float(self.activation) - float(self.resting)) > eps
            or self.refractory_ticks > 0
            or self.just_fired
        )

    def as_dict(self) -> dict:
        return {
            "kind": type(self).__name__,
            "id": self.id,
            "label": self.label,
            "transient": self.transient,
            "layer": self.layer.value if isinstance(self.layer, ThoughtLayer) else str(self.layer),
            "activation": self.activation,
            "threshold": self.threshold,
            "refractory_ticks": self.refractory_ticks,
            "dynamics_enabled": self.dynamics_enabled,
            "spectral_phase": float(self.spectral_phase),
        }
