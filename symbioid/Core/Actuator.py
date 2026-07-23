"""Actuator aspect — output only when Outerface gates allow."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from symbioid.Core.System import System

if TYPE_CHECKING:
    from symbioid.Core.Symbioid import Symbioid


@dataclass
class Actuator(System):
    """
    Act inward or outward; list on Symbioid.

    H4: firing must go through Outerface.check_action — no ungated world effect.
    """

    direction: str = "out"  # out | both
    # World output seen by Sensors (phase / command value). Advanced on each fire.
    output: float = 0.0
    output_step: float = 0.2
    fire_count: int = field(default=0, init=False, repr=False)
    deny_count: int = field(default=0, init=False, repr=False)
    last_action: Optional[str] = field(default=None, init=False, repr=False)
    last_gate: Optional[str] = field(default=None, init=False, repr=False)
    last_allowed: Optional[bool] = field(default=None, init=False, repr=False)

    def request_fire(
        self,
        host: "Symbioid",
        action: str,
        *,
        belief_id: Optional[str] = None,
        **gate_flags: Any,
    ) -> tuple[bool, str]:
        """
        Propose an action; Outerface constitutional gate decides.
        Returns (allowed, reason). Increments fire_count only when allowed.
        """
        if host is None or host.outerface is None:
            self.deny_count += 1
            self.last_allowed = False
            self.last_gate = "no_outerface"
            return False, "no_outerface"
        if not host.outerface.enabled:
            self.deny_count += 1
            self.last_allowed = False
            self.last_gate = "outerface_disabled"
            return False, "outerface_disabled"

        allowed, reason = host.outerface.check_action(host, **gate_flags)
        self.last_gate = reason
        self.last_action = action
        if not allowed:
            self.deny_count += 1
            self.last_allowed = False
            return False, reason

        self.fire_count += 1
        self.last_allowed = True
        # Advance world output so Sensors can read f(hand) as Feedback
        self.output = float(self.output) + float(self.output_step)
        # Optional hook: record on Outerface for telemetry
        if hasattr(host.outerface, "record_actuator_fire"):
            host.outerface.record_actuator_fire(
                actuator_id=self.id,
                action=action,
                reason=reason,
                belief_id=belief_id,
            )
        return True, reason
