"""Energy aspect — falsifiable nested budgets for Symbioid cognition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from symbioid.Core.System import System
from symbioid.Core.ids import _new_id


@dataclass
class Energy(System):
    """
    Nested energy pool for a Symbioid (or sub-aspect).

    Design contract (Antelligence / MassAI): energy is first-class pressure —
    budgets are **falsifiable** in tests (spend / nest / refuse when empty).

    Nesting: ``parent.nest(cap)`` reserves ``cap`` from the parent pool and
    returns a child Energy. Child spend does not auto-pull more from parent
    (hard sub-budget); parent remaining is reduced at nest time.
    """

    id: str = field(default_factory=lambda: _new_id("energy-"))
    label: Optional[str] = "Energy"
    capacity: float = 100.0
    remaining: float = 100.0
    parent: Optional["Energy"] = field(default=None, repr=False)
    # Metrics
    spent_total: float = field(default=0.0, init=False, repr=False)
    nest_count: int = field(default=0, init=False, repr=False)
    refuse_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.capacity = max(0.0, float(self.capacity))
        self.remaining = max(0.0, min(float(self.remaining), self.capacity))

    def reset(self, *, fill: bool = True) -> None:
        """Restore remaining to capacity (or empty)."""
        self.remaining = float(self.capacity) if fill else 0.0

    def spend(self, amount: float) -> bool:
        """
        Spend energy. Returns False if insufficient (falsifiable refuse).
        """
        amt = float(amount)
        if amt <= 0:
            return True
        if self.remaining + 1e-12 < amt:
            self.refuse_count += 1
            return False
        self.remaining = max(0.0, self.remaining - amt)
        self.spent_total += amt
        return True

    def can_spend(self, amount: float) -> bool:
        return float(amount) <= 0 or self.remaining + 1e-12 >= float(amount)

    def nest(self, capacity: float, *, label: Optional[str] = None) -> "Energy":
        """
        Allocate a nested sub-budget from this pool.

        Reserves ``capacity`` units from remaining (or all remaining if smaller).
        """
        cap = max(0.0, float(capacity))
        take = min(cap, self.remaining)
        if take <= 0 and cap > 0:
            self.refuse_count += 1
            # Empty child still records nest attempt
            child = Energy(
                capacity=0.0,
                remaining=0.0,
                parent=self,
                label=label or "Energy.nested",
            )
            self.nest_count += 1
            return child
        ok = self.spend(take)
        assert ok or take == 0
        child = Energy(
            capacity=take,
            remaining=take,
            parent=self,
            label=label or "Energy.nested",
        )
        self.nest_count += 1
        return child

    def fraction_remaining(self) -> float:
        if self.capacity <= 0:
            return 0.0
        return max(0.0, min(1.0, self.remaining / self.capacity))

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "capacity": self.capacity,
            "remaining": self.remaining,
            "spent_total": self.spent_total,
            "nest_count": self.nest_count,
            "refuse_count": self.refuse_count,
            "parent_id": self.parent.id if self.parent else None,
        }
