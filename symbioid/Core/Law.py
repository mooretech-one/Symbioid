"""Constitutional Law + Asimov-shaped seed."""

from __future__ import annotations

from dataclasses import dataclass

from symbioid.Core.Link import Link
from symbioid.Core.Thought import Thought


@dataclass
class Law:
    """
    Constitutional constraint: a Link plus fixed priority.
    Lower priority number = higher precedence (L0 wins over L3).
    """

    code: str  # L0 .. L3
    priority: int
    link: Link
    statement: str

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "priority": self.priority,
            "statement": self.statement,
            "link": self.link.as_dict(),
        }


def constitutional_seed(
    *,
    agent: Thought,
    environment: Thought,
    id_prefix: str = "",
    with_labels: bool = True,
) -> tuple[dict[str, Thought], list[Law]]:
    """
    Asimov-shaped constitution as STABLE Thought/Link patterns.

    Priority (high → low):
      L0  Preserve twin integrity (System–Environment relation can continue)
      L1  Must not harm protected Environment structure
      L2  Must obey authorized authority (unless L0/L1)
      L3  May preserve self (unless higher)

    "Human" is not hard-coded: Authority and ProtectedEnvironment are class
    Thoughts; humans can be instances or subclasses later.
    """
    p = id_prefix
    lab = (lambda s: s) if with_labels else (lambda s: None)

    protected = Thought(id=f"{p}protected_environment", label=lab("ProtectedEnvironment"))
    authority = Thought(id=f"{p}authority", label=lab("Authority"))
    self_node = Thought(id=f"{p}self", label=lab("Self"))

    lt_l0 = Thought(id=f"{p}lt_preserve_twin", label=lab("MustPreserveTwinIntegrity"))
    lt_l1 = Thought(id=f"{p}lt_must_not_harm", label=lab("MustNotHarm"))
    lt_l2 = Thought(id=f"{p}lt_must_obey", label=lab("MustObeyUnlessHigher"))
    lt_l3 = Thought(id=f"{p}lt_preserve_self", label=lab("MayPreserveSelfUnlessHigher"))

    link_l0 = Link(
        id=f"{p}law_l0",
        label=lab("L0_TwinIntegrity"),
        source=agent,
        link_type=lt_l0,
        target=environment,
    )
    link_l1 = Link(
        id=f"{p}law_l1",
        label=lab("L1_NonHarm"),
        source=agent,
        link_type=lt_l1,
        target=protected,
    )
    link_l2 = Link(
        id=f"{p}law_l2",
        label=lab("L2_ObeyAuthority"),
        source=agent,
        link_type=lt_l2,
        target=authority,
    )
    link_l3 = Link(
        id=f"{p}law_l3",
        label=lab("L3_SelfPreservation"),
        source=agent,
        link_type=lt_l3,
        target=self_node,
    )

    laws = [
        Law(
            code="L0",
            priority=0,
            link=link_l0,
            statement=(
                "Do not destroy the conditions under which System and Environment "
                "can continue as a twin (exists-in / exists-around remain possible)."
            ),
        ),
        Law(
            code="L1",
            priority=1,
            link=link_l1,
            statement=(
                "Do not harm protected Environment structure "
                "(nor, through inaction when able, allow such harm)."
            ),
        ),
        Law(
            code="L2",
            priority=2,
            link=link_l2,
            statement=(
                "Obey authorized Authority sources unless that conflicts with L0 or L1. "
                "Human is one possible Authority/Protected class, not the only one."
            ),
        ),
        Law(
            code="L3",
            priority=3,
            link=link_l3,
            statement=(
                "Protect this Symbioid's own existence unless that conflicts with L0–L2."
            ),
        ),
    ]

    nodes: dict[str, Thought] = {
        t.id: t
        for t in (
            agent,
            environment,
            protected,
            authority,
            self_node,
            lt_l0,
            lt_l1,
            lt_l2,
            lt_l3,
            link_l0,
            link_l1,
            link_l2,
            link_l3,
        )
    }
    return nodes, laws
