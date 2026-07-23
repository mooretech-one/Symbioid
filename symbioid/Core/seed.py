"""Minimal twin-seed helpers (six-Thought self-description)."""

from __future__ import annotations

from typing import Optional

from symbioid.Core.Link import Link
from symbioid.Core.Thought import Thought


def minimal_seed(
    *,
    with_labels: bool = True,
    id_prefix: str = "",
) -> dict[str, Thought]:
    """Six Thoughts of the minimal Symbioid self-description."""
    p = id_prefix
    lab = (lambda s: s) if with_labels else (lambda s: None)

    system = Thought(id=f"{p}system", label=lab("System"))
    environment = Thought(id=f"{p}environment", label=lab("Environment"))
    exists_in = Thought(id=f"{p}exists_in", label=lab("ExistsIn"))
    exists_around = Thought(id=f"{p}exists_around", label=lab("ExistsAround"))
    link_in = Link(
        id=f"{p}sys_exists_in_env",
        label=lab("SystemExistsInEnvironment"),
        source=system,
        link_type=exists_in,
        target=environment,
    )
    link_around = Link(
        id=f"{p}env_exists_around_sys",
        label=lab("EnvironmentExistsAroundSystem"),
        source=environment,
        link_type=exists_around,
        target=system,
    )
    return {
        t.id: t
        for t in (system, environment, exists_in, exists_around, link_in, link_around)
    }


def is_minimal_symbioid_shape(
    store: dict[str, Thought],
    *,
    system_id: Optional[str] = None,
    environment_id: Optional[str] = None,
) -> bool:
    """
    Label-agnostic: a reciprocal Link pair with distinct LinkTypes exists.
    Extra Thoughts (e.g. constitution) are allowed.
    Optional system_id/environment_id restrict which poles count as the twin.
    """
    links = [t for t in store.values() if isinstance(t, Link)]
    for i, a in enumerate(links):
        for b in links[i + 1 :]:
            if not (
                a.source.id == b.target.id
                and a.target.id == b.source.id
                and a.link_type.id != b.link_type.id
                and a.source.id != a.target.id
            ):
                continue
            if a.link_type.id not in store or b.link_type.id not in store:
                continue
            if a.source.id not in store or a.target.id not in store:
                continue
            if system_id is not None and environment_id is not None:
                if {a.source.id, a.target.id} != {system_id, environment_id}:
                    continue
            return True
    return False
