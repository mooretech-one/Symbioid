"""Rodin cycle + sensor Input formation + six-set display helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from symbioid.Core.Link import Link
from symbioid.Core.Thought import Thought
from symbioid.Core.seed import is_minimal_symbioid_shape

if TYPE_CHECKING:
    from symbioid.Core.Sensor import Sensor

# Cycle: 1 → 2 → 4 → 8 → 7 → 5 → 1 …
#
# Per-sensor Input formation (Interface starts, Innerface completes):
#   1 Sensor (Source) | 2 Observation (Target)
#   4 Perceives | 8 Link(Sensor -Perceives→ Observation)
#   7 PerceivedBy | 5 Link(Observation -PerceivedBy→ Sensor)
#
# Lateral sync: ObservationA ⇄ ObservationB via Follows / FollowedBy

RODIN_CYCLE: tuple[int, ...] = (1, 2, 4, 8, 7, 5)
INTERFACE_RODIN_STAGES: tuple[int, ...] = (1, 2)
INNERFACE_RODIN_STAGES: tuple[int, ...] = (4, 8, 7, 5)

FORMATION_ROLES: tuple[str, ...] = (
    "sensor",
    "observation",
    "perceives",
    "link_perceives",
    "perceived_by",
    "link_perceived_by",
)

FOLLOWS_ROLES: tuple[str, ...] = (
    "observation_a",
    "observation_b",
    "follows",
    "link_follows",
    "followed_by",
    "link_followed_by",
)


def digital_root(n: int) -> int:
    """Single-digit digital root (1–9 for n≥1; 0 for 0)."""
    n = abs(int(n))
    if n == 0:
        return 0
    return 1 + (n - 1) % 9


def rodin_double(n: int) -> int:
    """Double then reduce to digital root (Rodin vortex step)."""
    return digital_root(2 * n)


def rodin_halve(n: int) -> int:
    """
    Inverse Rodin step (halving on the 1-2-4-8-7-5 vortex).

    Doubling is ×2 under digital root; halving is ×5 (mod-9 inverse of 2):
    1→5→7→8→4→2→1.
    """
    cur = digital_root(n) or 1
    return digital_root(5 * cur)


def rodin_sequence(start: int = 1, steps: int = 6) -> list[int]:
    """Walk `steps` Rodin doubles from `start` (default full 1-2-4-8-7-5)."""
    cur = digital_root(start) or 1
    out = [cur]
    for _ in range(steps - 1):
        cur = rodin_double(cur)
        out.append(cur)
    return out


def rodin_halve_sequence(start: int = 1, steps: int = 6) -> list[int]:
    """Walk `steps` Rodin halves from `start` (default full 1-5-7-8-4-2)."""
    cur = digital_root(start) or 1
    out = [cur]
    for _ in range(steps - 1):
        cur = rodin_halve(cur)
        out.append(cur)
    return out


# Inverse of RODIN_CYCLE (halving walk from 1)
RODIN_HALVE_CYCLE: tuple[int, ...] = (1, 5, 7, 8, 4, 2)


def formation_id_for_sensor(host_id: str, sensor_id: str, generation: int = 0) -> str:
    return f"{host_id}:form:{sensor_id}:g{generation}"


def sensor_thought_id(host_id: str, sensor_id: str) -> str:
    """Stable Thought id for a hard-wired Sensor (reused across Inputs)."""
    return f"{host_id}:sensor:{sensor_id}"


def ensure_sensor_thought(
    sensor: "Sensor",
    *,
    host_id: str,
    with_labels: bool = True,
    existing: Optional[Thought] = None,
) -> Thought:
    """
    Source/grounding Thought for a Sensor.
    Stable id so many Observations share the same Sensor pole.
    """
    if existing is not None:
        return existing
    lab = (lambda s: s) if with_labels else (lambda s: None)
    return Thought(
        id=sensor_thought_id(host_id, sensor.id),
        label=lab(sensor.label or sensor.id),
    )


def begin_sensor_formation(
    sensor: "Sensor",
    *,
    host_id: str,
    generation: int = 0,
    with_labels: bool = True,
    sense: Optional[dict[str, Any]] = None,
    sensor_thought: Optional[Thought] = None,
) -> dict[str, Any]:
    """
    Interface stages (Rodin 1→2) for one Sensor Input.

    Source = Sensor (grounding Thought)
    Target = Observation (Input value Thought)

    Returns a handoff for complete_formation.
    """
    lab = (lambda s: s) if with_labels else (lambda s: None)
    fid = formation_id_for_sensor(host_id, sensor.id, generation)
    p = f"{fid}:"
    sensor_lab = sensor.label or sensor.id

    source = ensure_sensor_thought(
        sensor, host_id=host_id, with_labels=with_labels, existing=sensor_thought
    )

    sense_value: Any = sensor_lab
    if sense is not None:
        sense_value = sense.get("value")
        if sense_value is None and sense.get("sample") is not None:
            sense_value = f"{sensor_lab}:{sense['sample']}"
        elif sense_value is None:
            sense_value = sensor_lab
    observation = Thought(
        id=f"{p}observation",
        label=lab(str(sense_value)),
        transient=True,
    )
    return {
        "kind": "formation_handoff",
        "formation_id": fid,
        "sensor_id": sensor.id,
        "sensor_label": sensor_lab,
        "generation": generation,
        "rodin_at": INTERFACE_RODIN_STAGES[-1],
        "interface_stages": list(INTERFACE_RODIN_STAGES),
        "pending_stages": list(INNERFACE_RODIN_STAGES),
        "with_labels": with_labels,
        "sense": sense,
        "tick": (sense or {}).get("tick"),
        "partial": {
            "sensor": source,
            "observation": observation,
        },
    }


def complete_formation(handoff: dict[str, Any]) -> dict[str, Thought]:
    """
    Innerface stages (Rodin 4→8→7→5): Sensor ⇄ Observation via Perceives/PerceivedBy.
    """
    if handoff.get("kind") != "formation_handoff":
        raise ValueError(f"expected formation_handoff, got kind={handoff.get('kind')!r}")
    partial = handoff.get("partial") or {}
    source = partial.get("sensor") or partial.get("system")
    target = partial.get("observation") or partial.get("environment")
    if not isinstance(source, Thought) or not isinstance(target, Thought):
        raise ValueError(
            "formation_handoff.partial must include sensor (Source) and observation (Target)"
        )

    with_labels = bool(handoff.get("with_labels", True))
    lab = (lambda s: s) if with_labels else (lambda s: None)
    fid = handoff["formation_id"]
    p = f"{fid}:"

    perceives = Thought(id=f"{p}perceives", label=lab("Perceives"))
    perceived_by = Thought(id=f"{p}perceived_by", label=lab("PerceivedBy"))
    link_perceives = Link(
        id=f"{p}sensor_perceives_obs",
        label=lab("SensorPerceivesObservation"),
        source=source,
        link_type=perceives,
        target=target,
    )
    link_perceived_by = Link(
        id=f"{p}obs_perceived_by_sensor",
        label=lab("ObservationPerceivedBySensor"),
        source=target,
        link_type=perceived_by,
        target=source,
    )
    store = {
        t.id: t
        for t in (
            source,
            target,
            perceives,
            perceived_by,
            link_perceives,
            link_perceived_by,
        )
    }
    if not is_minimal_symbioid_shape(store, system_id=source.id, environment_id=target.id):
        raise RuntimeError("complete_formation failed shape check")
    return store


def complete_follows_set(
    observation_a: Thought,
    observation_b: Thought,
    *,
    sync_id: str,
    with_labels: bool = True,
) -> dict[str, Thought]:
    """Lateral six-Thought set: ObservationA ⇄ ObservationB via Follows / FollowedBy."""
    lab = (lambda s: s) if with_labels else (lambda s: None)
    p = f"{sync_id}:"
    follows = Thought(id=f"{p}follows", label=lab("Follows"))
    followed_by = Thought(id=f"{p}followed_by", label=lab("FollowedBy"))
    link_follows = Link(
        id=f"{p}a_follows_b",
        label=lab("ObservationFollowsObservation"),
        source=observation_a,
        link_type=follows,
        target=observation_b,
    )
    link_followed_by = Link(
        id=f"{p}b_followed_by_a",
        label=lab("ObservationFollowedByObservation"),
        source=observation_b,
        link_type=followed_by,
        target=observation_a,
    )
    store = {
        t.id: t
        for t in (
            observation_a,
            observation_b,
            follows,
            followed_by,
            link_follows,
            link_followed_by,
        )
    }
    if not is_minimal_symbioid_shape(
        store, system_id=observation_a.id, environment_id=observation_b.id
    ):
        raise RuntimeError("complete_follows_set failed shape check")
    return store


_SIX_SET_TYPE_LABELS = frozenset(
    {
        "Perceives",
        "PerceivedBy",
        "Follows",
        "FollowedBy",
        "Integrates",
        "IntegratedBy",
        "Expects",
        "ExpectedBy",
        "Has",
        "IsPartOf",
        "ExistsIn",
        "ExistsAround",
    }
)

# Role keys for Rodin-halving integration six-sets
INTEGRATE_ROLES: tuple[str, ...] = (
    "observation_a",
    "observation_b",
    "integrates",
    "link_integrates",
    "integrated_by",
    "link_integrated_by",
)


def six_set_labels(store: dict[str, Thought]) -> list[str]:
    """Labels of a six-Thought set (fallback to id)."""
    return [t.label if t.label else t.id for t in store.values()]


def six_set_poles(store: dict[str, Thought]) -> list[Thought]:
    """The two pole Thoughts of a reciprocal six-set (excludes Links and type roles)."""
    return [
        t
        for t in store.values()
        if not isinstance(t, Link) and (t.label not in _SIX_SET_TYPE_LABELS)
    ]


def extract_observation(store: dict[str, Thought]) -> Optional[Thought]:
    """
    Best-effort Observation pole from a sense/sync/integrate six-set.
    Prefers transient Thoughts, then ids ending in :observation, else second pole.
    """
    for t in store.values():
        if isinstance(t, Link):
            continue
        if t.transient:
            return t
    for t in store.values():
        if isinstance(t, Link):
            continue
        if t.id.endswith(":observation") or ":observation" in t.id:
            return t
    poles = six_set_poles(store)
    if len(poles) >= 2:
        return poles[1]
    return poles[0] if poles else None


def complete_awareness_set(
    host_pole: Thought,
    aspect_pole: Thought,
    *,
    awareness_id: str,
    with_labels: bool = True,
    aspect_kind: str = "Sensor",
) -> dict[str, Thought]:
    """
    Structural awareness six-set: Symbioid/Agent **Has** a Sensor or Actuator.

    Example labels: host "Agent", aspect "Ear" → "Symbioid has Ear" awareness.
    These sets act as **integration terminators**: channels for different
    Sensors/Actuators stay separate under Rodin halving.
    """
    lab = (lambda s: s) if with_labels else (lambda s: None)
    p = f"{awareness_id}:"
    has = Thought(id=f"{p}has", label=lab("Has"))
    is_part_of = Thought(id=f"{p}is_part_of", label=lab("IsPartOf"))
    link_has = Link(
        id=f"{p}host_has_aspect",
        label=lab(f"Has{aspect_kind}"),
        source=host_pole,
        link_type=has,
        target=aspect_pole,
    )
    link_part = Link(
        id=f"{p}aspect_part_of_host",
        label=lab(f"{aspect_kind}IsPartOf"),
        source=aspect_pole,
        link_type=is_part_of,
        target=host_pole,
    )
    store = {
        t.id: t
        for t in (host_pole, aspect_pole, has, is_part_of, link_has, link_part)
    }
    if not is_minimal_symbioid_shape(
        store, system_id=host_pole.id, environment_id=aspect_pole.id
    ):
        raise RuntimeError("complete_awareness_set failed shape check")
    return store


def complete_belief_set(
    expected_observation: Thought,
    feedback: Thought,
    *,
    belief_id: str,
    with_labels: bool = True,
) -> dict[str, Thought]:
    """
    Outerface Belief six-set (manuscript Outerfaces + Expectations).

    An Interface Observation caused by Feedback is stored as the **expected value**
    for that Feedback. Reciprocal six-Thought shape:

      Feedback -Expects→ ExpectedObservation
      ExpectedObservation -ExpectedBy→ Feedback

    Many Beliefs may coexist (humans hold many beliefs at once).
    """
    lab = (lambda s: s) if with_labels else (lambda s: None)
    p = f"{belief_id}:"
    expects = Thought(id=f"{p}expects", label=lab("Expects"))
    expected_by = Thought(id=f"{p}expected_by", label=lab("ExpectedBy"))
    link_expects = Link(
        id=f"{p}feedback_expects_obs",
        label=lab("FeedbackExpectsObservation"),
        source=feedback,
        link_type=expects,
        target=expected_observation,
    )
    link_expected_by = Link(
        id=f"{p}obs_expected_by_feedback",
        label=lab("ObservationExpectedByFeedback"),
        source=expected_observation,
        link_type=expected_by,
        target=feedback,
    )
    store = {
        t.id: t
        for t in (
            feedback,
            expected_observation,
            expects,
            expected_by,
            link_expects,
            link_expected_by,
        )
    }
    if not is_minimal_symbioid_shape(
        store, system_id=feedback.id, environment_id=expected_observation.id
    ):
        raise RuntimeError("complete_belief_set failed shape check")
    return store


def complete_integrate_set(
    observation_a: Thought,
    observation_b: Thought,
    *,
    integrate_id: str,
    with_labels: bool = True,
) -> dict[str, Thought]:
    """
    H1: Rodin-halving integration six-set — two Observations → Integrates/IntegratedBy.

    Reduces a pair of Input-level Observations into one reciprocal six-Thought pattern
    (symmetric to complete_follows_set, different link semantics).
    """
    if observation_a.id == observation_b.id:
        raise ValueError("cannot integrate an Observation with itself")
    lab = (lambda s: s) if with_labels else (lambda s: None)
    p = f"{integrate_id}:"
    integrates = Thought(id=f"{p}integrates", label=lab("Integrates"))
    integrated_by = Thought(id=f"{p}integrated_by", label=lab("IntegratedBy"))
    link_integrates = Link(
        id=f"{p}a_integrates_b",
        label=lab("ObservationIntegratesObservation"),
        source=observation_a,
        link_type=integrates,
        target=observation_b,
    )
    link_integrated_by = Link(
        id=f"{p}b_integrated_by_a",
        label=lab("ObservationIntegratedByObservation"),
        source=observation_b,
        link_type=integrated_by,
        target=observation_a,
    )
    store = {
        t.id: t
        for t in (
            observation_a,
            observation_b,
            integrates,
            integrated_by,
            link_integrates,
            link_integrated_by,
        )
    }
    if not is_minimal_symbioid_shape(
        store, system_id=observation_a.id, environment_id=observation_b.id
    ):
        raise RuntimeError("complete_integrate_set failed shape check")
    return store


def format_six_set_line(kind: str, store: dict[str, Thought], *, index: Optional[int] = None) -> str:
    """One-line label dump for a completed six-set."""
    poles = six_set_poles(store)
    if kind == "sense" and len(poles) >= 2:
        title = f"sense {poles[0].label}→{poles[1].label}"
    elif kind == "sync" and len(poles) >= 2:
        title = f"sync {poles[0].label}⇄{poles[1].label}"
    elif kind == "integrate" and len(poles) >= 2:
        title = f"integrate {poles[0].label}⊕{poles[1].label}"
    elif kind == "depth" and len(poles) >= 2:
        title = f"depth {poles[0].label}⊕{poles[1].label}"
    elif kind == "belief" and len(poles) >= 2:
        # poles: Feedback, ExpectedObservation — compact (no full six-set dump)
        title = f"belief {poles[0].label} anticipates {poles[1].label}"
        if index is not None:
            return f"[{index}] {title}"
        return title
    elif kind == "belief" and len(poles) >= 1:
        title = f"belief {poles[0].label}"
        if index is not None:
            return f"[{index}] {title}"
        return title
    elif kind == "awareness" and len(poles) >= 2:
        title = f"awareness {poles[0].label} has {poles[1].label}"
    elif kind == "twin":
        title = "twin seed"
    else:
        title = kind
    body = ", ".join(six_set_labels(store))
    if index is not None:
        return f"[{index}] {title}: {body}"
    return f"{title}: {body}"


def emit_six_set(kind: str, store: dict[str, Thought], *, index: Optional[int] = None) -> None:
    """Print a six-set at the moment it is completed (only process-loop output)."""
    print(format_six_set_line(kind, store, index=index), flush=True)
