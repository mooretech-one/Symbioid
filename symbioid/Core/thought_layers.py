"""
Thought MVP layers — Simon Atomic Thought + SevenSphere alignment.

Sharper type surface so one word "Thought" is not overloaded:

  STRUCTURE — long-lived graph poles (seed, LinkTypes, awareness poles)
  PATTERN   — dynamical / formation content (Observations, Links as events)
  FEELING   — policy/affect poles (Actions, valence-coupled targets)

Mind is **not** a Thought (see ``assert_mind_not_thought``).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union


class ThoughtLayer(str, Enum):
    """Cognitive layer of a Thought (Structure / Pattern / Feeling)."""

    STRUCTURE = "structure"
    PATTERN = "pattern"
    FEELING = "feeling"


# Simon Atomic Thought — dual face of structure vs signal (design contract)
SIMON_ATOMIC_THOUGHT = {
    "structure": "id/label/links — long-term graph identity",
    "signal": "activation/threshold/decay — short-term energy (neuron-like)",
    "link": "Source + LinkType + Target are all Thoughts",
    "not_1to1_neuron": "One Thought is not one biological neuron (Simon)",
}

# SevenSphere — author tool / mapping poles (design contract, not physics)
SEVENSPHERE_ALIGNMENT = {
    "poles": "Multiple simultaneous aspects of a being (nested spheres)",
    "center": "Host identity (Symbioid / Self) is not a graph Thought alone",
    "map_to_layers": {
        ThoughtLayer.STRUCTURE: "stable self/world description (seed, types)",
        ThoughtLayer.PATTERN: "co-activated events and formations",
        ThoughtLayer.FEELING: "valued Action / Feedback policy poles",
    },
}


def normalize_layer(
    layer: Optional[Union[ThoughtLayer, str]] = None,
    *,
    default: ThoughtLayer = ThoughtLayer.PATTERN,
) -> ThoughtLayer:
    if layer is None:
        return default
    if isinstance(layer, ThoughtLayer):
        return layer
    try:
        return ThoughtLayer(str(layer).strip().lower())
    except ValueError:
        return default


def assert_mind_not_thought(obj: Any) -> None:
    """
    Enforce Mind ≠ Thought at type boundaries.

    Mind is a System aspect (processor/substrate). It must never be a Thought
    subclass and must not be treated as graph content.
    """
    from symbioid.Core.Mind import Mind
    from symbioid.Core.Thought import Thought

    if not isinstance(obj, Mind):
        raise TypeError(f"expected Mind, got {type(obj)!r}")
    if isinstance(obj, Thought):
        raise TypeError("Mind must not be a Thought subclass (Mind ≠ Thought)")
    # Runtime inheritance guard (future accidental subclassing)
    if issubclass(type(obj), Thought) and type(obj) is not Thought:
        raise TypeError(f"Mind type {type(obj)!r} illegally subclasses Thought")


def is_thought_content(obj: Any) -> bool:
    """True if obj is graph/dynamical content (Thought or Link)."""
    from symbioid.Core.Thought import Thought

    return isinstance(obj, Thought)


def is_mind_aspect(obj: Any) -> bool:
    """True if obj is the Mind substrate (not Thought content)."""
    from symbioid.Core.Mind import Mind
    from symbioid.Core.Thought import Thought

    return isinstance(obj, Mind) and not isinstance(obj, Thought)


def layer_for_role(role: str) -> ThoughtLayer:
    """
    Default layer for common formation roles.

    role examples: sensor, observation, perceives, follows, action, feedback, seed
    """
    r = (role or "").strip().lower()
    if r in (
        "system",
        "environment",
        "exists_in",
        "exists_around",
        "existsin",
        "existsaround",
        "perceives",
        "perceived_by",
        "follows",
        "followed_by",
        "integrates",
        "integrated_by",
        "expects",
        "expected_by",
        "has",
        "is_part_of",
        "seed",
        "link_type",
        "law",
        "awareness",
    ):
        return ThoughtLayer.STRUCTURE
    if r in ("action", "act", "feedback", "feeling", "valence_pole"):
        return ThoughtLayer.FEELING
    # observation, link instance, transient pattern, default
    return ThoughtLayer.PATTERN
