"""
Core package: one class per module (+ seed/formation/ids helpers).

Hierarchy:

  System
    ├── Thought → Link
    ├── Symbioid
    ├── Body, Mind, Sensor, Actuator

  Process
    ├── Innerface
    ├── Interface
    └── Outerface
"""

from symbioid.Core.Actuator import Actuator
from symbioid.Core.Body import Body
from symbioid.Core.Innerface import Innerface
from symbioid.Core.Interface import Interface
from symbioid.Core.Law import Law, constitutional_seed
from symbioid.Core.Link import Link
from symbioid.Core.Mind import Mind
from symbioid.Core.Outerface import Outerface
from symbioid.Core.Process import Process
from symbioid.Core.Sensor import Sensor
from symbioid.Core.Symbioid import Symbioid
from symbioid.Core.System import System
from symbioid.Core.Thought import Thought
from symbioid.Core.formation import (
    FORMATION_ROLES,
    FOLLOWS_ROLES,
    INNERFACE_RODIN_STAGES,
    INTEGRATE_ROLES,
    INTERFACE_RODIN_STAGES,
    RODIN_CYCLE,
    RODIN_HALVE_CYCLE,
    begin_sensor_formation,
    complete_awareness_set,
    complete_belief_set,
    complete_follows_set,
    complete_formation,
    complete_integrate_set,
    console_emit_enabled,
    digital_root,
    emit_six_set,
    ensure_sensor_thought,
    extract_observation,
    format_six_set_line,
    rodin_double,
    rodin_halve,
    rodin_halve_sequence,
    rodin_sequence,
    set_console_emit,
    six_set_labels,
    six_set_poles,
)
from symbioid.Core.seed import is_minimal_symbioid_shape, minimal_seed

__all__ = [
    "System",
    "Thought",
    "Link",
    "Body",
    "Mind",
    "Sensor",
    "Actuator",
    "Process",
    "Innerface",
    "Interface",
    "Outerface",
    "Law",
    "Symbioid",
    "minimal_seed",
    "constitutional_seed",
    "is_minimal_symbioid_shape",
    "RODIN_CYCLE",
    "RODIN_HALVE_CYCLE",
    "INTERFACE_RODIN_STAGES",
    "INNERFACE_RODIN_STAGES",
    "FORMATION_ROLES",
    "FOLLOWS_ROLES",
    "INTEGRATE_ROLES",
    "digital_root",
    "rodin_double",
    "rodin_halve",
    "rodin_sequence",
    "rodin_halve_sequence",
    "begin_sensor_formation",
    "complete_formation",
    "complete_follows_set",
    "complete_integrate_set",
    "complete_belief_set",
    "complete_awareness_set",
    "ensure_sensor_thought",
    "extract_observation",
    "six_set_labels",
    "six_set_poles",
    "format_six_set_line",
    "emit_six_set",
    "set_console_emit",
    "console_emit_enabled",
]
