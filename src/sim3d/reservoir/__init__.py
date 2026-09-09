"""Mechanistic reservoir state: pressure and saturation without a simulator.

Spec sections 30-39.  The platform does not require a black-oil simulator
to ask causal seismic questions.  It generates controllable, plausible
spatial distributions of pressure and saturation directly, so an experiment
can vary one cause at a time - which a coupled flow simulation makes
harder, not easier.

The same rock-physics and seismic pipeline runs unchanged on states that
come from here or from an imported simulation (spec section 147).
"""

from .state import ReservoirState, initial_state
from .mechanistic import (
    GasBreakout, PressureHalo, ReservoirScenario, SaturationFront, TIME_STATES,
)
from .relperm import CoreyRelativePermeability
from .flow import FlowResult, FlowSettings, FlowSimulator, WellHistory

__all__ = [
    "ReservoirState", "initial_state", "PressureHalo", "SaturationFront",
    "GasBreakout", "ReservoirScenario", "TIME_STATES",
    "CoreyRelativePermeability", "FlowSettings", "FlowSimulator", "FlowResult",
    "WellHistory",
]
