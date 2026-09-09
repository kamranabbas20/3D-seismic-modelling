"""Rock physics: the transformation from reservoir state to elastic properties.

The chain (spec section 40) is::

    minerals + porosity -> dry frame -> fluid properties -> mixed fluid
        -> Gassmann -> Vp, Vs, rho -> AI, SI

Every intermediate stays accessible; nothing is collapsed into a single
opaque velocity function.  Each model documents its equation, assumptions,
units and source.
"""

from .minerals import (
    MINERALS, Mineral, hashin_shtrikman, mixed_mineral, reuss, voigt, vrh,
)
from .fluids import (
    FluidState, brine_properties, gas_properties, mix_fluids, oil_properties,
)
from .dryframe import (
    critical_porosity_model, hertz_mindlin, soft_sand, stiff_sand,
)
from .gassmann import gassmann_saturated_modulus, gassmann_substitute
from .pressure import PressureModel, effective_stress
from .model import RockPhysicsConfig, RockPhysicsResult, elastic_from_state

__all__ = [
    "MINERALS", "Mineral", "hashin_shtrikman", "mixed_mineral", "reuss", "voigt", "vrh",
    "FluidState", "brine_properties", "gas_properties", "mix_fluids", "oil_properties",
    "critical_porosity_model", "hertz_mindlin", "soft_sand", "stiff_sand",
    "gassmann_saturated_modulus", "gassmann_substitute",
    "PressureModel", "effective_stress",
    "RockPhysicsConfig", "RockPhysicsResult", "elastic_from_state",
]
