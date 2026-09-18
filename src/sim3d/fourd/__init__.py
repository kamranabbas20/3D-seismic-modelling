"""4D scenario construction, decomposition and metrics (spec sections 48-58, 87-92)."""

from .scenarios import (
    SCENARIO_NAMES, FourDEarth, FourDStates, build_earth_models, build_states,
    build_states_from_flow, confining_pressure_from_density,
)
from .decomposition import decompose, interaction
from .metrics import (
    cross_correlation, local_time_shift, nrms, radial_profile, rms,
    well_centred_statistics,
)

__all__ = [
    "SCENARIO_NAMES", "FourDStates", "FourDEarth", "build_states",
    "build_states_from_flow", "build_earth_models",
    "confining_pressure_from_density",
    "decompose", "interaction",
    "rms", "nrms", "cross_correlation", "local_time_shift",
    "radial_profile", "well_centred_statistics",
]
