"""Light processing and the two 1D synthetic modes.

``preview`` filters the whole property cube; ``sparse`` builds K vertical
traces at chosen locations.  Both are screening paths, and neither produces
a seismic image - that is what ``imaging`` is for.
"""

from .preview import (
    PREVIEW_LABEL, ConvolutionPreview, convolution_preview, convolve_reflectivity,
    reflectivity, synthetic_columns, time_from_depth,
)
from .sparse import (
    LAYOUTS, SPARSE_LABEL, SparseSynthetic, TraceLocation, locations_from_points,
    locations_from_wells, locations_on_grid, resolve_locations, snap_to,
    sparse_synthetic,
)
from .sim2seis import (
    DEFAULT_STACKS, SIM2SEIS_LABEL, AngleStack, Sim2SeisVolume, aki_richards,
    build_stacks, sim2seis_volume, stacked_reflectivity,
)
from .filters import agc, bandpass, direct_wave_mute, normalise, taper_mute

__all__ = [
    "PREVIEW_LABEL", "ConvolutionPreview", "convolution_preview", "reflectivity",
    "convolve_reflectivity", "synthetic_columns", "time_from_depth",
    "DEFAULT_STACKS", "SIM2SEIS_LABEL", "AngleStack", "Sim2SeisVolume",
    "aki_richards", "build_stacks", "sim2seis_volume", "stacked_reflectivity",
    "LAYOUTS", "SPARSE_LABEL", "SparseSynthetic", "TraceLocation",
    "locations_from_points", "locations_from_wells", "locations_on_grid",
    "resolve_locations", "snap_to", "sparse_synthetic",
    "agc", "bandpass", "direct_wave_mute", "normalise", "taper_mute",
]
