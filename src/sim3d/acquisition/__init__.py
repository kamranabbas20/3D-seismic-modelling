"""Synthetic acquisition geometry (spec sections 66-72)."""

from .geometry import (
    Acquisition, OBNGeometry, fold_map, offset_distribution, azimuth_distribution,
)

__all__ = ["Acquisition", "OBNGeometry", "fold_map", "offset_distribution",
           "azimuth_distribution"]
