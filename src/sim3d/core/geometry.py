"""Shared map-view geometry conventions.

One azimuth convention is used everywhere in sim3d: **degrees clockwise
from ``+y`` (map north)**.  It applies to fault strike, surface dip
direction, the long axis of an anisotropic random field, and the long axis
of a pressure halo or saturation front alike.  Having a single convention
in one place is the only way to keep a fault, the permeability anisotropy
it juxtaposes, and the flood front that follows them all pointing the same
way.
"""

from __future__ import annotations

import numpy as np


def bearing_vector(azimuth: float) -> tuple[float, float]:
    """Unit map vector at ``azimuth`` degrees clockwise from ``+y``."""
    a = np.radians(azimuth)
    return float(np.sin(a)), float(np.cos(a))


def to_principal(dx, dy, azimuth: float):
    """Rotate map offsets into a frame whose major axis is at ``azimuth``.

    Returns ``(major, minor)``: the component along the azimuth direction,
    and the component 90 degrees clockwise from it.  With ``azimuth = 0``
    the major axis is ``+y`` and the minor axis is ``+x``.
    """
    a = np.radians(azimuth)
    sin_a, cos_a = np.sin(a), np.cos(a)
    major = dx * sin_a + dy * cos_a
    minor = dx * cos_a - dy * sin_a
    return major, minor
