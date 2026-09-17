"""Turning picks made on a section into a stratigraphy the builder accepts.

A user draws where the *base* of a unit should sit, because that is the
intuitive thing to draw.  What the model stores is how *thick* that makes
the unit, measured from its own top and clamped at zero.

That indirection is the whole of the design.  Horizons in this codebase
accumulate from thicknesses:

.. math:: z_{k+1}(x, y) = z_k(x, y) + h_k(x, y), \\qquad h_k \\ge 0

so as long as every thickness is non-negative the tops are monotone and can
touch but never cross - which is the invariant
:func:`~sim3d.geology.builder.build_geology` enforces.  Storing the drawn
depths instead would let the next structural edit push two horizons through
each other, and the model would be refused.  Clamping at zero means a base
drawn above its own top is a pinchout rather than an impossible model.

This lives in the geology package rather than in the interface that
collects the clicks: it is geometry, it is testable without a browser, and
the interface is meant to be a frontend.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def _section_top(layers, index: int, grid, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """One unit's top along a section through the middle of the model."""
    if axis not in (0, 1):
        raise ConfigError(f"axis must be 0 (x) or 1 (y), got {axis}")
    if not 0 <= index < len(layers):
        raise ConfigError(
            f"there is no unit {index} in a stack of {len(layers)}")
    other = 1 - axis
    fixed = grid.axis(other)
    middle = grid.origin[other] + 0.5 * grid.extent[other]
    column = int(np.argmin(np.abs(fixed - middle)))
    surface = layers[index].top.on_grid(grid)
    top = surface[:, column] if axis == 0 else surface[column, :]
    return grid.axis(axis), top


def picks_to_profile(picks, layers, index: int, grid, axis: int = 0) -> dict:
    """Clicked ``(position, depth)`` pairs into a stored thickness profile.

    Returns the ``thickness_profile`` dictionary
    :func:`~sim3d.geology.templates.layer_cake` takes.  Picks may arrive in
    any order and anywhere, including above the unit's own top: those become
    a thickness of zero.
    """
    along, top = _section_top(layers, index, grid, axis)
    points = []
    for position, depth in sorted((float(a), float(b)) for a, b in picks):
        here = float(np.interp(position, along, top))
        points.append([position, max(float(depth) - here, 0.0)])
    return {"axis": int(axis), "points": points}


def profile_to_picks(profile: dict, layers, index: int, grid) -> list:
    """The stored thickness back into section picks, so a drawing can be resumed.

    The inverse of :func:`picks_to_profile` up to the clamp, which is not
    invertible: a pick that was flattened to zero comes back sitting on the
    unit's top, which is where it ended up rather than where it was made.
    """
    axis = int(profile.get("axis", 0))
    along, top = _section_top(layers, index, grid, axis)
    return [[float(position),
             float(np.interp(float(position), along, top)) + float(thickness)]
            for position, thickness in (profile.get("points") or [])]


def profile_is_empty(profile: dict) -> bool:
    """Is this drawing one that would leave the unit absent everywhere?"""
    points = profile.get("points") or []
    return not points or max(float(t) for _, t in points) <= 0.0
