r"""Planar faults: horizon displacement and flow compartmentalisation.

Spec sections 19 and 37.  A fault here is *kinematic*, not mechanical: it
displaces the stratigraphy across a plane and attenuates pressure and
saturation transport through it.  It does not solve for stress, and it does
not create a damage zone or a drag fold.  That is enough to answer the
questions the platform is for - does a sealing fault stop the flood front,
does it split the pressure response into compartments, is the offset
resolvable on seismic - and it is honest about what it is not.

Geometry
--------
Strike is degrees clockwise from ``+y`` (map north); dip is degrees from
horizontal, dipping down towards ``strike + 90``.  With ``z`` positive
downwards, the unit normal is

.. math:: \hat n = (\cos\phi\sin\delta,\; -\sin\phi\sin\delta,\; -\cos\delta)

and the signed distance of a point from the plane is
:math:`\hat n\cdot(\mathbf p - \mathbf p_0)`.  Points with positive signed
distance are on the hanging wall.

Sign convention: a positive ``throw`` moves the hanging wall **down**, a
normal fault.  A negative throw is a reverse fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.errors import ConfigError


@dataclass
class Fault:
    """A planar fault with finite extent, displacement and transmissibility."""

    name: str = "F1"
    origin: tuple[float, float, float] = (1500.0, 1500.0, 1500.0)
    strike: float = 0.0
    dip: float = 65.0
    throw: float = 30.0
    #: Half-length along strike and half-height down dip, in metres.
    #: ``None`` means unlimited in that direction.
    strike_extent: float | None = None
    dip_extent: float | None = None
    #: Width of the smoothing zone across the plane, in metres.  A fault
    #: sharper than the grid produces stair-step artefacts, not resolution.
    zone_width: float = 20.0
    #: 1.0 fully transmissive, 0.0 sealing, in between partially sealing.
    transmissibility: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.dip <= 90.0:
            raise ConfigError(f"dip must be within (0, 90] degrees, got {self.dip}")
        if not 0.0 <= self.transmissibility <= 1.0:
            raise ConfigError(
                f"transmissibility must be in [0, 1], got {self.transmissibility}"
            )
        if self.zone_width <= 0:
            raise ConfigError(f"zone_width must be positive, got {self.zone_width}")

    @property
    def normal(self) -> np.ndarray:
        phi, delta = np.radians(self.strike), np.radians(self.dip)
        return np.array([np.cos(phi) * np.sin(delta),
                         -np.sin(phi) * np.sin(delta),
                         -np.cos(delta)])

    @property
    def strike_vector(self) -> np.ndarray:
        phi = np.radians(self.strike)
        return np.array([np.sin(phi), np.cos(phi), 0.0])

    @property
    def dip_vector(self) -> np.ndarray:
        """Down-dip unit vector in the fault plane."""
        return np.cross(self.normal, self.strike_vector)

    def signed_distance(self, x, y, z) -> np.ndarray:
        """Perpendicular distance from the fault plane; positive on the hanging wall."""
        n = self.normal
        return (n[0] * (x - self.origin[0]) + n[1] * (y - self.origin[1])
                + n[2] * (z - self.origin[2]))

    def within_extent(self, x, y, z) -> np.ndarray:
        """Boolean mask of where the finite fault surface actually exists."""
        dx, dy, dz = x - self.origin[0], y - self.origin[1], z - self.origin[2]
        inside = np.ones(np.broadcast(dx, dy, dz).shape, dtype=bool)
        if self.strike_extent is not None:
            s = self.strike_vector
            along = dx * s[0] + dy * s[1] + dz * s[2]
            inside &= np.abs(along) <= self.strike_extent
        if self.dip_extent is not None:
            d = self.dip_vector
            down = dx * d[0] + dy * d[1] + dz * d[2]
            inside &= np.abs(down) <= self.dip_extent
        return inside

    def hanging_wall_fraction(self, x, y, z) -> np.ndarray:
        """Smooth 0-to-1 indicator of the hanging-wall side.

        The transition is spread over ``zone_width`` so that displacement
        is applied gradually rather than as a one-cell step, which would
        alias badly against any grid.
        """
        s = self.signed_distance(x, y, z)
        frac = 0.5 * (1.0 + np.tanh(2.0 * s / self.zone_width))
        return np.where(self.within_extent(x, y, z), frac, 0.0)

    def restore(self, x, y, z) -> np.ndarray:
        """Map faulted coordinates back to their pre-faulting depth.

        Horizons are defined on the undeformed stratigraphy; evaluating them
        at ``restore(x, y, z)`` displaces every horizon across the fault at
        once, keeping them mutually consistent.

        The sign is what makes a positive throw a *normal* fault: subtracting
        the throw on the hanging wall means a horizon is found at a greater
        present-day depth there, so the hanging wall has moved down.
        """
        return z - self.throw * self.hanging_wall_fraction(x, y, z)

    def transmissibility_multiplier(self, x, y, z, distance_scale: float | None = None):
        """Multiplier in ``[transmissibility, 1]`` for transport across the plane.

        Used by the mechanistic reservoir module to attenuate a pressure
        halo or arrest a saturation front at a sealing fault.
        """
        width = distance_scale if distance_scale is not None else self.zone_width
        s = np.abs(self.signed_distance(x, y, z))
        near = np.exp(-((s / width) ** 2))
        near = np.where(self.within_extent(x, y, z), near, 0.0)
        return 1.0 - (1.0 - self.transmissibility) * near

    def describe(self) -> str:
        kind = "normal" if self.throw > 0 else ("reverse" if self.throw < 0 else "no-slip")
        seal = ("sealing" if self.transmissibility == 0 else
                "transmissive" if self.transmissibility == 1 else
                f"partially sealing ({self.transmissibility:g})")
        return (f"fault {self.name}: {kind}, strike {self.strike:g} deg, dip "
                f"{self.dip:g} deg, throw {abs(self.throw):g} m, {seal}")


@dataclass
class FaultSet:
    """Several faults acting together."""

    faults: list[Fault] = field(default_factory=list)

    def __iter__(self):
        return iter(self.faults)

    def __len__(self) -> int:
        return len(self.faults)

    def restore(self, x, y, z) -> np.ndarray:
        """Apply every fault's displacement in turn.

        Faults are applied in list order.  For intersecting faults the order
        matters geologically (which cuts which), so it is the user's to set,
        not something to be inferred.
        """
        out = np.asarray(z, dtype=float)
        for fault in self.faults:
            out = fault.restore(x, y, out)
        return out

    def transmissibility_multiplier(self, x, y, z) -> np.ndarray:
        """Product of every fault's multiplier: the most sealing one dominates."""
        out = np.ones(np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape)
        for fault in self.faults:
            out = out * fault.transmissibility_multiplier(x, y, z)
        return out

    def compartment_id(self, x, y, z) -> np.ndarray:
        """Integer compartment label from which side of each fault a point is on."""
        label = np.zeros(np.broadcast(np.asarray(x), np.asarray(y), np.asarray(z)).shape,
                         dtype=int)
        for bit, fault in enumerate(self.faults):
            side = (fault.hanging_wall_fraction(x, y, z) > 0.5).astype(int)
            label |= side << bit
        return label

    def describe(self) -> str:
        return "\n".join(f.describe() for f in self.faults) or "no faults"
