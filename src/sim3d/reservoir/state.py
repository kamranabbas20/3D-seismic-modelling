"""The reservoir state variables and their invariants (spec section 39).

Porosity, net-to-gross, shale volume, pressure, temperature and the three
saturations are stored independently.  Reservoir state is never embedded
directly into seismic values: it is an input to rock physics, which is an
input to the earth model, which is an input to the wave equation.  Keeping
those layers separate is what makes the 4D decomposition meaningful.

The one hard invariant is saturation closure, ``Sw + So + Sg = 1``, checked
on construction and after every perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..core.errors import ConfigError, ValidationError
from ..core.grid import Grid3D

SATURATION_TOLERANCE = 1e-6


@dataclass
class ReservoirState:
    """Petrophysical and dynamic state on a grid, all SI (temperature in degC)."""

    grid: Grid3D
    porosity: np.ndarray
    ntg: np.ndarray
    vsh: np.ndarray
    pressure: np.ndarray          #: pore pressure, Pa
    sw: np.ndarray
    so: np.ndarray
    sg: np.ndarray
    temperature: np.ndarray | float = 80.0
    reservoir_mask: np.ndarray | None = None
    name: str = "state"
    provenance: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.reservoir_mask is None:
            self.reservoir_mask = np.ones(self.grid.shape, dtype=bool)
        self.validate()

    # -- invariants ------------------------------------------------------
    def validate(self, tolerance: float = SATURATION_TOLERANCE) -> None:
        """Raise if the state is unphysical."""
        for label in ("porosity", "ntg", "vsh", "sw", "so", "sg"):
            arr = np.asarray(getattr(self, label))
            if arr.shape != self.grid.shape:
                raise ValidationError(
                    f"{label} has shape {arr.shape} but the grid is {self.grid.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValidationError(f"{label} contains non-finite values")
            if np.any(arr < -tolerance) or np.any(arr > 1.0 + tolerance):
                raise ValidationError(
                    f"{label} must lie in [0, 1]; it spans "
                    f"[{arr.min():.6f}, {arr.max():.6f}]"
                )
        total = self.sw + self.so + self.sg
        if np.any(np.abs(total - 1.0) > tolerance):
            worst = np.unravel_index(int(np.argmax(np.abs(total - 1.0))), total.shape)
            raise ValidationError(
                f"saturations must sum to 1 within {tolerance}; the worst cell "
                f"{worst} sums to {total[worst]:.8f}"
            )
        if np.any(self.pressure <= 0):
            raise ValidationError(
                f"pore pressure must be positive; minimum is {np.min(self.pressure):g} Pa"
            )

    def renormalise(self) -> "ReservoirState":
        """Rescale saturations to close exactly, then re-validate."""
        total = self.sw + self.so + self.sg
        if np.any(total <= 0):
            raise ValidationError("cannot renormalise: some cell has zero total saturation")
        return replace(self, sw=self.sw / total, so=self.so / total, sg=self.sg / total)

    # -- derived ---------------------------------------------------------
    @property
    def saturations(self) -> dict[str, np.ndarray]:
        """Phase name -> saturation, in the form the rock-physics chain wants."""
        return {"brine": self.sw, "oil": self.so, "gas": self.sg}

    def copy(self, **overrides) -> "ReservoirState":
        """A copy with selected fields replaced, arrays duplicated."""
        base = dict(
            grid=self.grid, porosity=self.porosity.copy(), ntg=self.ntg.copy(),
            vsh=self.vsh.copy(), pressure=self.pressure.copy(), sw=self.sw.copy(),
            so=self.so.copy(), sg=self.sg.copy(), temperature=self.temperature,
            reservoir_mask=self.reservoir_mask.copy(), name=self.name,
            provenance=list(self.provenance),
        )
        base.update(overrides)
        return ReservoirState(**base)

    def difference(self, baseline: "ReservoirState") -> dict[str, np.ndarray]:
        """``self - baseline`` for every dynamic variable (spec section 57)."""
        return {
            "dP": self.pressure - baseline.pressure,
            "dSw": self.sw - baseline.sw,
            "dSo": self.so - baseline.so,
            "dSg": self.sg - baseline.sg,
        }

    def summary(self) -> str:
        res = self.reservoir_mask
        def rng(label, arr, scale=1.0, unit=""):
            a = np.asarray(arr, dtype=float)
            a = a[res] if a.shape == self.grid.shape else np.atleast_1d(a)
            return f"  {label:12s} {a.min() / scale:9.3f} - {a.max() / scale:9.3f} {unit}"
        lines = [f"Reservoir state '{self.name}' "
                 f"({int(res.sum()):,} reservoir cells of {self.grid.n_cells:,})",
                 rng("porosity", self.porosity),
                 rng("NTG", self.ntg),
                 rng("Vsh", self.vsh),
                 rng("pressure", self.pressure, 1e5, "bar"),
                 rng("Sw", self.sw), rng("So", self.so), rng("Sg", self.sg)]
        lines += [f"  from: {p}" for p in self.provenance]
        return "\n".join(lines)


def saturation_from_contacts(z: np.ndarray, sw_irreducible: float,
                             owc: float | None, goc: float | None,
                             transition: float = 0.0, sw_max: float = 1.0):
    """Saturations against depth for a gas/oil/water column.

    Returns ``(sw, so, sg)``.  Depth increases downwards, so the water leg
    is *below* ``owc`` and any gas cap is *above* ``goc``.

    The contacts are horizontal planes, which is the point of modelling
    them at all: a contact cuts across dipping stratigraphy, so it is the
    one reflector in the model whose geometry owes nothing to the layering
    - the flat spot.

    ``transition`` is the height above the oil-water contact over which
    water saturation falls from ``sw_max`` to ``sw_irreducible``.  It is a
    linear ramp, not a J-function: a capillary saturation-height curve needs
    a pore-throat model the rest of this tool does not carry, and inventing
    one would dress a straight line up as measurement.  Zero gives a sharp
    contact.

    ``sw_max`` is the water saturation of the water leg.  It defaults to 1,
    but a flow simulation cannot hold more water than its relative
    permeability allows: the Corey model clamps saturation into
    ``[swc, 1 - sor]`` on every step, so a water leg initialised at 1 is
    pulled to ``1 - sor`` on the first one and the difference appears as
    hydrocarbon that was never there.  On this project that manufactured a
    25% oil saturation across the whole water leg and a spurious 4D
    response equal and opposite to the real flood.
    """
    if not 0.0 <= sw_irreducible <= 1.0:
        raise ConfigError(
            f"irreducible water saturation must be in [0, 1], got {sw_irreducible}")
    if not sw_irreducible <= sw_max <= 1.0:
        raise ConfigError(
            f"the water leg saturation must lie between the irreducible value "
            f"and 1, got sw_max={sw_max} with Swirr={sw_irreducible}")
    if transition < 0.0:
        raise ConfigError(f"transition zone must not be negative, got {transition}")
    if owc is not None and goc is not None and goc >= owc:
        raise ConfigError(
            f"the gas-oil contact must sit above the oil-water contact, got "
            f"goc={goc:g} m and owc={owc:g} m (depth increases downwards)")

    sw = np.full(z.shape, float(sw_irreducible))
    if owc is not None:
        if transition > 0.0:
            height = np.clip((owc - z) / transition, 0.0, 1.0)
        else:
            height = (z < owc).astype(float)
        sw = sw_max + height * (sw_irreducible - sw_max)
    sg = np.zeros(z.shape)
    if goc is not None:
        # A gas cap displaces the oil, not the irreducible water.
        in_gas = z < goc
        sg = np.where(in_gas, 1.0 - sw, 0.0)
    so = np.clip(1.0 - sw - sg, 0.0, 1.0)
    return sw, so, sg


def initial_state(geology, pressure_gradient: float = 10500.0,
                  datum_pressure: float = 101325.0,
                  sw: float = 0.30, sg: float = 0.0, temperature: float = 80.0,
                  owc: float | None = None, goc: float | None = None,
                  transition: float = 0.0, sw_max: float = 1.0,
                  name: str = "baseline") -> ReservoirState:
    """Build a baseline state from a geological model.

    Pore pressure follows a hydrostatic gradient
    ``P = datum_pressure + gradient * z`` with ``gradient`` in Pa/m
    (10,500 Pa/m is a typical brine gradient, about 0.105 bar/m).  The
    default datum is one atmosphere, so ``z = 0`` is a land surface; for a
    marine setting add the water column to ``datum_pressure``.

    With no ``owc`` the saturations are constant through the reservoir,
    which is the older behaviour and is what a mechanistic sweep test
    wants.  Given one, ``sw`` becomes the irreducible saturation of the
    hydrocarbon column and the model grows a water leg, an optional
    capillary transition and, with ``goc``, a gas cap - see
    :func:`saturation_from_contacts`.  Cells outside the reservoir are
    fully brine-saturated either way.
    """
    grid = geology.grid
    z = grid.axis(2)[None, None, :] * np.ones(grid.shape)
    pressure = datum_pressure + pressure_gradient * z

    mask = geology.reservoir_mask
    provenance = [f"hydrostatic gradient {pressure_gradient:g} Pa/m"]
    if owc is None and goc is None:
        sw_field = np.where(mask, sw, 1.0)
        sg_field = np.where(mask, sg, 0.0)
        so_field = np.clip(1.0 - sw_field - sg_field, 0.0, 1.0)
        provenance.append(f"reservoir Sw = {sw:g}, Sg = {sg:g}, no contacts")
    else:
        sw_c, so_c, sg_c = saturation_from_contacts(z, sw, owc, goc, transition,
                                                    sw_max=sw_max)
        sw_field = np.where(mask, sw_c, 1.0)
        so_field = np.where(mask, so_c, 0.0)
        sg_field = np.where(mask, sg_c, 0.0)
        provenance.append(
            "contacts: "
            + (f"OWC {owc:g} m" if owc is not None else "no OWC")
            + (f", GOC {goc:g} m" if goc is not None else "")
            + (f", {transition:g} m transition" if transition else ", sharp")
            + f", Swirr = {sw:g}, water leg Sw = {sw_max:g}")

    return ReservoirState(
        grid=grid, porosity=geology.porosity, ntg=geology.ntg, vsh=geology.vsh,
        pressure=pressure, sw=sw_field, so=so_field, sg=sg_field,
        temperature=temperature, reservoir_mask=mask, name=name,
        provenance=provenance,
    )
