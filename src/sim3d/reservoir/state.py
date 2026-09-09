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

from ..core.errors import ValidationError
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


def initial_state(geology, pressure_gradient: float = 10500.0,
                  datum_pressure: float = 101325.0,
                  sw: float = 0.30, sg: float = 0.0, temperature: float = 80.0,
                  name: str = "baseline") -> ReservoirState:
    """Build a baseline state from a geological model.

    Pore pressure follows a hydrostatic gradient
    ``P = datum_pressure + gradient * z`` with ``gradient`` in Pa/m
    (10,500 Pa/m is a typical brine gradient, about 0.105 bar/m).  The
    default datum is one atmosphere, so ``z = 0`` is a land surface; for a
    marine setting add the water column to ``datum_pressure``.
    Saturations are constant in the reservoir and fully brine-saturated
    outside it, since only reservoir cells carry hydrocarbons.
    """
    grid = geology.grid
    z = grid.axis(2)[None, None, :] * np.ones(grid.shape)
    pressure = datum_pressure + pressure_gradient * z

    mask = geology.reservoir_mask
    sw_field = np.where(mask, sw, 1.0)
    sg_field = np.where(mask, sg, 0.0)
    so_field = np.clip(1.0 - sw_field - sg_field, 0.0, 1.0)

    return ReservoirState(
        grid=grid, porosity=geology.porosity, ntg=geology.ntg, vsh=geology.vsh,
        pressure=pressure, sw=sw_field, so=so_field, sg=sg_field,
        temperature=temperature, reservoir_mask=mask, name=name,
        provenance=[f"hydrostatic gradient {pressure_gradient:g} Pa/m",
                    f"reservoir Sw = {sw:g}, Sg = {sg:g}"],
    )
