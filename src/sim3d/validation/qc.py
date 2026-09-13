"""Automatic model QC before simulation (spec sections 127-128).

Every check returns a PASS, WARNING or FAIL with the numbers behind it.  A
FAIL means the requested run is not physically or numerically valid and
must not proceed; a WARNING means it will proceed but the user should know
what they are getting.

The checks deliberately state values, not verdicts alone: "WARNING -
minimum wavelength sampled at 3.1 cells" is actionable, "WARNING - grid
may be coarse" is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.grid import Grid3D
from ..core.units import pa_to_psi
from ..wave.acoustic import AcousticModel
from ..wave.fdscheme import (
    cells_per_wavelength, max_stable_dt, recommend_spacing, required_ppw,
)


class Status(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass
class Check:
    """One QC statement."""

    status: Status
    message: str

    def __str__(self) -> str:
        return f"{self.status.value:7s} - {self.message}"


@dataclass
class QCResult:
    """A collection of checks with a single overall verdict."""

    checks: list[Check] = field(default_factory=list)

    def add(self, status: Status, message: str) -> None:
        self.checks.append(Check(status, message))

    @property
    def failed(self) -> bool:
        return any(c.status is Status.FAIL for c in self.checks)

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status is Status.WARNING]

    def report(self) -> str:
        return "\n".join(str(c) for c in self.checks)

    def __str__(self) -> str:
        return self.report()


def check_layer_connectivity(geology, result: QCResult | None = None,
                             minimum_cells: int = 2) -> QCResult:
    """Check a dipping reservoir is still laterally connected on this grid.

    A bed of thickness ``g`` dipping at ``theta`` drops ``dx tan(theta)``
    per lateral cell, so two adjacent columns of it overlap by only
    ``g - dx tan(theta)``.  Once that is smaller than a cell the reservoir
    stops being a layer and becomes a staircase of disconnected blocks:
    wells lose pressure communication, rates collapse against their BHP
    limits, and *the material balance still closes*, because nothing is
    conserved incorrectly - the fluid simply has nowhere to go.

    That combination is what makes it worth a check.  Every other symptom
    looks like a badly chosen rate.
    """
    result = result or QCResult()
    mask = np.asarray(geology.reservoir_mask, dtype=bool)
    if not mask.any():
        result.add(Status.WARNING, "no reservoir cells: nothing to connect")
        return result

    worst = None
    for axis in (0, 1):
        a = np.moveaxis(mask, axis, 0)
        lower, upper = a[:-1], a[1:]
        both = lower.any(axis=-1) & upper.any(axis=-1)
        if not both.any():
            continue
        shared = (lower & upper).sum(axis=-1)[both]
        broken = int((shared == 0).sum())
        pairs = int(both.sum())
        name = "xy"[axis]
        if worst is None or broken > worst[1]:
            worst = (name, broken, pairs, int(shared.min()), float(np.median(shared)))

    if worst is None:
        result.add(Status.PASS, "reservoir occupies a single column; "
                                "lateral connectivity does not apply")
        return result
    name, broken, pairs, smallest, median = worst
    if broken:
        status = Status.FAIL
        verdict = (f"reservoir is laterally disconnected along {name}: "
                   f"{broken} of {pairs} adjacent column pairs share no cell")
    elif smallest < minimum_cells:
        status = Status.WARNING
        verdict = (f"reservoir is thinly connected along {name}: the weakest "
                   f"of {pairs} adjacent column pairs shares {smallest} cell(s)")
    else:
        status = Status.PASS
        verdict = (f"reservoir is laterally connected along {name}: every one "
                   f"of {pairs} adjacent column pairs shares at least "
                   f"{smallest} cells")
    result.add(status, verdict + f" (median {median:.0f}). A dipping bed needs "
                                 f"dx*tan(dip) well under its thickness.")
    return result


def check_state(state, result: QCResult | None = None) -> QCResult:
    """Petrophysical checks on a reservoir state."""
    result = result or QCResult()
    total = state.sw + state.so + state.sg
    worst = float(np.max(np.abs(total - 1.0)))
    result.add(Status.PASS if worst < 1e-6 else Status.FAIL,
               f"Sw + So + Sg = 1 (worst residual {worst:.3e})")

    phi = state.porosity
    ok = np.all((phi > 0) & (phi < 1))
    result.add(Status.PASS if ok else Status.FAIL,
               f"porosity within (0, 1): spans [{phi.min():.4f}, {phi.max():.4f}]")

    p = state.pressure
    result.add(Status.PASS if np.all(p > 0) else Status.FAIL,
               f"pore pressure positive: spans "
               f"[{pa_to_psi(p.min()):,.0f}, {pa_to_psi(p.max()):,.0f}] psi")
    return result


def check_model(model: AcousticModel, f0: float, spatial_order: int = 8,
                dt: float | None = None, courant_safety: float = 0.9,
                tolerance: float = 0.01, fmax: float | None = None,
                result: QCResult | None = None) -> QCResult:
    """Numerical and physical checks on an acoustic earth model."""
    result = result or QCResult()
    grid = model.grid

    result.add(Status.PASS if np.all(model.vp > 0) else Status.FAIL,
               f"Vp positive: spans [{model.vmin:.1f}, {model.vmax:.1f}] m/s")
    result.add(Status.PASS if np.all(model.rho > 0) else Status.FAIL,
               f"density positive: spans "
               f"[{model.rho.min():.1f}, {model.rho.max():.1f}] kg/m3")

    # Velocity contrasts sharp enough to be worth flagging.
    jumps = [np.max(np.abs(np.diff(model.vp, axis=a)) / model.vp.min())
             for a in range(3)]
    biggest = float(max(jumps))
    result.add(Status.WARNING if biggest > 1.0 else Status.PASS,
               f"largest cell-to-cell Vp jump is {100 * biggest:.0f}% of Vmin; "
               f"contrasts above 100% are sharp for an 8th-order operator and "
               f"will scatter numerical energy")

    fmax = float(fmax if fmax is not None else 2.5 * f0)
    ppw = cells_per_wavelength(grid.spacing, model.vmin, fmax)
    needed = required_ppw(spatial_order, tolerance=tolerance, safety=courant_safety)
    if ppw >= needed:
        status = Status.PASS
    elif ppw >= 0.75 * needed:
        status = Status.WARNING
    else:
        status = Status.FAIL
    recommended = recommend_spacing(model.vmin, fmax, spatial_order, tolerance)
    result.add(status,
               f"minimum wavelength ({model.vmin / fmax:.1f} m at Vmin = "
               f"{model.vmin:.0f} m/s, Fmax = {fmax:.1f} Hz) is sampled at "
               f"{ppw:.2f} cells; order {spatial_order} needs {needed:.2f} for "
               f"{tolerance:.1%} phase error, i.e. dx <= "
               f"{recommended.max_spacing:.2f} m")

    limit = max_stable_dt(grid.spacing, model.vmax, spatial_order, safety=1.0)
    safe = max_stable_dt(grid.spacing, model.vmax, spatial_order, safety=courant_safety)
    used = float(dt) if dt is not None else safe
    if used > limit:
        result.add(Status.FAIL,
                   f"CFL criterion violated: dt = {used * 1e3:.4f} ms exceeds the "
                   f"stability limit {limit * 1e3:.4f} ms")
    elif used > safe:
        result.add(Status.WARNING,
                   f"dt = {used * 1e3:.4f} ms is stable but above the "
                   f"{courant_safety:g} safety margin ({safe * 1e3:.4f} ms)")
    else:
        result.add(Status.PASS,
                   f"CFL satisfied: dt = {used * 1e3:.4f} ms, limit "
                   f"{limit * 1e3:.4f} ms, ratio {used / limit:.3f}")
    return result


def check_geometry(acquisition, grid: Grid3D, pml_nodes: int,
                   result: QCResult | None = None,
                   target: Grid3D | None = None) -> QCResult:
    """Check that the acquisition fits inside the propagation domain and its interior.

    With a ``target`` it also checks that the survey reaches past it.  A
    carpet that stops at the target boundary leaves the edge of the anomaly
    illuminated from one side and carrying almost no fold, which is exactly
    where a 4D interpretation gets read - so it is worth saying out loud
    rather than leaving to be noticed in the image.
    """
    result = result or QCResult()
    outside = acquisition.outside(grid)
    result.add(Status.PASS if not outside else Status.FAIL,
               f"all {acquisition.n_sources + acquisition.n_receivers} source and "
               f"receiver positions inside the propagation domain"
               + (f"; {len(outside)} outside, first {outside[0]}" if outside else ""))

    interior = grid.padded(-pml_nodes) if pml_nodes else grid
    in_pml = [tuple(p) for p in (*acquisition.sources, *acquisition.receivers)
              if not interior.contains(p)]
    result.add(Status.PASS if not in_pml else Status.FAIL,
               f"no source or receiver inside the {pml_nodes}-node absorbing layer"
               + (f"; {len(in_pml)} are, first {in_pml[0]}" if in_pml else ""))

    if target is not None:
        check_target_coverage(acquisition, target, result)
    return result


#: Surface footprint beyond each target edge below which coverage is called
#: marginal.  Not a physical threshold - the physical one is the aperture,
#: which depends on depth and velocity - but a spacing-free floor that
#: catches a survey sized to the target instead of past it.
MIN_TARGET_MARGIN = 100.0


def check_target_coverage(acquisition, target: Grid3D,
                          result: QCResult | None = None) -> QCResult:
    """Report how far the survey footprint reaches beyond the target."""
    result = result or QCResult()
    points = np.vstack([np.asarray(acquisition.sources, dtype=float),
                        np.asarray(acquisition.receivers, dtype=float)])
    (tx0, tx1), (ty0, ty1), _ = target.bounds
    margins = (tx0 - points[:, 0].min(), points[:, 0].max() - tx1,
               ty0 - points[:, 1].min(), points[:, 1].max() - ty1)
    worst = float(min(margins))
    # A warning, not a failure: sim3d reserves FAIL for what cannot proceed,
    # and an under-sized survey models and migrates perfectly well - it just
    # produces an image whose edges nobody should read.
    if worst < 0.0:
        status, verdict = Status.WARNING, "stops short of the target edge"
    elif worst < MIN_TARGET_MARGIN:
        status, verdict = Status.WARNING, "barely reaches past the target"
    else:
        status, verdict = Status.PASS, "extends past the target on every side"
    result.add(status,
               f"survey {verdict}: {worst:,.0f} m of footprint beyond the "
               f"narrowest edge")
    return result


def run_qc(model: AcousticModel, f0: float, state=None, acquisition=None,
           spatial_order: int = 8, dt: float | None = None,
           pml_nodes: int = 12, **kwargs) -> QCResult:
    """Run every applicable check and return the combined result."""
    result = QCResult()
    if state is not None:
        check_state(state, result)
    check_model(model, f0, spatial_order=spatial_order, dt=dt, result=result, **kwargs)
    if acquisition is not None:
        check_geometry(acquisition, model.grid, pml_nodes, result)
    return result
