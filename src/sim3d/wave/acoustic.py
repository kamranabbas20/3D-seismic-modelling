r"""3D variable-density acoustic finite-difference solver (spec section 61).

Governing equations
-------------------
The first-order velocity-pressure system, in SI units throughout:

.. math::
    \frac{\partial \mathbf v}{\partial t} = -\frac{1}{\rho}\nabla p,
    \qquad
    \frac{\partial p}{\partial t} = -\kappa\,\nabla\cdot\mathbf v + s,
    \qquad \kappa = \rho V_p^2 .

Assumptions
-----------
* Acoustic: no shear stress, so no S waves and no mode conversion.  ``Vs``
  is still carried by the rock-physics layer and is simply unused here.
* Lossless: no intrinsic attenuation, ``Q = infinity``.
* Isotropic.
* Density varies in space, which is what makes impedance contrasts - not
  only velocity contrasts - generate reflections.
* Every outer face is absorbing.  There is **no free surface and no water
  layer**, so surface multiples and ghosts are absent from the output.

Numerics
--------
Staggered grid, ``2N``-order in space, 2nd-order leapfrog in time, CPML
boundaries.  Stability and grid sampling come from
:mod:`sim3d.wave.fdscheme`; neither is hard-coded and neither is silently
adjusted.

What this reproduces
--------------------
Direct arrivals, primary reflections, transmission, refraction and head
waves, diffraction, finite-frequency interference and tuning, geometric
spreading, focusing and defocusing by structure, and internal multiples.
What it does not reproduce: S waves, converted waves, anisotropy,
attenuation, and free-surface multiples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np

from ..compute import ComputeBackend, get_backend
from ..core.errors import ConfigError, StabilityError, ValidationError
from ..core.grid import Grid3D
from . import operators as ops
from .cpml import AxisCPML, PMLSettings, build_profile
from .fdscheme import CFLReport, check_dt, half_stencil, max_stable_dt
from .interp import PointSet
from .sources import SourceTerm


@dataclass
class AcousticModel:
    """A ``Vp``/``rho`` earth model on a propagation grid.

    This is the only thing the solver knows about the earth: geology,
    reservoir state and rock physics all reduce to these two volumes, and
    the provenance record keeps the link back to what produced them.
    """

    grid: Grid3D
    vp: np.ndarray
    rho: np.ndarray
    name: str = "model"

    def __post_init__(self) -> None:
        for label, arr in (("vp", self.vp), ("rho", self.rho)):
            if arr.shape != self.grid.shape:
                raise ConfigError(
                    f"{label} has shape {arr.shape} but the grid is {self.grid.shape}"
                )
            if not np.all(np.isfinite(arr)):
                raise ValidationError(f"{label} contains non-finite values")
            if np.any(arr <= 0):
                raise ValidationError(
                    f"{label} must be strictly positive everywhere; minimum is {arr.min():g}"
                )

    @property
    def vmin(self) -> float:
        return float(np.min(self.vp))

    @property
    def vmax(self) -> float:
        return float(np.max(self.vp))

    @property
    def impedance(self) -> np.ndarray:
        """Acoustic impedance ``rho * Vp`` in kg/m^2/s."""
        return self.rho * self.vp

    def cropped(self, grid: Grid3D) -> "AcousticModel":
        """Sample this model onto a sub-grid by nearest node.

        Used to build a propagation domain out of a larger geological
        model.  The sub-grid must be contained in this one.
        """
        idx = [
            np.clip(np.round((grid.axis(a) - self.grid.origin[a]) / self.grid.spacing[a]).astype(int),
                    0, self.grid.shape[a] - 1)
            for a in range(3)
        ]
        sl = np.ix_(*idx)
        return AcousticModel(grid, self.vp[sl].copy(), self.rho[sl].copy(), name=f"{self.name}|cropped")


@dataclass
class SolverSettings:
    """Numerical settings.  Nothing here is adjusted behind the user's back."""

    spatial_order: int = 8
    courant_safety: float = 0.90
    dt: float | None = None  #: ``None`` derives dt from the CFL limit
    pml: PMLSettings = field(default_factory=PMLSettings)
    dtype: type = np.float32
    #: Abort if ``max|p|`` exceeds this multiple of its running maximum.
    blowup_factor: float = 1.0e6

    def __post_init__(self) -> None:
        if not 0 < self.courant_safety <= 1.0:
            raise ConfigError(
                f"courant_safety must be in (0, 1]; got {self.courant_safety}"
            )


@dataclass
class ShotRecord:
    """Output of one forward simulation."""

    traces: np.ndarray            #: ``(n_receivers, nt)`` pressure
    dt: float
    receiver_positions: np.ndarray
    source_position: tuple[float, float, float] | None
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)
    snapshot_times: dict[int, float] = field(default_factory=dict)
    cfl: CFLReport | None = None

    @property
    def nt(self) -> int:
        return int(self.traces.shape[1])

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.nt) * self.dt

    @property
    def record_length(self) -> float:
        """Recording duration in seconds."""
        return self.nt * self.dt


class AcousticSolver:
    """Time-domain 3D acoustic solver on a fixed model and grid.

    The instance owns its field buffers and CPML memory, so a single solver
    can be reused across many shots without reallocating; :meth:`reset`
    clears the state between runs.
    """

    def __init__(self, model: AcousticModel, settings: SolverSettings | None = None,
                 f0: float = 20.0, backend: ComputeBackend | str | None = None):
        self.model = model
        self.settings = settings or SolverSettings()
        self.backend = get_backend(backend)
        self.f0 = float(f0)
        self.grid = model.grid
        order = self.settings.spatial_order
        dtype = self.settings.dtype

        npml = self.settings.pml.n_nodes
        if npml <= half_stencil(order):
            raise ConfigError(
                f"the absorbing layer ({npml} nodes) must be thicker than the "
                f"stencil half-width ({half_stencil(order)} nodes) so that the "
                f"nodes the operator cannot reach stay inside the layer"
            )

        # -- time step ---------------------------------------------------
        limit = max_stable_dt(self.grid.spacing, model.vmax, order,
                              safety=self.settings.courant_safety)
        self.dt = float(self.settings.dt) if self.settings.dt is not None else limit
        self.cfl = check_dt(self.dt, self.grid.spacing, model.vmax, order,
                            safety=self.settings.courant_safety)

        # -- medium coefficients ----------------------------------------
        nx, ny, nz = self.grid.shape
        self.kappa = (model.rho * model.vp**2).astype(dtype)
        buoyancy = (1.0 / model.rho).astype(dtype)
        self.bx = ops.average_to_half(buoyancy, 0)
        self.by = ops.average_to_half(buoyancy, 1)
        self.bz = ops.average_to_half(buoyancy, 2)
        cell_volume = self.grid.dx * self.grid.dy * self.grid.dz
        self.dt_kappa_over_cell = (self.dt * self.kappa / cell_volume).astype(dtype)
        # Pre-multiplied coefficients keep the time loop free of whole-grid
        # temporaries: every operation below is in place on a scratch buffer.
        self._dt_kappa = (self.dt * self.kappa).astype(dtype)
        self._dt_b = [(self.dt * b).astype(dtype) for b in (self.bx, self.by, self.bz)]

        # -- fields ------------------------------------------------------
        self.p = np.zeros((nx, ny, nz), dtype=dtype)
        self.vx = np.zeros((nx - 1, ny, nz), dtype=dtype)
        self.vy = np.zeros((nx, ny - 1, nz), dtype=dtype)
        self.vz = np.zeros((nx, ny, nz - 1), dtype=dtype)
        self._dp = [np.zeros_like(self.vx), np.zeros_like(self.vy), np.zeros_like(self.vz)]
        self._div = np.zeros_like(self.p)
        self._div_term = np.zeros_like(self.p)

        # -- CPML --------------------------------------------------------
        self._pml_grad = []   # d p / d x_i, on the half grid
        self._pml_div = []    # d v_i / d x_i, on the nodes
        for axis in range(3):
            n_nodes = self.grid.shape[axis]
            spacing = self.grid.spacing[axis]
            low, high = self.settings.pml.active[2 * axis], self.settings.pml.active[2 * axis + 1]
            half_pos = np.arange(n_nodes - 1, dtype=float) + 0.5
            node_pos = np.arange(n_nodes, dtype=float)
            prof_half = build_profile(half_pos, n_nodes, spacing, self.dt, model.vmax,
                                      self.f0, self.settings.pml, low, high, dtype)
            prof_node = build_profile(node_pos, n_nodes, spacing, self.dt, model.vmax,
                                      self.f0, self.settings.pml, low, high, dtype)
            self._pml_grad.append(
                AxisCPML(self._dp[axis].shape, axis, prof_half, npml, low, high, dtype)
            )
            self._pml_div.append(
                AxisCPML(self.p.shape, axis, prof_node, npml, low, high, dtype)
            )

    # -- state -----------------------------------------------------------
    def reset(self) -> None:
        """Zero the wavefield and every CPML memory variable."""
        for arr in (self.p, self.vx, self.vy, self.vz):
            arr.fill(0.0)
        for pml in (*self._pml_grad, *self._pml_div):
            pml.reset()

    @property
    def state_bytes(self) -> int:
        """Bytes held by fields and boundary memory (excludes stored snapshots)."""
        fields = sum(a.nbytes for a in (self.p, self.vx, self.vy, self.vz, self.kappa,
                                        self.bx, self.by, self.bz, self.dt_kappa_over_cell,
                                        self._dt_kappa, *self._dt_b,
                                        self._div, self._div_term, *self._dp))
        return fields + sum(pml.nbytes for pml in (*self._pml_grad, *self._pml_div))

    # -- one leapfrog step -----------------------------------------------
    def step(self, source: SourceTerm, istep: int) -> None:
        """Advance the wavefield by one time step."""
        # velocities at t + dt/2 from the pressure gradient at t
        for axis, v in enumerate((self.vx, self.vy, self.vz)):
            d = self.backend.forward_diff(self.p, axis, self.grid.spacing[axis],
                                          self.settings.spatial_order, self._dp[axis])
            self._pml_grad[axis].apply(d)
            d *= self._dt_b[axis]
            v -= d

        # pressure at t + dt from the velocity divergence at t + dt/2
        self._div.fill(0.0)
        for axis, v in enumerate((self.vx, self.vy, self.vz)):
            d = self.backend.backward_diff(v, axis, self.grid.spacing[axis],
                                           self.settings.spatial_order,
                                           self.grid.shape[axis], self._div_term)
            self._pml_div[axis].apply(d)
            self._div += d
        self._div *= self._dt_kappa
        self.p -= self._div

        source.inject(self.p, istep, self.dt_kappa_over_cell)

    # -- full runs ---------------------------------------------------------
    def run(self, source: SourceTerm, nt: int,
            receivers: PointSet | np.ndarray | None = None,
            snapshot_steps: Iterable[int] = (),
            on_step: Callable[[int, np.ndarray], None] | None = None,
            reset: bool = True) -> ShotRecord:
        """Propagate for ``nt`` steps, recording receivers and snapshots.

        Parameters
        ----------
        source:
            Anything implementing :class:`~sim3d.wave.sources.SourceTerm`.
        nt:
            Number of time steps; the recording length is ``nt * dt``.
        receivers:
            A :class:`~sim3d.wave.interp.PointSet` or an ``(n, 3)`` array of
            receiver coordinates in metres.
        snapshot_steps:
            Time-step indices at which to keep a copy of the pressure field.
        on_step:
            Called as ``on_step(istep, p)`` after every step, before the
            snapshot copy.  Used by RTM to stream the source wavefield out.
        reset:
            Zero the state first (the default; pass ``False`` to continue a run).
        """
        if nt < 1:
            raise ConfigError(f"nt must be >= 1, got {nt}")
        if reset:
            self.reset()

        rec = receivers
        if rec is not None and not isinstance(rec, PointSet):
            rec = PointSet(self.grid, rec, "receivers")
        n_rec = 0 if rec is None else len(rec)
        traces = np.zeros((n_rec, nt), dtype=self.settings.dtype)

        want = set(int(s) for s in snapshot_steps)
        snapshots: dict[int, np.ndarray] = {}
        running_max = 0.0

        for istep in range(nt):
            self.step(source, istep)
            if rec is not None:
                traces[:, istep] = rec.gather(self.p)
            if on_step is not None:
                on_step(istep, self.p)
            if istep in want:
                snapshots[istep] = self.p.copy()
            if istep % 64 == 0:
                running_max = self._check_finite(istep, running_max)

        return ShotRecord(
            traces=traces, dt=self.dt,
            receiver_positions=np.zeros((0, 3)) if rec is None else rec.points,
            source_position=getattr(source, "position", None),
            snapshots=snapshots,
            snapshot_times={s: s * self.dt for s in snapshots},
            cfl=self.cfl,
        )

    def _check_finite(self, istep: int, running_max: float) -> float:
        peak = float(np.max(np.abs(self.p)))
        if not np.isfinite(peak):
            raise StabilityError(
                f"the wavefield became non-finite at step {istep} "
                f"(t = {istep * self.dt * 1e3:.1f} ms). {self.cfl.describe()}"
            )
        if running_max > 0 and peak > self.settings.blowup_factor * running_max:
            raise StabilityError(
                f"the wavefield grew by more than {self.settings.blowup_factor:g}x "
                f"by step {istep} (t = {istep * self.dt * 1e3:.1f} ms), which "
                f"indicates numerical instability. {self.cfl.describe()}"
            )
        return max(running_max, peak)

    # -- reporting ---------------------------------------------------------
    def describe(self) -> str:
        """Multi-line summary of the numerical setup, for the QC page and logs."""
        pml_m = self.settings.pml.thickness_m(self.grid.spacing)
        return "\n".join(
            [
                f"Acoustic solver on {self.grid.describe()}",
                f"  Vp range      {self.model.vmin:g} - {self.model.vmax:g} m/s",
                f"  backend       {self.backend.name}",
                f"  spatial order {self.settings.spatial_order}, "
                f"dt = {self.dt * 1e3:.4f} ms",
                f"  {self.cfl.describe()}",
                f"  CPML          {self.settings.pml.n_nodes} nodes "
                f"({pml_m[0]:g}, {pml_m[1]:g}, {pml_m[2]:g}) m, "
                f"R0 = {self.settings.pml.r0:g}",
                f"  state memory  {self.state_bytes / 2**20:.1f} MiB",
                "  physics: acoustic, lossless, isotropic, variable density; "
                "no S waves, no attenuation, no free surface",
            ]
        )


def common_dt(models: Sequence[AcousticModel], spatial_order: int = 8,
              safety: float = 0.90) -> float:
    """One stable time step valid for every model in an experiment.

    Baseline, pressure-only, saturation-only and combined earth models have
    different maximum velocities, so each would pick a different CFL time
    step if left alone - and shot records written on different time axes
    cannot be differenced.  Every 4D comparison must therefore pin ``dt``
    to the value returned here, set by the fastest model of the set.
    """
    if not models:
        raise ConfigError("common_dt needs at least one model")
    spacings = {tuple(m.grid.spacing) for m in models}
    if len(spacings) > 1:
        raise ConfigError(
            f"models must share a grid spacing to share a time step; got {spacings}"
        )
    vmax = max(m.vmax for m in models)
    return max_stable_dt(models[0].grid.spacing, vmax, spatial_order, safety=safety)


def steps_for_duration(duration: float, dt: float) -> int:
    """Number of time steps covering ``duration`` seconds (rounded up)."""
    if duration <= 0:
        raise ConfigError(f"recording duration must be positive, got {duration}")
    return int(np.ceil(duration / dt))
