r"""Two-phase flow simulation (requirement sections 7, 9, 10, 11, 12).

What this is
------------
A slightly-compressible, two-phase (water/oil), three-dimensional IMPES
simulator on the reservoir cells of the geological model:

.. math::
    \phi c_t \frac{\partial p}{\partial t}
        = \nabla\!\cdot\!\big(\lambda_t K \nabla \Phi\big) + q_t,
    \qquad
    \phi \frac{\partial S_w}{\partial t}
        + \nabla\!\cdot\!\big(f_w \mathbf{u}_t\big) = q_w

with :math:`\Phi = p - \rho g z` the phase potential, Corey relative
permeabilities, upstream mobility weighting, harmonic transmissibilities,
fault transmissibility multipliers on the faces they cut, and Peaceman well
indices.  Pressure is solved implicitly each step; saturation is advanced
explicitly under a CFL limit on the saturation change.

What this is not
----------------
It is not a black-oil simulator.  There is no dissolved gas, no free gas
phase, no PVT table, no compositional behaviour, no capillary pressure, no
thermal effect and no geomechanics.  Oil and water are slightly
compressible fluids with constant viscosity and constant formation volume
factors.

That is a deliberate floor rather than an oversight.  The questions this
platform exists to answer - where the flood front is, how far the pressure
halo reaches, what the 4D response looks like and whether pressure can be
told from saturation - are all answered by exactly these two equations.
Adding a gas phase would change the rock-physics response substantially and
is the first extension worth making; everything else on that list would
change the answers hardly at all while making them much harder to trust.

Where free gas matters, the mechanistic generator in
:mod:`sim3d.reservoir.mechanistic` remains the instrument: it imposes a gas
saturation directly, which is the honest way to study a response this
solver cannot produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, cg, spsolve

from ..core.errors import ConfigError, ValidationError
from ..core.units import DAY, STB, pa_to_psi, si_to_stb_per_day
from ..wells.completion import resolve_completions
from ..wells.controls import WELLBORE_RADIUS, ControlMode, WellControl
from .relperm import CoreyRelativePermeability
from .state import ReservoirState

GRAVITY = 9.81


def _scatter_add(target: np.ndarray, indices: np.ndarray, values: np.ndarray) -> None:
    """``target[indices] += values`` with repeats handled.

    ``np.add.at`` is the obvious way to write this and is roughly an order of
    magnitude slower than ``bincount``; on a face list this runs several
    times per timestep, so it is worth the indirection.
    """
    if indices.size:
        target += np.bincount(indices, weights=values, minlength=target.size)


@dataclass
class FlowSettings:
    """Numerical and fluid settings for the flow model."""

    relperm: CoreyRelativePermeability = field(
        default_factory=CoreyRelativePermeability)
    #: Total compressibility (rock plus fluid), 1/Pa.  4e-10 is about
    #: 3e-6 /psi, typical for an undersaturated oil reservoir.
    total_compressibility: float = 4.0e-10
    water_density: float = 1030.0     #: kg/m^3
    oil_density: float = 800.0        #: kg/m^3
    #: Vertical-to-horizontal permeability ratio.
    kv_over_kh: float = 0.1
    gravity: bool = True
    #: Largest saturation change allowed in one step, which sets the timestep.
    max_saturation_change: float = 0.05
    max_timestep_days: float = 30.0
    min_timestep_days: float = 1.0e-3
    #: Skin at every well unless overridden.
    skin: float = 0.0

    def __post_init__(self) -> None:
        if self.total_compressibility <= 0:
            raise ConfigError("total compressibility must be positive")
        if not 0 < self.max_saturation_change <= 1:
            raise ConfigError("max_saturation_change must lie in (0, 1]")


@dataclass
class _Connection:
    """One well's contribution to a timestep, as the solver needs it."""

    cells: np.ndarray
    weights: np.ndarray      #: well index times mobility, m^3/s per Pa
    q_water: np.ndarray      #: m^3/s, positive into the reservoir
    q_oil: np.ndarray
    mode: "ControlMode"
    bhp: float

    def __iter__(self):
        """Legacy unpacking as ``(cells, weights, q_water, q_oil)``."""
        return iter((self.cells, self.weights, self.q_water, self.q_oil))


@dataclass
class WellHistory:
    """Per-well time series (section 11).  SI internally, oilfield on display."""

    name: str
    role: str
    days: list[float] = field(default_factory=list)
    oil_rate: list[float] = field(default_factory=list)      #: m^3/s, positive out
    water_rate: list[float] = field(default_factory=list)
    bhp: list[float] = field(default_factory=list)           #: Pa
    control: list[str] = field(default_factory=list)

    def arrays(self) -> dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=float) for k, v in
                (("days", self.days), ("oil_rate", self.oil_rate),
                 ("water_rate", self.water_rate), ("bhp", self.bhp))}

    @property
    def liquid_rate(self) -> np.ndarray:
        a = self.arrays()
        return a["oil_rate"] + a["water_rate"]

    @property
    def water_cut(self) -> np.ndarray:
        total = self.liquid_rate
        water = np.asarray(self.water_rate, dtype=float)
        return np.divide(water, total, out=np.zeros_like(water), where=np.abs(total) > 0)

    def cumulative(self, which: str) -> np.ndarray:
        """Cumulative produced or injected volume in m^3, by trapezoid in time."""
        a = self.arrays()
        rate = np.abs(a[f"{which}_rate"])
        seconds = a["days"] * DAY
        if rate.size < 2:
            return np.zeros_like(rate)
        return np.concatenate([[0.0], np.cumsum(
            0.5 * (rate[1:] + rate[:-1]) * np.diff(seconds))])

    def summary(self) -> str:
        a = self.arrays()
        if a["days"].size == 0:
            return f"{self.name}: never active"
        oil = self.cumulative("oil")[-1] / STB
        water = self.cumulative("water")[-1] / STB
        label = "injected" if self.role == "injector" else "produced"
        return (f"{self.name:5s} ({self.role:8s}) "
                f"final BHP {pa_to_psi(a['bhp'][-1]):7,.0f} psi, "
                f"final liquid {si_to_stb_per_day(abs(self.liquid_rate[-1])):8,.0f} STB/day, "
                f"water cut {100 * self.water_cut[-1]:5.1f}%, "
                f"cumulative {label} oil {oil:12,.0f} STB, water {water:12,.0f} STB")


@dataclass
class FlowResult:
    """Reservoir states through time, plus the well histories."""

    days: list[float]
    pressure: list[np.ndarray]
    water_saturation: list[np.ndarray]
    wells: dict[str, WellHistory]
    settings: FlowSettings
    material_balance_error: float
    n_timesteps: int
    notes: list[str] = field(default_factory=list)

    def at(self, day: float) -> tuple[np.ndarray, np.ndarray]:
        """Pressure and water saturation at the reported time nearest ``day``."""
        index = int(np.argmin(np.abs(np.asarray(self.days) - day)))
        return self.pressure[index], self.water_saturation[index]

    def state_at(self, day: float, baseline: ReservoirState) -> ReservoirState:
        """A :class:`ReservoirState` at ``day``, carrying the simulated fields.

        Only reservoir cells are overwritten.  The flow model solves nothing
        outside them, so the overburden keeps its baseline hydrostatic
        pressure - which the rock physics needs, since effective stress
        there is what sets the overburden velocities.

        Saturation closure is maintained by moving oil only: the flow model
        has no gas phase, so any gas in the baseline stays exactly where it
        was rather than being quietly redistributed.
        """
        pressure, sw = self.at(day)
        state = baseline.copy(name=f"{baseline.name}@day{day:g}")
        mask = np.asarray(baseline.reservoir_mask, dtype=bool)
        state.pressure = np.where(mask, pressure, baseline.pressure)
        state.sw = np.where(mask, sw, baseline.sw)
        state.so = np.clip(1.0 - state.sw - state.sg, 0.0, 1.0)
        state.sw = 1.0 - state.so - state.sg
        state.provenance = [*baseline.provenance,
                            f"two-phase IMPES flow simulation to day {day:g}"]
        state.validate()
        return state

    def summary(self) -> str:
        lines = [
            f"Flow simulation: {self.days[0]:g} to {self.days[-1]:g} days in "
            f"{self.n_timesteps:,} timesteps, {len(self.days)} reports",
            f"  material balance closes to {self.material_balance_error:.3e} "
            f"of the injected/produced volume",
            f"  {self.settings.relperm.describe()}",
        ]
        lines += [f"  {h.summary()}" for h in self.wells.values()]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


class FlowSimulator:
    """IMPES solver on the reservoir cells of a geological model."""

    def __init__(self, geology, wells, completions: dict, controls: dict,
                 initial_pressure: np.ndarray, initial_sw: np.ndarray,
                 settings: FlowSettings | None = None):
        self.geology = geology
        self.grid = geology.grid
        self.settings = settings or FlowSettings()
        self.wells = list(wells)
        self.completions = completions
        self.controls = controls

        self.active = np.asarray(geology.reservoir_mask, dtype=bool)
        if not self.active.any():
            raise ValidationError(
                "the geological model has no reservoir cells, so there is nothing "
                "to flow through")
        self.index = -np.ones(self.grid.shape, dtype=np.int64)
        self.index[self.active] = np.arange(int(self.active.sum()))
        self.n = int(self.active.sum())

        cell = self.grid.dx * self.grid.dy * self.grid.dz
        self.pore_volume = (geology.porosity * geology.ntg * cell)[self.active]
        if np.any(self.pore_volume <= 0):
            raise ValidationError("a reservoir cell has zero pore volume")

        self._build_transmissibilities()
        self._build_well_connections()

        self.p = np.asarray(initial_pressure, dtype=float)[self.active].copy()
        self.sw = np.asarray(initial_sw, dtype=float)[self.active].copy()
        swc = self.settings.relperm.swc
        if np.any(self.sw < swc - 1e-9):
            self.sw = np.maximum(self.sw, swc)
        self.z = (self.grid.axis(2)[None, None, :] * np.ones(self.grid.shape))[self.active]

    # ---------------------------------------------------------------- setup
    def _build_transmissibilities(self) -> None:
        r"""Harmonic face transmissibilities, with fault multipliers.

        :math:`T = A\,\bar k / d` with :math:`\bar k` the harmonic mean of
        the two cells' effective permeability - harmonic because the two
        half-cells are in series, and using an arithmetic mean would let a
        single good cell short-circuit a barrier.
        """
        grid = self.grid
        k = self.geology.permeability * self.geology.ntg
        areas = (grid.dy * grid.dz, grid.dx * grid.dz, grid.dx * grid.dy)
        distances = grid.spacing
        x, y, z = np.meshgrid(grid.axis(0), grid.axis(1), grid.axis(2), indexing="ij")

        rows, cols, values = [], [], []
        self._faces = []
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, grid.shape[axis] - 1)
            hi[axis] = slice(1, grid.shape[axis])
            lo, hi = tuple(lo), tuple(hi)

            both = self.active[lo] & self.active[hi]
            if not both.any():
                continue
            k_lo, k_hi = k[lo][both], k[hi][both]
            harmonic = np.where(k_lo + k_hi > 0,
                                2.0 * k_lo * k_hi / np.where(k_lo + k_hi > 0,
                                                             k_lo + k_hi, 1.0), 0.0)
            if axis == 2:
                harmonic = harmonic * self.settings.kv_over_kh
            trans = areas[axis] * harmonic / distances[axis]

            # A fault between the two cells throttles the connection.
            mid = ((x[lo][both] + x[hi][both]) / 2,
                   (y[lo][both] + y[hi][both]) / 2,
                   (z[lo][both] + z[hi][both]) / 2)
            for fault in self.geology.faults:
                side_lo = fault.hanging_wall_fraction(
                    x[lo][both], y[lo][both], z[lo][both]) > 0.5
                side_hi = fault.hanging_wall_fraction(
                    x[hi][both], y[hi][both], z[hi][both]) > 0.5
                crosses = side_lo != side_hi
                trans = np.where(crosses, trans * fault.transmissibility, trans)

            rows.append(self.index[lo][both])
            cols.append(self.index[hi][both])
            values.append(trans)
            self._faces.append((self.index[lo][both], self.index[hi][both], trans,
                                mid[2]))

        self.face_lo = np.concatenate([r for r in rows]) if rows else np.zeros(0, int)
        self.face_hi = np.concatenate([c for c in cols]) if cols else np.zeros(0, int)
        self.face_trans = (np.concatenate([v for v in values]) if values
                           else np.zeros(0, float))
        self.face_dz = (np.concatenate([f[3] for f in self._faces]) if self._faces
                        else np.zeros(0, float))

    def _build_well_connections(self) -> None:
        r"""Peaceman well indices for every completed cell.

        .. math:: WI = \frac{2\pi k h}{\ln(r_e/r_w) + s},
                  \qquad r_e = 0.28\,\frac{\sqrt{\Delta x^2 + \Delta y^2}}{2}

        The equivalent radius is what connects a well - a line source - to a
        cell-averaged pressure; using the cell pressure directly would make
        the answer depend on the grid spacing.
        """
        grid = self.grid
        r_e = 0.28 * np.hypot(grid.dx, grid.dy) / 2.0
        self.connections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for well in self.wells:
            completions = self.completions.get(well.name)
            if not completions:
                continue
            ix = int(np.argmin(np.abs(grid.axis(0) - well.x)))
            iy = int(np.argmin(np.abs(grid.axis(1) - well.y)))
            zs = grid.axis(2)
            cells, indices = [], []
            for top, base, _ in resolve_completions(well, self.geology, completions):
                # Cell boundaries, not centres: see completion_mask.
                inside = (zs >= top) & (zs <= base)
                for iz in np.flatnonzero(inside):
                    if not self.active[ix, iy, iz]:
                        continue
                    k = float(self.geology.permeability[ix, iy, iz]
                              * self.geology.ntg[ix, iy, iz])
                    wi = (2.0 * np.pi * k * grid.dz
                          / (np.log(r_e / WELLBORE_RADIUS) + self.settings.skin))
                    cells.append(int(self.index[ix, iy, iz]))
                    indices.append(wi)
            if cells:
                self.connections[well.name] = (np.array(cells), np.array(indices))

    # ------------------------------------------------------------------ run
    def run(self, duration_days: float, report_every_days: float = 30.0,
            progress=None) -> FlowResult:
        """Advance to ``duration_days``, reporting every ``report_every_days``."""
        if duration_days <= 0:
            raise ConfigError(f"duration must be positive, got {duration_days}")
        settings = self.settings
        relperm = settings.relperm

        reports = list(np.arange(0.0, duration_days + 1e-9, report_every_days))
        if reports[-1] < duration_days - 1e-9:
            reports.append(float(duration_days))
        histories = {w.name: WellHistory(w.name, w.role) for w in self.wells}

        days = [0.0]
        pressures = [self._expand(self.p)]
        saturations = [self._expand(self.sw)]
        # Record the rates the wells start on. Without this the series begins
        # at the end of the first timestep and every cumulative volume is
        # short by that step's production.
        self._well_rates(0.0, histories, record=True, at_day=0.0)
        injected = produced = 0.0
        initial_mass = float(np.sum(self.pore_volume * settings.total_compressibility
                                    * self.p))
        step = 0
        day = 0.0
        next_report = 1

        while day < duration_days - 1e-9 and step < 2_000_000:
            dt_days = min(settings.max_timestep_days,
                          reports[next_report] - day if next_report < len(reports)
                          else settings.max_timestep_days)
            dt_days = max(dt_days, settings.min_timestep_days)
            dt = dt_days * DAY

            rates = self._well_rates(day, histories, record=False)
            dt, dt_days = self._limit_timestep(dt, dt_days, rates)
            modes = self._pinned_modes(rates)
            self._solve_pressure(dt, rates)
            # Re-evaluate on the new pressure - that is the rate actually
            # delivered - but hold the control decision fixed, so the rate
            # applied here is exactly the one the solve honoured.
            rates = self._well_rates(day, histories, record=True,
                                     at_day=day + dt_days, modes=modes)
            self._advance_saturation(dt, rates)

            for connection in rates.values():
                total = float(np.sum(connection.q_water)
                              + np.sum(connection.q_oil)) * dt
                if total > 0:
                    injected += total
                else:
                    produced -= total

            day += dt_days
            step += 1
            if next_report < len(reports) and day >= reports[next_report] - 1e-9:
                days.append(round(day, 9))
                pressures.append(self._expand(self.p))
                saturations.append(self._expand(self.sw))
                next_report += 1
                if progress is not None:
                    progress(day, duration_days)

        final_mass = float(np.sum(self.pore_volume * settings.total_compressibility
                                  * self.p))
        throughput = max(injected + produced, 1e-12)
        error = abs((final_mass - initial_mass) - (injected - produced)) / throughput

        notes = []
        if not settings.gravity:
            notes.append("gravity disabled: the flood will not segregate vertically")
        return FlowResult(days=days, pressure=pressures, water_saturation=saturations,
                          wells=histories, settings=settings,
                          material_balance_error=error, n_timesteps=step, notes=notes)

    # ------------------------------------------------------------- internals
    def _expand(self, values: np.ndarray) -> np.ndarray:
        """Scatter an active-cell vector back onto the full grid."""
        out = np.zeros(self.grid.shape)
        out[self.active] = values
        return out

    def _potential_terms(self):
        """Gravity head across each face, per phase, in Pa."""
        if not self.settings.gravity or self.face_trans.size == 0:
            zero = np.zeros(self.face_trans.shape)
            return zero, zero
        dz = self.z[self.face_hi] - self.z[self.face_lo]
        return (self.settings.water_density * GRAVITY * dz,
                self.settings.oil_density * GRAVITY * dz)

    def _upstream(self, potential_difference):
        """Index of the upstream cell of each face, by potential."""
        return np.where(potential_difference > 0, self.face_hi, self.face_lo)

    def _well_rates(self, day: float, histories, record: bool, at_day=None,
                    modes: dict | None = None):
        """Per-connection water and oil rates, honouring mode and BHP limits.

        ``modes`` pins the control decision made before the pressure solve,
        as ``{well: (mode, bhp)}``.  Both halves matter.  Pinning the mode
        alone is worse than not pinning at all: a well that switched to
        pressure control because its rate target implied an illegal drawdown
        would come back through the pressure branch and read
        ``control.target`` - which for that well is a *rate* - as its
        bottom-hole pressure.
        """
        relperm = self.settings.relperm
        out = {}
        for well in self.wells:
            control: WellControl | None = self.controls.get(well.name)
            connection = self.connections.get(well.name)
            if control is None or connection is None or not control.active(day):
                if record:
                    self._record(histories[well.name], at_day if at_day is not None
                                 else day, 0.0, 0.0, float("nan"), "shut")
                continue
            cells, wi = connection
            injector = well.role == "injector"
            lam_w, lam_o = relperm.mobilities(self.sw[cells])
            lam_t = lam_w + lam_o
            # An injector pushes water into the formation at the injected
            # fluid's mobility, not at the in-situ mixture's.
            mobility = (np.full_like(lam_t, relperm.krw_max / relperm.water_viscosity)
                        if injector else lam_t)
            weights = wi * mobility
            total_weight = float(weights.sum())
            if total_weight <= 0:
                if record:
                    self._record(histories[well.name],
                                 at_day if at_day is not None else day,
                                 0.0, 0.0, float("nan"), "zero mobility")
                continue

            pinned = modes.get(well.name) if modes else None
            mode = pinned[0] if pinned else control.mode
            if mode is ControlMode.BHP:
                bhp = pinned[1] if pinned else control.target
                q = weights * (bhp - self.p[cells])
            else:
                target = control.target * (1.0 if injector else -1.0)
                q = target * weights / total_weight
                bhp = float(np.average(self.p[cells] + q / np.maximum(weights, 1e-300),
                                       weights=weights))
                if control.bhp_limit is not None and pinned is None:
                    violated = (bhp > control.bhp_limit if injector
                                else bhp < control.bhp_limit)
                    if violated:
                        bhp = control.bhp_limit
                        q = weights * (bhp - self.p[cells])
                        mode = ControlMode.BHP
            fw_cell = relperm.fractional_flow(self.sw[cells])
            if injector:
                q_water = np.where(q >= 0.0, q, q * fw_cell)
            else:
                q_water = q * fw_cell
            q_oil = q - q_water
            out[well.name] = _Connection(cells=cells, weights=weights,
                                         q_water=q_water, q_oil=q_oil,
                                         mode=mode, bhp=bhp)
            if record:
                self._record(histories[well.name],
                             at_day if at_day is not None else day,
                             float(np.sum(q_oil)), float(np.sum(q_water)), bhp,
                             mode.value)
        return out

    @staticmethod
    def _pinned_modes(rates) -> dict:
        """The control decision each well arrived at - mode *and* pressure."""
        return {name: (connection.mode, connection.bhp)
                for name, connection in rates.items()}

    @staticmethod
    def _record(history: WellHistory, day, oil, water, bhp, control) -> None:
        history.days.append(float(day))
        history.oil_rate.append(-float(oil))     # positive out of the reservoir
        history.water_rate.append(-float(water))
        history.bhp.append(float(bhp))
        history.control.append(control)

    def _solve_pressure(self, dt: float, rates) -> None:
        """Implicit pressure solve for one timestep."""
        relperm = self.settings.relperm
        n = self.n
        storage = self.pore_volume * self.settings.total_compressibility / dt

        lam_w, lam_o = relperm.mobilities(self.sw)
        head_w, head_o = self._potential_terms()
        # Upstream weighting on the previous pressure field.
        dphi = (self.p[self.face_hi] - self.p[self.face_lo])
        up = self._upstream(dphi)
        trans_t = self.face_trans * (lam_w[up] + lam_o[up])
        gravity = self.face_trans * (lam_w[up] * head_w + lam_o[up] * head_o)

        diagonal = storage.copy()
        rhs = storage * self.p
        _scatter_add(diagonal, self.face_lo, trans_t)
        _scatter_add(diagonal, self.face_hi, trans_t)
        # Cell ``lo``'s equation carries -T.lambda.rho.g.(z_hi - z_lo) and
        # cell ``hi``'s the opposite: the gravity head drives flow *down*, so
        # getting this sign backwards inverts the hydrostatic gradient and
        # makes a waterflood override instead of underrun.
        _scatter_add(rhs, self.face_lo, -gravity)
        _scatter_add(rhs, self.face_hi, gravity)

        # A well on rate control contributes a known source term. A well on
        # bottom-hole pressure does not: its rate depends on the pressure
        # being solved for, so it enters the matrix implicitly. Treating it
        # explicitly - using last step's pressure - makes the rate the solve
        # honours differ from the rate the saturation update then applies,
        # and material balance stops closing.
        for connection in rates.values():
            if connection.mode is ControlMode.BHP:
                _scatter_add(diagonal, connection.cells, connection.weights)
                _scatter_add(rhs, connection.cells,
                             connection.weights * connection.bhp)
            else:
                _scatter_add(rhs, connection.cells,
                             connection.q_water + connection.q_oil)

        rows = np.concatenate([np.arange(n), self.face_lo, self.face_hi])
        cols = np.concatenate([np.arange(n), self.face_hi, self.face_lo])
        data = np.concatenate([diagonal, -trans_t, -trans_t])
        matrix = csr_matrix((data, (rows, cols)), shape=(n, n))

        # The matrix is symmetric positive definite and changes only through
        # the mobilities, so conjugate gradients warm-started from the last
        # pressure converges in a handful of iterations. Re-factorising with a
        # direct solver every step costs far more than the solve itself.
        jacobi = LinearOperator((n, n), matvec=lambda v: v / diagonal, dtype=float)
        solution, info = cg(matrix, rhs, x0=self.p, rtol=1e-10, atol=0.0,
                            maxiter=1000, M=jacobi)
        if info != 0:
            solution = spsolve(matrix, rhs)   # fall back rather than drift
        self.p = solution
        if not np.all(np.isfinite(self.p)):
            raise ValidationError(
                "the pressure solve produced non-finite values; the reservoir may "
                "be disconnected, or a transmissibility may be zero everywhere")

    def _face_water_flux(self):
        """Water volumetric flux on each face, positive from lo to hi."""
        relperm = self.settings.relperm
        lam_w, lam_o = relperm.mobilities(self.sw)
        head_w, _ = self._potential_terms()
        dphi_w = (self.p[self.face_hi] - self.p[self.face_lo]) - head_w
        up = np.where(dphi_w > 0, self.face_hi, self.face_lo)
        return -self.face_trans * lam_w[up] * dphi_w

    def _net_water(self, flux, rates) -> np.ndarray:
        """Net water volumetric rate into every cell, m^3/s."""
        net = np.zeros(self.n)
        _scatter_add(net, self.face_lo, -flux)
        _scatter_add(net, self.face_hi, flux)
        for connection in rates.values():
            _scatter_add(net, connection.cells, connection.q_water)
        return net

    def _advance_saturation(self, dt: float, rates) -> None:
        """Explicit upstream saturation update."""
        flux = self._face_water_flux()
        net = self._net_water(flux, rates)
        self.sw = self.sw + dt * net / self.pore_volume
        relperm = self.settings.relperm
        self.sw = np.clip(self.sw, relperm.swc, 1.0 - relperm.sor)

    def _limit_timestep(self, dt: float, dt_days: float, rates):
        """Shrink the step so no cell's saturation moves more than allowed."""
        net = self._net_water(self._face_water_flux(), rates)
        change = np.abs(net) / self.pore_volume
        peak = float(change.max()) if change.size else 0.0
        if peak * dt > self.settings.max_saturation_change:
            dt = self.settings.max_saturation_change / peak
            dt_days = max(dt / DAY, self.settings.min_timestep_days)
            dt = dt_days * DAY
        return dt, dt_days
