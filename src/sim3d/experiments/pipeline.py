"""The end-to-end pipeline, driven entirely by an :class:`ExperimentConfig`.

Spec sections 118-119: the scientific engine must work without any GUI, and
no scientific logic may live in a UI callback.  Everything the CLI and any
future Streamlit front end do is call into this module.

Stages, in dependency order::

    geology -> reservoir -> rockphysics -> acquisition -> qc/plan
            -> preview | simulate -> migrate -> decompose

Each stage is idempotent and caches its result on the pipeline instance, so
re-running a later stage does not rebuild the earlier ones - the dependency
awareness of spec sections 109-110, at the granularity that actually
matters here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..acquisition.geometry import Acquisition, OBNGeometry, sampling_report
from ..core.config import ExperimentConfig
from ..core.errors import ConfigError
from ..core.grid import DomainSet
from ..core.units import pa_to_psi, psi_to_pa, stb_per_day_to_si
from ..core.planning import (
    CostClass, ResourceBudget, check_budget, estimate_experiment, measure_throughput,
)
from ..fourd.decomposition import decompose
from ..fourd.metrics import nrms
from ..fourd.noise import NoiseModel, add_survey_noise
from ..fourd.timeshift import (
    align, estimated_time_shift, resample_shift, true_time_shift,
)
from ..fourd.scenarios import (
    SCENARIO_NAMES, build_earth_models, build_states, build_states_from_flow,
)
from ..geology.bodies import apply_bodies, build_bodies
from ..geology.builder import (
    GeologyModel, build_geology, override_layers,
)
from ..geology.templates import template
from ..imaging.rtm import RTMSettings, RTMResult, migrate_survey
from ..processing.filters import direct_wave_mute, offset_mute
from ..processing.preview import convolution_preview, time_from_depth
from ..processing.sim2seis import sim2seis_volume
from ..processing.sparse import resolve_locations, sparse_synthetic
from ..reservoir.mechanistic import (
    GasBreakout, PressureHalo, ReservoirScenario, SaturationFront,
)
from ..reservoir.flow import FlowSettings, FlowSimulator
from ..reservoir.relperm import CoreyRelativePermeability
from ..reservoir.state import initial_state
from ..rockphysics.fluids import bubble_point
from ..rockphysics.model import RockPhysicsConfig
from ..rockphysics.pressure import PressureModel
from ..validation.qc import (
    Check, QCResult, Status, check_geometry, check_layer_connectivity, check_model,
    check_state,
)
from ..wave.acoustic import (
    AcousticModel, AcousticSolver, ShotRecord, SolverSettings, common_dt,
    steps_for_duration,
)
from ..wave.cpml import PMLSettings
from ..wave.sources import PointSource
from ..wave.wavelets import build_wavelet, wavelet_fmax
from ..wells.completion import layer_intersections, Completion, default_completions
from ..wells.controls import ControlMode, WellControl, check_rates, suggest_control
from ..wells.well import Well, WellSet, pattern

STAGES = ("geology", "wells", "completions", "controls", "flow", "reservoir",
          "rockphysics", "acquisition", "qc", "plan", "preview", "synthetic",
          "sim2seis", "simulate", "migrate", "decompose")

DTYPES = {"float32": np.float32, "float64": np.float64}


@dataclass
class ExperimentResult:
    """Whatever the requested stages produced."""

    config: ExperimentConfig
    domains: DomainSet
    notes: list[str] = field(default_factory=list)
    geology: GeologyModel | None = None
    wells: object | None = None
    states: object | None = None
    earth: object | None = None
    completions: dict | None = None
    controls: dict | None = None
    flow: object | None = None
    rate_warnings: list = field(default_factory=list)
    acquisition: Acquisition | None = None
    qc: QCResult | None = None
    cost: object | None = None
    preview: dict | None = None
    synthetics: dict | None = None
    volumes: dict | None = None
    time_shifts: dict | None = None
    gathers: dict[str, list[ShotRecord]] = field(default_factory=dict)
    images: dict[str, RTMResult] = field(default_factory=dict)
    decomposition: dict | None = None
    timings: dict[str, float] = field(default_factory=dict)


def _cost_class(value) -> CostClass:
    """Resolve a configured cost class, naming the options when it is wrong.

    The enum's own error is a bare ValueError quoting the string back, which
    does not say that the values carry a space or what they are.
    """
    try:
        return CostClass(value)
    except ValueError as exc:
        raise ConfigError(
            f"budget.max_cost_class is {value!r}; valid classes are "
            f"{[c.value for c in CostClass]} - note the space in 'VERY HIGH'"
        ) from exc


class Pipeline:
    """Builds an experiment stage by stage, caching what it has already done."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.domains = config.domains.build()
        self.result = ExperimentResult(config=config, domains=self.domains)

    # -- helpers ---------------------------------------------------------
    @property
    def dtype(self):
        try:
            return DTYPES[self.config.solver.dtype]
        except KeyError:
            raise ConfigError(
                f"unknown dtype {self.config.solver.dtype!r}; choose from {sorted(DTYPES)}"
            ) from None

    def wavelet(self, t) -> np.ndarray:
        """The configured source time function on the sample times ``t``."""
        source = self.config.source
        return build_wavelet(t, source.type, source.frequency, source.corners)

    @property
    def fmax(self) -> float:
        """Practical maximum frequency of the source, for sampling decisions.

        A Ricker is not band-limited at its peak: sampling the grid for the
        peak frequency alone under-samples the model by roughly a factor of
        two (spec section 11).
        """
        source = self.config.source
        return wavelet_fmax(source.type, source.frequency, source.corners,
                            source.bandwidth_fraction)

    def _timed(self, name: str, fn):
        start = time.perf_counter()
        out = fn()
        self.result.timings[name] = time.perf_counter() - start
        return out

    def rock_physics_config(self) -> RockPhysicsConfig:
        rp = self.config.rock_physics
        return RockPhysicsConfig(
            mineral_model=rp.mineral_model, dry_frame_model=rp.dry_frame_model,
            fluid_mixing=rp.fluid_mixing, brie_exponent=rp.brie_exponent,
            critical_porosity=rp.critical_porosity, coordination=rp.coordination,
            pressure=PressureModel(model=rp.pressure_model, biot=rp.biot),
            temperature=rp.temperature, salinity=rp.salinity, api=rp.api,
            gas_gravity=rp.gas_gravity, gor=rp.gor,
            min_effective_pressure=rp.min_effective_pressure,
        )

    def solver_settings(self, dt: float | None = None) -> SolverSettings:
        s = self.config.solver
        return SolverSettings(
            spatial_order=s.spatial_order, courant_safety=s.courant_safety, dt=dt,
            pml=PMLSettings(n_nodes=s.pml_nodes, r0=s.pml_r0), dtype=self.dtype,
        )

    # -- stages ----------------------------------------------------------
    def geology(self) -> GeologyModel:
        if self.result.geology is None:
            cfg = self.config.geology
            layers, faults = template(cfg.template, **cfg.parameters)
            override_layers(layers, cfg.layers)
            bodies = build_bodies(cfg.bodies)
            self.result.geology = self._timed(
                "geology", lambda: apply_bodies(
                    build_geology(self.domains.geology, layers, faults), bodies))
            self.result.notes.append(
                f"geology: template '{cfg.template}' with {len(layers)} layers, "
                f"{len(faults)} fault(s)")
            self.result.notes += [
                f"layer override: {e['name']} — "
                + ", ".join(f"{k}={v}" for k, v in e.items() if k != "name")
                for e in (cfg.layers or [])]
            # Painted before anything reads the property cube, so the flow
            # simulation sees the channel rather than the template it replaced.
            self.result.notes += [f"geobody: {b.describe()}" for b in bodies]
            self._faults = faults
        return self.result.geology

    def wells(self):
        """Explicit wells if the configuration has them, otherwise a pattern.

        A pattern is a starting point; the moment a well is placed, moved or
        renamed the configuration carries the wells themselves, and the
        pattern name stops being consulted.
        """
        if self.result.wells is None:
            spec = self.config.wells
            if spec.wells:
                ws = WellSet([
                    Well(name=w["name"], role=w["role"], x=float(w["x"]),
                         y=float(w["y"]),
                         perforation=self._perforation(w))
                    for w in spec.wells])
            else:
                ws = pattern(spec.pattern, **spec.parameters)
            ws.min_spacing = spec.min_spacing
            self.result.wells = ws
            self.result.notes += [f"wells: {n}" for n in ws.check_spacing()]
        return self.result.wells

    def _perforation(self, spec: dict) -> tuple[float, float]:
        """The interval a well is open over, in metres.

        An explicit ``perforation`` wins.  Otherwise it comes from the units
        named in ``completions``, resolved against the geology - because the
        fallback used to be a hard-coded 1,200-1,350 m, which is a depth from
        whichever model happened to be first.  On a reservoir at 600 m that
        put every well's nominal position 645 m below its own pay.

        The flow simulation never noticed: it resolves ``completions`` itself
        and perforates the right cells regardless.  Everything that reads
        ``well.position`` did notice, silently - the mechanistic generator
        centres its fronts and gas breakouts on that depth, so it was shaping
        them into barren shale and returning a scenario in which nothing
        anywhere had changed.
        """
        if "perforation" in spec:
            lo, hi = spec["perforation"]
            return float(lo), float(hi)
        names = spec.get("completions") or []
        if names:
            probe = Well(name=spec["name"], role=spec["role"],
                         x=float(spec["x"]), y=float(spec["y"]))
            found = {item.name: item for item in
                     layer_intersections(probe, self.geology())}
            hit = [found[n] for n in names if n in found]
            if hit:
                return (float(min(i.top for i in hit)),
                        float(max(i.base for i in hit)))
        return 1200.0, 1350.0

    def completions(self) -> dict:
        """Open intervals per well, named by geological unit (requirement 2)."""
        if self.result.completions is None:
            geology = self.geology()
            declared = {w["name"]: w.get("completions", [])
                        for w in self.config.wells.wells}
            out = {}
            for well in self.wells():
                names = declared.get(well.name) or []
                out[well.name] = ([Completion(layer=n) for n in names] if names
                                  else default_completions(well, geology))
            self.result.completions = out
        return self.result.completions

    def controls(self) -> dict:
        """Control mode, target and schedule per well (requirements 9, 10, 12)."""
        if self.result.controls is None:
            geology = self.geology()
            wells = self.wells()
            baseline = self.baseline_state()
            pressure = float(baseline.pressure[geology.reservoir_mask].mean())
            declared = {w["name"]: w for w in self.config.wells.wells}
            sim = self.config.simulation
            out = {}
            for well in wells:
                spec = declared.get(well.name)
                suggestion = suggest_control(
                    well, geology, self.completions()[well.name], list(wells),
                    pressure, drawdown=psi_to_pa(sim.suggested_drawdown_psi),
                    sweep_years=sim.sweep_years)
                if spec and spec.get("target") is not None:
                    mode = ControlMode(spec.get("control", suggestion.mode.value))
                    target = (psi_to_pa(float(spec["target"]))
                              if mode is ControlMode.BHP
                              else stb_per_day_to_si(float(spec["target"])))
                    suggestion = WellControl(
                        mode=mode, target=target,
                        bhp_limit=(psi_to_pa(float(spec["bhp_limit_psi"]))
                                   if spec.get("bhp_limit_psi") is not None
                                   else suggestion.bhp_limit),
                        start_day=float(spec.get("start_day", 0.0)),
                        end_day=(None if spec.get("end_day") is None
                                 else float(spec["end_day"])),
                        provenance="set explicitly")
                elif spec:
                    suggestion.start_day = float(spec.get("start_day", 0.0))
                    suggestion.end_day = (None if spec.get("end_day") is None
                                          else float(spec["end_day"]))
                out[well.name] = suggestion
            self.result.controls = out
            self.result.rate_warnings = check_rates(
                list(wells), out, geology, pressure, sim.duration_days)
            self.result.notes += [f"rates: {w}" for w in self.result.rate_warnings]
        return self.result.controls

    def baseline_state(self):
        """The initial reservoir state, before any well has produced."""
        if getattr(self, "_baseline", None) is None:
            b = self.config.reservoir.baseline
            self._baseline = initial_state(
                self.geology(), pressure_gradient=b.pressure_gradient,
                datum_pressure=b.datum_pressure, sw=b.sw, sg=b.sg,
                temperature=b.temperature, owc=b.owc, goc=b.goc,
                transition=b.transition)
        return self._baseline

    def flow_settings(self) -> FlowSettings:
        sim = self.config.simulation
        return FlowSettings(
            relperm=CoreyRelativePermeability(
                swc=sim.swc, sor=sim.sor, krw_max=sim.krw_max, kro_max=sim.kro_max,
                nw=sim.nw, no=sim.no,
                water_viscosity=sim.water_viscosity_cp * 1e-3,
                oil_viscosity=sim.oil_viscosity_cp * 1e-3),
            total_compressibility=sim.total_compressibility_per_psi / 6894.757293168,
            water_density=sim.water_density, oil_density=sim.oil_density,
            kv_over_kh=sim.kv_over_kh, gravity=sim.gravity,
            max_saturation_change=sim.max_saturation_change,
            max_timestep_days=sim.max_timestep_days)

    def flow(self, progress=None):
        """Run the two-phase flow simulation (requirement 7)."""
        if self.result.flow is None:
            sim = self.config.simulation
            baseline = self.baseline_state()
            simulator = FlowSimulator(
                self.geology(), self.wells(), self.completions(), self.controls(),
                baseline.pressure, baseline.sw, self.flow_settings())
            self.result.flow = self._timed("flow", lambda: simulator.run(
                sim.duration_days, sim.report_every_days, progress=progress))
            self.result.notes.append(
                f"flow: {self.result.flow.n_timesteps:,} timesteps, material "
                f"balance {self.result.flow.material_balance_error:.2e}")
        return self.result.flow

    def scenario(self) -> ReservoirScenario:
        s = self.config.reservoir.scenario

        def geometry(entry: dict) -> dict:
            out = {k: v for k, v in entry.items()
                   if k in ("well", "azimuth", "profile", "exponent", "taper", "z_range")}
            if "radius" in entry:
                out["radius"] = tuple(float(v) for v in entry["radius"])
            return out

        return ReservoirScenario(
            name=s.name, time_state=s.time_state,
            pressure=[PressureHalo(delta_p=float(e["delta_p_bar"]) * 1e5, **geometry(e))
                      for e in s.pressure],
            water_fronts=[SaturationFront(target_sw=float(e["target_sw"]), **geometry(e))
                          for e in s.water_fronts],
            gas=[GasBreakout(target_sg=float(e["target_sg"]), **geometry(e))
                 for e in s.gas],
        )

    def reservoir(self):
        """The four reservoir states.

        From the flow simulation when ``reservoir.source`` is ``flow`` - the
        default, and the physically consistent one - or from the parametric
        mechanistic generator, which remains the way to impose free gas that
        the two-phase flow model cannot produce.
        """
        if self.result.states is None:
            baseline = self.baseline_state()
            if self.config.reservoir.source == "flow":
                flow = self.flow()
                sim = self.config.simulation
                day = (sim.duration_days if sim.monitor_day is None
                       else float(sim.monitor_day))
                self.result.states = self._timed(
                    "reservoir",
                    lambda: build_states_from_flow(baseline, flow, day))
                self.result.notes.append(
                    f"reservoir: four states from the flow simulation at day {day:g}")
            else:
                self.result.states = self._timed(
                    "reservoir",
                    lambda: build_states(baseline, self.scenario(), self.wells(),
                                         getattr(self, "_faults", None)))
                self.result.notes.append(
                    "reservoir: four states from the mechanistic generator")
        return self.result.states

    def rockphysics(self):
        if self.result.earth is None:
            self.result.earth = self._timed(
                "rockphysics",
                lambda: build_earth_models(
                    self.reservoir(), self.geology(), self.rock_physics_config(),
                    overburden_density=self.config.rock_physics.overburden_density,
                    facies_configs=self.config.rock_physics.facies))
            warnings = self.result.earth.rock_physics["baseline"].warnings
            self.result.notes += [f"rock physics: {w}" for w in warnings]
            for name, overrides in (self.config.rock_physics.facies or {}).items():
                self.result.notes.append(
                    "rock physics [" + name + "]: "
                    + ", ".join(f"{k}={v}" for k, v in (overrides or {}).items()))
        return self.result.earth

    def propagation_models(self) -> tuple[dict[str, AcousticModel], float]:
        """Crop the four earth models to the propagation grid and pin one dt.

        Baseline and monitor gathers that sit on different time axes cannot
        be differenced, so the time step is set once, by the fastest of the
        four models (spec section 85).
        """
        earth = self.rockphysics()
        grid = self.domains.propagation
        models = {name: model.cropped(grid) for name, model in earth.models.items()}
        dt = common_dt(list(models.values()), self.config.solver.spatial_order,
                       self.config.solver.courant_safety)
        return models, dt

    def _survey_extent(self, configured: float | None, which: str) -> float:
        """Resolve one survey footprint, deriving it from the target if unset.

        A carpet that stops at the target boundary illuminates the edge of
        the anomaly from one side only and has almost no fold there, so the
        least trustworthy part of the image is exactly the part a 4D
        interpretation is read from.  The default therefore hangs the
        footprint off the target rather than off a fixed number, and a
        configuration that names an extent still gets exactly that.
        """
        if configured is not None:
            return float(configured)
        (x0, x1), (y0, y1), _ = self.domains.target.bounds
        width = max(x1 - x0, y1 - y0)
        margin = float(self.config.acquisition.target_margin)
        # Stations land on a lattice, so a footprint of exactly the requested
        # width realises as the largest whole number of spacings that fits -
        # up to one station short of the margin.  Round the request up
        # instead, or the derived extent under-delivers the one thing it
        # exists to guarantee.
        spacing = float(getattr(self.config.acquisition, f"{which}_spacing"))
        needed = width + 2.0 * margin
        extent = np.ceil(needed / spacing) * spacing if spacing > 0 else needed
        room = self._interior_width()
        if extent > room:
            # Deriving an extent that puts receivers in the absorbing layer
            # would trade one defect for a worse one, so the domain wins and
            # says so: the fix is a larger propagation grid, not a quiet
            # survey that reports full coverage it does not have.
            self.result.notes.append(
                f"acquisition: {which} extent clamped to {room:,.0f} m by the "
                f"propagation domain; covering the {width:,.0f} m target with "
                f"{margin:,.0f} m either side needs {extent:,.0f} m, so widen "
                f"propagation_bounds to get it")
            return room
        self.result.notes.append(
            f"acquisition: {which} extent {extent:,.0f} m derived from the "
            f"target ({width:,.0f} m) plus {margin:,.0f} m either side")
        return extent

    def _interior_width(self) -> float:
        """Widest centred survey the propagation domain's interior allows."""
        grid = self.domains.propagation
        pml = self.config.solver.pml_nodes
        try:
            interior = grid.padded(-pml)
        except ConfigError as exc:
            # Name the real problem. Without this the user gets a grid error
            # about node counts from three frames down, which says nothing
            # about the absorbing layer being wider than the domain.
            raise ConfigError(
                f"the propagation domain {grid.shape} has no interior left "
                f"after a {pml}-node absorbing layer on each face: the layer "
                f"needs {2 * pml} nodes per axis and an axis here has fewer. "
                f"Widen propagation_bounds, coarsen propagation_spacing, or "
                f"reduce solver.pml_nodes."
            ) from exc
        (ix0, ix1), (iy0, iy1), _ = interior.bounds
        a = self.config.acquisition
        cx, cy = (tuple(a.centre) if a.centre else
                  (0.5 * sum(grid.bounds[0]), 0.5 * sum(grid.bounds[1])))
        return 2.0 * min(cx - ix0, ix1 - cx, cy - iy0, iy1 - cy)

    def acquisition(self) -> Acquisition:
        if self.result.acquisition is None:
            a = self.config.acquisition
            if a.type.upper() != "OBN":
                raise ConfigError(
                    f"acquisition type {a.type!r} is not implemented; only 'OBN' is "
                    f"available in this version (spec section 69 defers the others)")
            grid = self.domains.propagation
            centre = tuple(a.centre) if a.centre else (
                0.5 * sum(grid.bounds[0]), 0.5 * sum(grid.bounds[1]))
            receiver_extent = self._survey_extent(a.receiver_extent, "receiver")
            source_extent = self._survey_extent(a.source_extent, "source")
            acq = OBNGeometry(
                centre=centre, receiver_spacing=a.receiver_spacing,
                receiver_extent=receiver_extent, source_spacing=a.source_spacing,
                source_line_spacing=a.source_line_spacing, source_extent=source_extent,
                receiver_extent_y=a.receiver_extent_y,
                source_extent_y=a.source_extent_y,
                receiver_depth=a.receiver_depth, source_depth=a.source_depth,
            ).build()
            if a.source_decimation > 1 or a.receiver_decimation > 1:
                acq = acq.decimated(a.source_decimation, a.receiver_decimation)
                self.result.notes.append(
                    f"acquisition: decimated by the configuration "
                    f"({a.source_decimation}, {a.receiver_decimation}) - an explicit "
                    f"choice, not an automatic one")
            self.result.acquisition = acq
        return self.result.acquisition

    def geometry_qc(self) -> QCResult:
        """The checks that need only the survey and the grid.

        Full QC pulls the flow simulation and the rock physics in behind it,
        because it checks dispersion and stability on the propagation models.
        Whether the geometry fits inside the domain is a separate question
        with a separate cost, and the page that designs a survey should not
        have to run a reservoir simulation to answer it.
        """
        result = QCResult()
        check_geometry(self.acquisition(), self.domains.propagation,
                       self.config.solver.pml_nodes, result,
                       target=self.domains.target)
        return result

    def qc(self) -> QCResult:
        if self.result.qc is None:
            models, dt = self.propagation_models()
            states = self.reservoir()
            result = QCResult()
            check_layer_connectivity(self.geology(), result)
            check_state(states.baseline, result)
            for name in SCENARIO_NAMES:
                sub = check_model(models[name], self.config.source.frequency,
                                  spatial_order=self.config.solver.spatial_order,
                                  dt=dt, courant_safety=self.config.solver.courant_safety,
                                  fmax=self.fmax)
                for check in sub.checks:
                    check.message = f"[{name}] {check.message}"
                result.checks.extend(sub.checks)
            check_geometry(self.acquisition(), self.domains.propagation,
                           self.config.solver.pml_nodes, result,
                           target=self.domains.target)
            self._check_sampling(models["baseline"], result)
            self._check_bubble_point(states, result)
            self.result.qc = result
        return self.result.qc

    def _check_bubble_point(self, states, result: QCResult) -> None:
        """Is the oil single-phase everywhere the flow model assumes it is?

        The flow model has no gas phase: it cannot release gas however far the
        pressure falls.  That is a defensible floor while the oil stays above
        its bubble point and a silent fiction below it, and the rock physics
        carries a GOR that says which.  Nothing compared the two until a
        configuration turned up whose bubble point sat three times above any
        pressure the reservoir ever saw - the seismic seeing live oil the flow
        could never have kept in solution.

        Free gas is the one thing 4D seismic responds to most, so getting this
        wrong is not a rounding error on the answer; it is the answer.
        """
        cfg = self.config.rock_physics
        pb = float(bubble_point(cfg.api, cfg.gas_gravity, cfg.gor, cfg.temperature))
        if pb <= 0.0:
            return
        pressures = [s.pressure[s.reservoir_mask] for _, s in states.items()
                     if s.reservoir_mask is not None and s.reservoir_mask.any()]
        if not pressures:
            return
        lowest = float(min(p.min() for p in pressures))
        if lowest < pb:
            result.checks.append(Check(
                Status.WARNING,
                f"reservoir pressure falls to {pa_to_psi(lowest):,.0f} psi, below "
                f"the {pa_to_psi(pb):,.0f} psi bubble point implied by GOR "
                f"{cfg.gor:g} m3/m3 at {cfg.api:g} API; the flow model has no gas "
                f"phase, so gas that would come out of solution there is neither "
                f"flowed nor seen by the rock physics"))
        else:
            result.checks.append(Check(
                Status.PASS,
                f"oil stays above its {pa_to_psi(pb):,.0f} psi bubble point "
                f"everywhere (lowest reservoir pressure "
                f"{pa_to_psi(lowest):,.0f} psi), which is what the gas-free flow "
                f"model assumes"))

    def _check_sampling(self, model: AcousticModel, result: QCResult) -> None:
        """Aperture, standoff and operator aliasing for the survey as configured.

        :func:`sampling_report` used to be reachable only from the GUI, so a
        headless run could alias its migration operator by a factor of five
        and say nothing about it.  That is the failure this project keeps
        paying for: the image fills with arcs, and the arcs get blamed on the
        velocity, on the direct arrival, on the near-field - on everything
        except the trace spacing, which is the one number that was known
        before a single wavefield was propagated.
        """
        (_, _), (_, _), (z_top, _) = self.domains.target.bounds
        report = sampling_report(
            self.acquisition(), float(z_top), float(np.median(model.vp)), self.fmax,
            self.config.acquisition.source_spacing,
            self.config.acquisition.receiver_spacing)
        notes = report.notes()
        if notes:
            for note in notes:
                result.checks.append(Check(Status.WARNING, note))
        else:
            # State the factors rather than declaring compliance. `notes()`
            # stays quiet up to 2x the limit, so a spacing that passes here is
            # not necessarily at or under it - and a check that rounds that
            # away is the reason the aliasing went unnoticed in the first
            # place. The aperture quoted is the full spread; limiting it with
            # `imaging.max_offset` raises the limit these are measured against.
            src = report.factor(report.source_spacing)
            rec = report.factor(report.receiver_spacing)
            result.checks.append(Check(
                Status.PASS,
                f"survey samples the migration operator: {report.aperture_deg:.0f} deg "
                f"aperture over the full spread, {report.standoff_wavelengths:.1f} "
                f"wavelengths of standoff, operator limit "
                f"{report.operator_limit:,.0f} m - source spacing "
                f"{report.source_spacing:,.0f} m is {src:.1f}x it, receiver "
                f"{report.receiver_spacing:,.0f} m is {rec:.1f}x"))

    def plan(self, throughput: float | None = None, benchmark: bool = False):
        """Cost estimate and budget check.  Raises rather than degrading."""
        models, dt = self.propagation_models()
        acq = self.acquisition()
        grid = self.domains.propagation
        nt = steps_for_duration(self.config.solver.record_length, dt)
        n_scenarios = len(self.config.fourd.scenarios)
        # Per shot per earth model: one forward simulation, plus the source
        # and receiver propagations that RTM adds.
        propagations = 3
        solver = AcousticSolver(models["baseline"], self.solver_settings(dt),
                                f0=self.config.source.frequency,
                                backend=self.config.solver.backend)
        snapshots = nt // max(1, int(np.floor(1.0 / (4.0 * self.fmax * dt))))
        estimate = estimate_experiment(
            grid, nt, acq.n_sources, acq.n_receivers,
            solver_state_bytes=solver.state_bytes,
            wavefield_snapshots=snapshots, propagations_per_shot=propagations,
            dtype_bytes=np.dtype(self.dtype).itemsize, n_scenarios=n_scenarios)
        self.result.cost = estimate
        if benchmark and throughput is None:
            throughput = measure_throughput(backend=self.config.solver.backend,
                                            dtype=self.dtype)
        self._throughput = throughput
        budget = ResourceBudget(
            max_ram_bytes=self.config.budget.max_ram_gb * 2**30,
            max_disk_bytes=self.config.budget.max_disk_gb * 2**30,
            max_cost_class=_cost_class(self.config.budget.max_cost_class),
        )
        check_budget(estimate, budget)
        return estimate

    def preview(self) -> dict:
        """Fast 1D convolution screening (spec section 82)."""
        if self.result.preview is None:
            models, dt = self.propagation_models()
            wavelet = self.wavelet(np.arange(int(2.0 / dt)) * dt)
            self.result.preview = self._timed("preview", lambda: {
                name: convolution_preview(models[name], wavelet, dt,
                                          t_max=self.config.solver.record_length)
                for name in self.config.fourd.scenarios
            })
        return self.result.preview

    def synthetic(self) -> dict:
        """K vertical synthetic traces, the cheap alternative to migration.

        This is the second seismic mode (``imaging.method:
        sparse_synthetic``).  It never propagates a wavefield, so it costs
        K columns rather than a survey, and what it returns is a set of
        labelled traces, never an image.
        """
        if self.result.synthetics is None:
            models, dt = self.propagation_models()
            cfg = self.config.synthetic
            # Sampled on the model grid; a lattice layout spreads itself
            # over the target instead, which is the region of interest.
            locations = resolve_locations(
                models["baseline"].grid, layout=cfg.layout, wells=self.wells(),
                points=cfg.points, count=cfg.count, region=self.domains.target)
            wavelet = self.wavelet(np.arange(int(2.0 / dt)) * dt)
            self.result.synthetics = self._timed("synthetic", lambda: {
                name: sparse_synthetic(models[name], locations, wavelet, dt,
                                       t_max=self.config.solver.record_length)
                for name in self.config.fourd.scenarios
            })
        return self.result.synthetics

    def sim2seis(self, progress=None) -> dict:
        """Convert every earth model into a synthetic seismic volume.

        The whole model, column by column, in angle stacks - the product a
        reservoir study compares against real seismic.  No wavefield is
        propagated and nothing is migrated, so it costs seconds rather than
        the hours a survey would, and it is never an image.
        """
        if self.result.volumes is None:
            # The earth models come first and are most of the wait on a cold
            # cache, so say so rather than leave a bar at zero while they
            # build.
            if progress is not None:
                progress("building the earth models", 0, 1)
            earth = self.rockphysics()
            cfg = self.config.sim2seis
            grid = earth.models["baseline"].grid
            dt = self.seismic_sample_interval()
            wavelet = self.wavelet(np.arange(int(2.0 / dt)) * dt)
            # One time axis for all four, set by the slowest of them. Letting
            # each scenario end at its own deepest two-way time would put the
            # monitors on axes a baseline cannot be subtracted from - the same
            # trap ``common_dt`` exists to close on the modelled gathers.
            t_max = cfg.record_length or max(
                float(time_from_depth(earth.rock_physics[n].vp, grid.dz).max())
                for n in self.config.fourd.scenarios)

            # One angle stack of one scenario is the finest unit of work
            # that finishes in a bounded time, so that is what the count is
            # in: reporting whole scenarios leaves the bar still for a
            # quarter of the run at a time.
            names = list(self.config.fourd.scenarios)
            per_scenario = max(len(cfg.stacks), 1)
            total = max(len(names) * per_scenario, 1)

            def build() -> dict:
                out = {}
                done = 0
                for name in names:
                    rock = earth.rock_physics[name]

                    def tick(stack, index, count, scenario=name):
                        if progress is not None:
                            progress(f"{scenario} — {stack}",
                                     done + index, total)

                    out[name] = sim2seis_volume(
                        grid, rock.vp, rock.vs, rock.rho, wavelet, dt,
                        stacks=cfg.stacks, sub_angles=cfg.sub_angles,
                        map_to_depth=cfg.map_to_depth,
                        t_max=t_max, scenario=name, progress=tick)
                    done += per_scenario
                return out

            self.result.volumes = self._timed("sim2seis", build)
            for note in self.result.volumes["baseline"].notes:
                self.result.notes.append(f"sim2seis: {note}")
            self._add_noise(self.result.volumes, dt)
        return self.result.volumes

    def noise_model(self) -> NoiseModel:
        """The configured survey noise, band-limited to the source by default."""
        spec = dict(self.config.sim2seis.noise or {})
        if spec.get("band") is not None:
            spec["band"] = tuple(spec["band"])
        return NoiseModel(**spec)

    def _add_noise(self, volumes: dict, dt: float) -> None:
        """Add correlated-between-surveys noise to every angle stack.

        One realisation per stack, shared across the scenarios, so the part
        that repeats really is the same trace in each survey - drawing
        independently per scenario would make ``repeatability`` a setting
        with no effect.
        """
        model = self.noise_model()
        self.result.notes.append(f"sim2seis: {model.describe()}")
        if not model.active:
            return
        band = (1.0, self.fmax)
        for stack in volumes["baseline"].names:
            noisy = add_survey_noise(
                {name: volumes[name].time_cubes[stack] for name in volumes},
                dt, model, band=band)
            for name, cube in noisy.items():
                volumes[name].time_cubes[stack] = cube.astype(
                    volumes[name].time_cubes[stack].dtype)

    def time_shifts(self) -> dict:
        """4D time shifts per scenario, and the monitors aligned to baseline.

        The true shift comes from the two-way-time cubes the volumes already
        carry, which is exact; the estimated one is what a windowed
        correlation recovers from the traces, which is what a processor
        would have.  Both are kept: their difference is the measurement
        error, and it cannot be seen from either alone.
        """
        if self.result.time_shifts is None:
            volumes = self.sim2seis()
            cfg = self.config.sim2seis
            dt = self.seismic_sample_interval()
            base = volumes["baseline"]
            stack = base.names[0]
            times = base.times

            def build() -> dict:
                out = {}
                for name, volume in volumes.items():
                    if name == "baseline":
                        continue
                    true = true_time_shift(base.twt, volume.twt, times)
                    estimate, centres = estimated_time_shift(
                        base.time_cube(stack), volume.time_cube(stack), dt,
                        window=cfg.shift_window, step=cfg.shift_step,
                        max_shift=cfg.max_shift)
                    full = resample_shift(estimate, centres, times)
                    # The estimate on the true shift's own axis, so the two
                    # are subtractable: the residual is the measurement error
                    # and is the reason both are kept.
                    residual = full - true
                    # Where the baseline carries no signal there is nothing
                    # to correlate, so a shift measured there is whatever the
                    # noise happened to favour.  Quality is reported on the
                    # live samples; averaging in the dead ones would grade the
                    # estimator on data it never had.
                    amplitude = np.abs(base.time_cube(stack))
                    live = amplitude > 0.05 * amplitude.max()
                    out[name] = {
                        "true": true, "estimated": estimate, "full": full,
                        "residual": residual, "live": live,
                        "centres": centres, "stack": stack,
                        "aligned": {s: align(volume.time_cube(s), full, dt)
                                    for s in volume.names},
                    }
                return out

            self.result.time_shifts = self._timed("time shifts", build)
            for name, entry in self.result.time_shifts.items():
                live = entry["live"]
                peak = 1000.0 * float(np.abs(entry["true"]).max())
                error = 1000.0 * float(np.sqrt(
                    np.mean(entry["residual"][live] ** 2))) if live.any() else float("nan")
                self.result.notes.append(
                    f"time shift [{name}]: peak {peak:.2f} ms from the two-way-"
                    f"time cubes; the windowed estimate recovers it to "
                    f"{error:.2f} ms RMS where there is signal")
        return self.result.time_shifts

    #: The sample intervals real seismic is written at, coarsest first.
    SAMPLE_LADDER = (0.004, 0.002, 0.001)

    def seismic_sample_interval(self) -> float:
        """Time sampling for the synthetic volume, seconds.

        The finite-difference dt is set by stability, not by bandwidth, and
        is far finer than any seismic sample interval; using it would make
        the cubes tens of times larger for no extra information.  Nyquist
        alone would allow something coarser still - 13 ms for an 8 Hz
        source - but a cube sampled at twice Nyquist is chunky to read and
        picks poorly, so this takes the coarsest *conventional* interval
        that still gives at least eight samples across the shortest period
        in the wavelet, and refuses to invent one in between.
        """
        explicit = self.config.sim2seis.sample_interval
        if explicit:
            return float(explicit)
        limit = 1.0 / (8.0 * self.fmax)
        for interval in self.SAMPLE_LADDER:
            if interval <= limit:
                return interval
        return self.SAMPLE_LADDER[-1]

    def simulate(self, progress=None) -> dict[str, list[ShotRecord]]:
        """Forward-model shot gathers for every scenario, independently."""
        if self.result.gathers:
            return self.result.gathers
        models, dt = self.propagation_models()
        acq = self.acquisition()
        nt = steps_for_duration(self.config.solver.record_length, dt)
        wavelet = self.wavelet(np.arange(nt) * dt)
        settings = self.solver_settings(dt)
        start = time.perf_counter()
        for name in self.config.fourd.scenarios:
            solver = AcousticSolver(models[name], settings,
                                    f0=self.config.source.frequency,
                                    backend=self.config.solver.backend)
            records = []
            for ishot, src in enumerate(acq.sources):
                records.append(solver.run(
                    PointSource(self.domains.propagation, src, wavelet), nt,
                    receivers=acq.receivers))
                if progress is not None:
                    progress(name, ishot + 1, acq.n_sources)
            self.result.gathers[name] = records
        self.result.timings["simulate"] = time.perf_counter() - start
        self._wavelet = wavelet
        return self.result.gathers

    def _mute_direct(self, gathers: dict, model, pad: float) -> dict:
        """Remove the direct arrival from every trace before migrating it.

        A migration maps an event at time ``t`` onto the surface
        ``|x - S| + |x - R| = v t``.  A reflection has a stationary point on
        that surface and collapses to it; a direct arrival has none, so it
        paints the whole surface - a smile that reaches
        ``0.5 sqrt((X + vT)^2 - X^2)`` below the acquisition plane for a
        wavelet of duration ``T``.  On the flat three-layer model that is
        1,060 to 1,206 m for the offsets in the survey, which is exactly
        where the image carries a V-shaped artefact.

        The mute uses the *fastest* velocity in the migration model, so the
        cut is as early as it can be and no reflection is removed with it.
        The gathers themselves are left alone: the cached originals are what
        a later run, or a different imaging choice, has to start from.
        """
        velocity = float(np.max(model.vp))
        out = {}
        for name, records in gathers.items():
            muted = []
            for record in records:
                receivers = np.asarray(record.receiver_positions, dtype=float)
                source = np.asarray(record.source_position, dtype=float)
                offsets = np.linalg.norm(receivers - source, axis=1)
                traces = direct_wave_mute(record.traces, offsets, record.dt,
                                          velocity=velocity, pad=pad)
                muted.append(replace(record, traces=traces.astype(record.traces.dtype)))
            out[name] = muted
        self.result.notes.append(
            f"imaging: direct arrival muted before {velocity:,.0f} m/s moveout "
            f"plus {pad * 1e3:.0f} ms - it has no stationary point and images "
            f"as a smile rather than a reflector")
        return out

    def _mute_offsets(self, gathers: dict, max_offset: float,
                      taper: float | None) -> dict:
        """Limit the migration aperture by tapering away the far offsets.

        The aliasing limit is set by the steepest angle the migration sums,
        and a long spread subtends a wide one at its ends regardless of what
        those traces carry.  Cutting the aperture raises the limit, which is
        usually cheaper than shooting finely enough to meet it.

        As with the direct-arrival mute, the stored gathers are untouched:
        the cached originals are what a different imaging choice has to start
        from.
        """
        out = {}
        for name, records in gathers.items():
            muted = []
            for record in records:
                receivers = np.asarray(record.receiver_positions, dtype=float)
                source = np.asarray(record.source_position, dtype=float)
                offsets = np.linalg.norm(receivers - source, axis=1)
                traces = offset_mute(record.traces, offsets, max_offset, taper)
                muted.append(replace(record, traces=traces.astype(record.traces.dtype)))
            out[name] = muted
        kept = int((offsets < max_offset).sum())
        self.result.notes.append(
            f"imaging: aperture limited to {max_offset:,.0f} m of offset "
            f"({kept} of {offsets.size} traces on the last shot), which raises "
            f"the operator aliasing limit that the far offsets would otherwise set")
        return out

    def migration_model(self, true_model: AcousticModel) -> AcousticModel:
        """Build the migration velocity model (spec sections 76-78).

        Migrating with a scaled or smoothed velocity is a supported
        experiment, so the perturbation is applied here and recorded, not
        guarded against.  The migration model is always built from the
        **baseline** earth: a monitor survey is migrated with the same
        operator as the baseline, or the difference would contain the
        change in the operator rather than the change in the reservoir.
        """
        cfg = self.config.imaging
        vp = true_model.vp * cfg.velocity_scale
        if cfg.velocity_smoothing > 0:
            from scipy.ndimage import gaussian_filter
            sigma = [cfg.velocity_smoothing / d for d in true_model.grid.spacing]
            vp = gaussian_filter(vp, sigma=sigma, mode="nearest")
        if cfg.velocity_scale != 1.0 or cfg.velocity_smoothing > 0:
            self.result.notes.append(
                f"imaging: migration velocity scaled by {cfg.velocity_scale:g} and "
                f"smoothed by {cfg.velocity_smoothing:g} m - a deliberate experiment")
        return AcousticModel(true_model.grid, vp, true_model.rho, name="migration")

    def correlation_start_time(self, model, acq) -> float:
        """Two-way time before which the imaging condition cannot image.

        A reflection from the top of the target cannot reach a receiver
        sooner than the straight-down-and-back time at the fastest velocity
        in the model.  Correlating earlier than that can only accumulate the
        injection near-field around the sources and receivers, which on the
        three-layer model is what made the acquisition interval brighter
        than the reservoir.  Configuring ``0`` restores correlation from
        ``t = 0``; an explicit value overrides this estimate.
        """
        configured = self.config.imaging.correlation_start_time
        if configured is not None:
            return float(configured)
        top = float(self.domains.target.bounds[2][0])
        source_depth = float(np.min(np.asarray(acq.sources, dtype=float)[:, 2]))
        vmax = float(np.max(model.vp))
        return max(0.0, 2.0 * (top - source_depth) / vmax)

    def migrate(self, progress=None) -> dict[str, RTMResult]:
        """Migrate every scenario with the same operator."""
        if self.result.images:
            return self.result.images
        gathers = self.simulate()
        models, dt = self.propagation_models()
        acq = self.acquisition()
        cfg = self.config.imaging
        if cfg.method.upper() != "RTM":
            raise ConfigError(
                f"imaging method {cfg.method!r} is not implemented; only 'RTM' is "
                f"available (spec section 81 defers Kirchhoff)")
        migration = self.migration_model(models["baseline"])
        rtm = RTMSettings(
            imaging_condition=cfg.imaging_condition, time_decimation=cfg.time_decimation,
            epsilon=cfg.epsilon, laplacian_filter=cfg.laplacian_filter,
            taper_wavelengths=cfg.taper_wavelengths, taper_radius=cfg.taper_radius,
            correlation_start_time=self.correlation_start_time(models["baseline"], acq),
            workdir=Path(self.config.output.directory) / self.config.short_hash,
        )
        settings = self.solver_settings(dt)
        if cfg.mute_direct_arrival:
            gathers = self._mute_direct(gathers, migration, cfg.mute_pad)
        if cfg.max_offset is not None:
            gathers = self._mute_offsets(gathers, cfg.max_offset, cfg.offset_taper)
        start = time.perf_counter()
        for name, records in gathers.items():
            self.result.images[name] = migrate_survey(
                records, migration, self._wavelet, acq.sources,
                solver_settings=settings, rtm=rtm, fmax=self.fmax,
                f0=self.config.source.frequency, backend=self.config.solver.backend,
                progress=(lambda i, n, name=name: progress(name, i, n)) if progress else None)
        self.result.timings["migrate"] = time.perf_counter() - start
        return self.result.images

    def decompose(self) -> dict:
        """Property-space and seismic-space decomposition (spec sections 53, 55)."""
        earth = self.rockphysics()
        mask = self.reservoir().baseline.reservoir_mask
        out: dict[str, dict] = {"property": {}, "seismic": {}}
        for attribute in ("vp", "rho", "ai"):
            out["property"][attribute] = decompose(
                *[getattr(earth.rock_physics[n], attribute) for n in SCENARIO_NAMES])
        out["property_mask"] = mask

        if self.result.images and set(SCENARIO_NAMES) <= set(self.result.images):
            out["seismic"]["rtm"] = decompose(
                *[self.result.images[n].image for n in SCENARIO_NAMES])
            base = self.result.images["baseline"].image
            out["seismic"]["nrms"] = {
                n: nrms(base, self.result.images[n].image) for n in SCENARIO_NAMES[1:]
            }

        synth = self.result.synthetics
        if synth and set(SCENARIO_NAMES) <= set(synth):
            # Kept under its own key, never merged with the migrated result:
            # these are 1D traces, and a reader must be able to tell which
            # number came from which mode.
            out["seismic"]["sparse"] = decompose(
                *[synth[n].traces for n in SCENARIO_NAMES])
            base = synth["baseline"].traces
            out["seismic"]["sparse_nrms"] = {
                n: nrms(base, synth[n].traces) for n in SCENARIO_NAMES[1:]
            }
            out["seismic"]["sparse_nrms_per_trace"] = {
                n: {loc.name: nrms(base[i], synth[n].traces[i])
                    for i, loc in enumerate(synth["baseline"].locations)}
                for n in SCENARIO_NAMES[1:]
            }
            out["trace_locations"] = synth["baseline"].locations

        volumes = self.result.volumes
        if volumes and set(SCENARIO_NAMES) <= set(volumes):
            # One decomposition per angle stack: the whole point of having
            # near and far is that they do not respond alike, so collapsing
            # them into one number would throw away the discrimination the
            # stacks exist to provide.
            out["seismic"]["sim2seis"] = {
                stack: decompose(*[volumes[n].time_cubes[stack]
                                   for n in SCENARIO_NAMES])
                for stack in volumes["baseline"].names
            }
            out["seismic"]["sim2seis_nrms"] = {
                stack: {n: nrms(volumes["baseline"].time_cubes[stack],
                                volumes[n].time_cubes[stack])
                        for n in SCENARIO_NAMES[1:]}
                for stack in volumes["baseline"].names
            }
        self.result.decomposition = out
        return out

    def run(self, stages=("geology", "reservoir", "rockphysics", "acquisition",
                          "qc", "plan"), progress=None) -> ExperimentResult:
        """Run the named stages in dependency order."""
        unknown = set(stages) - set(STAGES)
        if unknown:
            raise ConfigError(f"unknown stage(s) {sorted(unknown)}; choose from {STAGES}")
        order = [s for s in STAGES if s in stages]
        for stage in order:
            if stage in ("simulate", "migrate"):
                getattr(self, stage)(progress=progress)
            else:
                getattr(self, stage)()
        return self.result
