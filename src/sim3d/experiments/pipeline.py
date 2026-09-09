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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..acquisition.geometry import Acquisition, OBNGeometry
from ..core.config import ExperimentConfig
from ..core.errors import ConfigError
from ..core.grid import DomainSet
from ..core.planning import (
    CostClass, ResourceBudget, check_budget, estimate_experiment, measure_throughput,
)
from ..fourd.decomposition import decompose
from ..fourd.metrics import nrms
from ..fourd.scenarios import SCENARIO_NAMES, build_earth_models, build_states
from ..geology.builder import GeologyModel, build_geology
from ..geology.templates import template
from ..imaging.rtm import RTMSettings, RTMResult, migrate_survey
from ..processing.preview import convolution_preview
from ..reservoir.mechanistic import (
    GasBreakout, PressureHalo, ReservoirScenario, SaturationFront,
)
from ..reservoir.state import initial_state
from ..rockphysics.model import RockPhysicsConfig
from ..rockphysics.pressure import PressureModel
from ..validation.qc import QCResult, check_geometry, check_model, check_state
from ..wave.acoustic import (
    AcousticModel, AcousticSolver, ShotRecord, SolverSettings, common_dt,
    steps_for_duration,
)
from ..wave.cpml import PMLSettings
from ..wave.sources import PointSource
from ..wave.wavelets import ricker, ricker_fmax
from ..wells.well import pattern

STAGES = ("geology", "reservoir", "rockphysics", "acquisition", "qc", "plan",
          "preview", "simulate", "migrate", "decompose")

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
    acquisition: Acquisition | None = None
    qc: QCResult | None = None
    cost: object | None = None
    preview: dict | None = None
    gathers: dict[str, list[ShotRecord]] = field(default_factory=dict)
    images: dict[str, RTMResult] = field(default_factory=dict)
    decomposition: dict | None = None
    timings: dict[str, float] = field(default_factory=dict)


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

    @property
    def fmax(self) -> float:
        """Practical maximum frequency of the source, for sampling decisions.

        A Ricker is not band-limited at its peak: sampling the grid for the
        peak frequency alone under-samples the model by roughly a factor of
        two (spec section 11).
        """
        source = self.config.source
        return ricker_fmax(source.frequency, source.bandwidth_fraction)

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
            self.result.geology = self._timed(
                "geology", lambda: build_geology(self.domains.geology, layers, faults))
            self.result.notes.append(
                f"geology: template '{cfg.template}' with {len(layers)} layers, "
                f"{len(faults)} fault(s)")
            self._faults = faults
        return self.result.geology

    def wells(self):
        if self.result.wells is None:
            w = self.config.wells
            ws = pattern(w.pattern, **w.parameters)
            ws.min_spacing = w.min_spacing
            self.result.wells = ws
            self.result.notes += [f"wells: {n}" for n in ws.check_spacing()]
        return self.result.wells

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
        if self.result.states is None:
            geology = self.geology()
            b = self.config.reservoir.baseline
            baseline = initial_state(
                geology, pressure_gradient=b.pressure_gradient,
                datum_pressure=b.datum_pressure, sw=b.sw, sg=b.sg,
                temperature=b.temperature)
            self.result.states = self._timed(
                "reservoir",
                lambda: build_states(baseline, self.scenario(), self.wells(),
                                     getattr(self, "_faults", None)))
            self.result.notes.append(
                "reservoir: four isolated states built and isolation verified")
        return self.result.states

    def rockphysics(self):
        if self.result.earth is None:
            self.result.earth = self._timed(
                "rockphysics",
                lambda: build_earth_models(
                    self.reservoir(), self.geology(), self.rock_physics_config(),
                    overburden_density=self.config.rock_physics.overburden_density))
            warnings = self.result.earth.rock_physics["baseline"].warnings
            self.result.notes += [f"rock physics: {w}" for w in warnings]
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
            acq = OBNGeometry(
                centre=centre, receiver_spacing=a.receiver_spacing,
                receiver_extent=a.receiver_extent, source_spacing=a.source_spacing,
                source_line_spacing=a.source_line_spacing, source_extent=a.source_extent,
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

    def qc(self) -> QCResult:
        if self.result.qc is None:
            models, dt = self.propagation_models()
            states = self.reservoir()
            result = QCResult()
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
                           self.config.solver.pml_nodes, result)
            self.result.qc = result
        return self.result.qc

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
            max_cost_class=CostClass(self.config.budget.max_cost_class),
        )
        check_budget(estimate, budget)
        return estimate

    def preview(self) -> dict:
        """Fast 1D convolution screening (spec section 82)."""
        if self.result.preview is None:
            models, dt = self.propagation_models()
            wavelet = ricker(np.arange(int(2.0 / dt)) * dt, self.config.source.frequency)
            self.result.preview = self._timed("preview", lambda: {
                name: convolution_preview(models[name], wavelet, dt,
                                          t_max=self.config.solver.record_length)
                for name in self.config.fourd.scenarios
            })
        return self.result.preview

    def simulate(self, progress=None) -> dict[str, list[ShotRecord]]:
        """Forward-model shot gathers for every scenario, independently."""
        if self.result.gathers:
            return self.result.gathers
        models, dt = self.propagation_models()
        acq = self.acquisition()
        nt = steps_for_duration(self.config.solver.record_length, dt)
        wavelet = ricker(np.arange(nt) * dt, self.config.source.frequency)
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
            workdir=Path(self.config.output.directory) / self.config.short_hash,
        )
        settings = self.solver_settings(dt)
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
