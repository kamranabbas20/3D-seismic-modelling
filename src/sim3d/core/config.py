"""Experiment configuration from YAML or JSON (spec sections 101-102, 117).

Every experiment is reproducible from its configuration file: the geology,
the wells, the reservoir scenario, the rock-physics models, the solver, the
acquisition and the imaging settings are all here, and nothing that affects
a result is set anywhere else.

Two rules make the file trustworthy rather than merely present:

* **Unknown keys are errors.**  A typo in ``dry_frame_model`` must not be
  silently ignored, leaving the run to proceed with a default the user
  never chose.
* **The content hash covers exactly the scientific inputs.**  Display
  settings and output paths are excluded, so changing a colour map does not
  invalidate a cached migration, and changing a wavelet frequency does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml

from .errors import ConfigError
from .grid import DomainSet, Grid3D
# Safe: nothing under ``processing`` or ``wave`` imports this module back.
from ..processing.sim2seis import DEFAULT_STACKS, build_stacks
from ..processing.sparse import LAYOUTS

#: The seismic modes ``imaging.method`` accepts.
IMAGING_METHODS = ("RTM", "sparse_synthetic", "sim2seis")


def _build(cls, data: dict[str, Any], path: str = ""):
    """Instantiate a dataclass from a mapping, rejecting unknown keys.

    ``from __future__ import annotations`` turns field types into strings, so
    the nested dataclasses have to be recovered with ``get_type_hints``
    rather than read off ``field.type`` directly.
    """
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise ConfigError(
            f"{path or cls.__name__} must be a mapping, got {type(data).__name__}"
        )
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in {path or cls.__name__}; "
            f"valid keys are {sorted(known)}. sim3d does not ignore unrecognised "
            f"settings, because a typo would otherwise run with a default you "
            f"did not choose."
        )
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        annotation = hints.get(f.name, f.type)
        where = f"{path}.{f.name}" if path else f.name
        if is_dataclass(annotation):
            if not isinstance(value, dict):
                raise ConfigError(
                    f"{where} must be a mapping of settings, got "
                    f"{type(value).__name__}"
                )
            kwargs[f.name] = _build(annotation, value, where)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


@dataclass
class ProjectConfig:
    name: str = "mechanistic_4d"
    description: str = ""
    seed: int = 1


@dataclass
class DomainConfig:
    """The three-domain hierarchy of spec section 6."""

    geology_bounds: list = field(default_factory=lambda: [[0, 3000], [0, 3000], [0, 2200]])
    geology_spacing: list = field(default_factory=lambda: [25.0, 25.0, 10.0])
    propagation_bounds: list | None = None
    propagation_spacing: list | None = None
    target_bounds: list | None = None

    def build(self) -> DomainSet:
        geology = Grid3D.from_bounds(self.geology_bounds, self.geology_spacing)
        prop_bounds = self.propagation_bounds or self.geology_bounds
        prop_spacing = self.propagation_spacing or self.geology_spacing
        propagation = Grid3D.from_bounds(prop_bounds, prop_spacing)
        target = Grid3D.from_bounds(self.target_bounds or prop_bounds, prop_spacing)
        return DomainSet(geology=geology, propagation=propagation, target=target)


@dataclass
class GeologyConfig:
    template: str = "anticline"
    parameters: dict = field(default_factory=dict)


@dataclass
class WellsConfig:
    pattern: str = "demonstration"
    parameters: dict = field(default_factory=dict)
    min_spacing: float = 500.0
    #: Explicit wells. When present these replace ``pattern`` entirely, which
    #: is how a pattern placed with the mouse is stored.
    wells: list = field(default_factory=list)


@dataclass
class BaselineConfig:
    sw: float = 0.30
    sg: float = 0.0
    temperature: float = 80.0
    pressure_gradient: float = 10500.0
    datum_pressure: float = 101325.0


@dataclass
class ScenarioConfig:
    name: str = "monitor"
    time_state: str = "T4"
    pressure: list = field(default_factory=list)
    water_fronts: list = field(default_factory=list)
    gas: list = field(default_factory=list)


@dataclass
class ReservoirConfig:
    #: ``flow`` runs the two-phase simulator; ``mechanistic`` uses the
    #: parametric generator, which is the only way to impose free gas.
    source: str = "flow"
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)


@dataclass
class WellSpec:
    """One explicitly defined well, as the GUI writes it (requirements 1, 2, 10, 12)."""

    name: str = "P1"
    role: str = "producer"
    x: float = 0.0
    y: float = 0.0
    #: Geological units the well is open in; empty means every reservoir unit.
    completions: list = field(default_factory=list)
    #: ``liquid_rate``, ``oil_rate``, ``water_rate`` or ``bhp``.
    control: str = "liquid_rate"
    #: Rate in STB/day, or BHP in psi when ``control`` is ``bhp``.
    target: float | None = None
    bhp_limit_psi: float | None = None
    start_day: float = 0.0
    end_day: float | None = None


@dataclass
class SimulationConfig:
    """Flow-simulation duration and timestepping (requirement 7)."""

    duration_days: float = 1825.0
    report_every_days: float = 91.25
    max_timestep_days: float = 30.0
    #: The monitor survey is modelled at this time; ``None`` uses the end.
    monitor_day: float | None = None
    #: Corey relative permeability.
    swc: float = 0.20
    sor: float = 0.25
    krw_max: float = 0.35
    kro_max: float = 0.90
    nw: float = 2.5
    no: float = 2.0
    water_viscosity_cp: float = 0.5
    oil_viscosity_cp: float = 1.0
    total_compressibility_per_psi: float = 3.0e-6
    water_density: float = 1030.0
    oil_density: float = 800.0
    kv_over_kh: float = 0.1
    gravity: bool = True
    max_saturation_change: float = 0.05
    #: Drawdown used when suggesting a rate for a new well, psi.
    suggested_drawdown_psi: float = 500.0
    sweep_years: float = 10.0


@dataclass
class RockPhysicsSection:
    mineral_model: str = "vrh"
    dry_frame_model: str = "soft_sand"
    fluid_mixing: str = "wood"
    brie_exponent: float = 3.0
    critical_porosity: float = 0.40
    coordination: float = 9.0
    pressure_model: str = "hertz_mindlin"
    biot: float = 1.0
    temperature: float = 80.0
    salinity: float = 35000.0
    api: float = 30.0
    gas_gravity: float = 0.65
    gor: float = 100.0
    min_effective_pressure: float = 1.0e6
    overburden_density: float = 2300.0


@dataclass
class SolverConfig:
    physics: str = "acoustic"
    spatial_order: int = 8
    courant_safety: float = 0.90
    pml_nodes: int = 12
    pml_r0: float = 1.0e-5
    dtype: str = "float32"
    backend: str = "auto"
    record_length: float = 2.0


@dataclass
class SourceConfig:
    type: str = "ricker"
    frequency: float = 20.0
    #: Spectral fraction defining the practical Fmax used for grid checks.
    bandwidth_fraction: float = 0.05


@dataclass
class AcquisitionConfig:
    type: str = "OBN"
    centre: list | None = None
    receiver_spacing: float = 200.0
    receiver_extent: float = 1600.0
    source_spacing: float = 150.0
    source_line_spacing: float = 300.0
    source_extent: float = 1600.0
    receiver_depth: float = 400.0
    source_depth: float = 380.0
    source_decimation: int = 1
    receiver_decimation: int = 1


@dataclass
class ImagingConfig:
    #: ``RTM`` migrates modelled gathers; ``sparse_synthetic`` skips
    #: propagation entirely and builds K vertical 1D traces instead, which
    #: is a screening mode and never an image.
    method: str = "RTM"
    imaging_condition: str = "source_normalized"
    time_decimation: int | None = None
    laplacian_filter: bool = True
    epsilon: float = 1.0e-4
    #: Migrate with a perturbed velocity: 1.0 is the true model.
    velocity_scale: float = 1.0
    #: Gaussian smoothing of the migration velocity, in metres.
    velocity_smoothing: float = 0.0

    def __post_init__(self) -> None:
        if self.method.upper() not in {m.upper() for m in IMAGING_METHODS}:
            raise ConfigError(
                f"unknown imaging method {self.method!r}; "
                f"choose from {list(IMAGING_METHODS)}")


@dataclass
class SyntheticConfig:
    """Settings for the sparse-synthetic mode.

    Its own section rather than more keys on ``imaging`` so that the
    dependency graph stays precise: changing an RTM setting must not
    invalidate the traces, and changing the trace layout must not
    invalidate a migration that took an hour.
    """

    #: Where the K traces go: at the wells, at explicit ``points``, or on
    #: a lattice of ``count``.
    layout: str = "wells"
    count: int = 9
    points: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.layout not in LAYOUTS:
            raise ConfigError(
                f"unknown trace layout {self.layout!r}; choose from {list(LAYOUTS)}")


@dataclass
class Sim2SeisConfig:
    """Settings for the synthetic seismic volume.

    Its own section, like ``synthetic``, so the dependency graph stays
    precise: re-stacking at different angles must not invalidate a
    migration, and an RTM setting must not invalidate the volume.
    """

    #: Angle stacks, each ``{name, angles: [min, max]}`` in degrees.
    stacks: list = field(default_factory=lambda: [dict(s) for s in DEFAULT_STACKS])
    #: Angles averaged within each stack.
    sub_angles: int = 5
    #: Time sample interval in seconds; ``None`` derives one from the
    #: source bandwidth, picking a conventional 4, 2 or 1 ms.
    sample_interval: float | None = None
    #: Length of the time axis in seconds; ``None`` uses the deepest
    #: two-way time in the model.  This is deliberately *not*
    #: ``solver.record_length``: that is the listening time of a survey
    #: whose sources sit near the reservoir, while a synthetic column is
    #: timed from the surface and needs the whole overburden.
    record_length: float | None = None
    #: Also resample every cube onto the depth axis.
    map_to_depth: bool = True

    def __post_init__(self) -> None:
        build_stacks(self.stacks)        # validate now, not at run time
        if self.sub_angles < 1:
            raise ConfigError(
                f"sub_angles must be at least 1, got {self.sub_angles}")
        if self.sample_interval is not None and self.sample_interval <= 0:
            raise ConfigError(
                f"sample_interval must be positive, got {self.sample_interval}")
        if self.record_length is not None and self.record_length <= 0:
            raise ConfigError(
                f"record_length must be positive, got {self.record_length}")


@dataclass
class FourDConfig:
    scenarios: list = field(
        default_factory=lambda: ["baseline", "pressure_only", "saturation_only", "combined"])


@dataclass
class BudgetConfig:
    max_ram_gb: float = 8.0
    max_disk_gb: float = 200.0
    max_cost_class: str = "HIGH"


@dataclass
class OutputConfig:
    """Excluded from the content hash: changing these cannot change the science."""

    directory: str = "runs"
    save_snapshots: list = field(default_factory=list)
    save_gathers: bool = True


@dataclass
class ExperimentConfig:
    """The complete, reproducible definition of one experiment."""

    project: ProjectConfig = field(default_factory=ProjectConfig)
    domains: DomainConfig = field(default_factory=DomainConfig)
    geology: GeologyConfig = field(default_factory=GeologyConfig)
    wells: WellsConfig = field(default_factory=WellsConfig)
    reservoir: ReservoirConfig = field(default_factory=ReservoirConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    rock_physics: RockPhysicsSection = field(default_factory=RockPhysicsSection)
    solver: SolverConfig = field(default_factory=SolverConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    acquisition: AcquisitionConfig = field(default_factory=AcquisitionConfig)
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    sim2seis: Sim2SeisConfig = field(default_factory=Sim2SeisConfig)
    fourd: FourDConfig = field(default_factory=FourDConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    #: Sections that do not affect any scientific result.
    NON_SCIENTIFIC = ("output",)

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def scientific_dict(self) -> dict:
        """The configuration with non-scientific sections removed."""
        return {k: v for k, v in self.to_dict().items() if k not in self.NON_SCIENTIFIC}

    def content_hash(self) -> str:
        """SHA-256 of the canonical scientific configuration.

        Two experiments with the same hash must produce the same numbers, so
        this is what the cache and the provenance record key on.
        """
        canonical = json.dumps(self.scientific_dict(), sort_keys=True,
                               separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    @property
    def short_hash(self) -> str:
        return self.content_hash()[:12]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        path.write_text(text, encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentConfig":
        return _build(cls, data)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"configuration file not found: {path}")
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ConfigError(f"{path} does not contain a mapping at the top level")
        return cls.from_dict(data)

    def describe(self) -> str:
        domains = self.domains.build()
        return "\n".join([
            f"Experiment '{self.project.name}' [{self.short_hash}]",
            domains.summary(),
            f"  geology template  {self.geology.template} {self.geology.parameters}",
            f"  well pattern      {self.wells.pattern}",
            f"  scenario          {self.reservoir.scenario.name} "
            f"at {self.reservoir.scenario.time_state}",
            f"  simulation        {self.simulation.duration_days:g} days, "
            f"reported every {self.simulation.report_every_days:g} days",
            f"  rock physics      {self.rock_physics.mineral_model} / "
            f"{self.rock_physics.dry_frame_model} / {self.rock_physics.fluid_mixing} / "
            f"{self.rock_physics.pressure_model}",
            f"  solver            {self.solver.physics}, order "
            f"{self.solver.spatial_order}, backend {self.solver.backend}",
            f"  source            {self.source.type} at {self.source.frequency:g} Hz",
            f"  acquisition       {self.acquisition.type}, nodes "
            f"{self.acquisition.receiver_spacing:g} m, shots "
            f"{self.acquisition.source_spacing:g} m",
            f"  imaging           {self.imaging.method}, "
            f"{self.imaging.imaging_condition}",
            f"  4D scenarios      {', '.join(self.fourd.scenarios)}",
        ])
