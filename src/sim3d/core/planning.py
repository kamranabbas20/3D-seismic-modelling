"""Computational cost estimation and the no-silent-degradation policy.

Spec sections 8 and 13.  Before any full-wave run the platform must state
what the run will cost and, when the cost is unreasonable, refuse and
offer scientifically defensible alternatives rather than quietly
coarsening the grid, lowering the bandwidth, shortening the record or
dropping shots.

Runtime is only ever reported when the caller supplies a *measured*
throughput for the machine in question (:func:`measure_throughput`).  A
predicted runtime with no benchmark behind it would be an invention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .errors import InfeasibleExperiment
from .grid import Grid3D


class CostClass(str, Enum):
    """Qualitative classification of an experiment's cost."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"
    IMPRACTICAL = "IMPRACTICAL"


#: Upper bounds in cell-steps (cells x time steps x propagations) per class.
COST_THRESHOLDS: tuple[tuple[float, CostClass], ...] = (
    (1e9, CostClass.LOW),
    (5e10, CostClass.MODERATE),
    (5e11, CostClass.HIGH),
    (5e12, CostClass.VERY_HIGH),
)

GIB = float(2**30)


@dataclass
class CostEstimate:
    """Everything spec section 13 requires before a full-wave run."""

    grid: Grid3D
    n_time_steps: int
    n_sources: int
    n_receivers: int
    dtype_bytes: int
    propagations_per_shot: int
    wavefield_snapshots: int
    solver_state_bytes: int

    # -- derived sizes ---------------------------------------------------
    @property
    def n_cells(self) -> int:
        return self.grid.n_cells

    @property
    def cell_steps(self) -> float:
        """Total cell updates across the whole experiment."""
        return float(self.n_cells) * self.n_time_steps * self.n_sources * self.propagations_per_shot

    @property
    def shot_data_bytes(self) -> int:
        """Size of the recorded gathers for the whole survey."""
        return self.n_sources * self.n_receivers * self.n_time_steps * self.dtype_bytes

    @property
    def checkpoint_bytes(self) -> int:
        """Peak storage for one shot's stored source wavefield."""
        return self.wavefield_snapshots * self.n_cells * self.dtype_bytes

    @property
    def ram_bytes(self) -> int:
        """Peak resident memory: solver state plus one shot's image and illumination."""
        image = 2 * self.n_cells * 8  # image and illumination, float64
        return self.solver_state_bytes + image

    @property
    def disk_bytes(self) -> int:
        """Peak disk: the survey's gathers plus one shot's wavefield store."""
        return self.shot_data_bytes + self.checkpoint_bytes

    @property
    def cost_class(self) -> CostClass:
        for limit, klass in COST_THRESHOLDS:
            if self.cell_steps < limit:
                return klass
        return CostClass.IMPRACTICAL

    def runtime_seconds(self, throughput_cell_steps_per_s: float) -> float:
        """Runtime implied by a *measured* throughput on this machine."""
        if throughput_cell_steps_per_s <= 0:
            raise InfeasibleExperiment("throughput must be positive")
        return self.cell_steps / float(throughput_cell_steps_per_s)

    def describe(self, throughput: float | None = None) -> str:
        lines = [
            "Computational estimate:",
            f"  grid              {self.grid.nx} x {self.grid.ny} x {self.grid.nz} "
            f"= {self.n_cells:,} cells",
            f"  time steps        {self.n_time_steps:,}",
            f"  sources           {self.n_sources:,}",
            f"  receivers         {self.n_receivers:,}",
            f"  propagations      {self.propagations_per_shot} per shot "
            f"({self.n_sources * self.propagations_per_shot:,} total)",
            f"  cell-steps        {self.cell_steps:.3g}",
            f"  shot data         {self.shot_data_bytes / GIB:.3f} GiB",
            f"  wavefield store   {self.checkpoint_bytes / GIB:.3f} GiB "
            f"({self.wavefield_snapshots} snapshots)",
            f"  peak RAM          {self.ram_bytes / GIB:.3f} GiB",
            f"  peak disk         {self.disk_bytes / GIB:.3f} GiB",
            f"  classification    {self.cost_class.value}",
        ]
        if throughput is not None:
            secs = self.runtime_seconds(throughput)
            lines.append(
                f"  runtime           ~{secs / 3600:.2f} h at a measured "
                f"{throughput / 1e6:.1f} Mcell-steps/s"
            )
        else:
            lines.append(
                "  runtime           not predicted; benchmark this machine with "
                "measure_throughput() first"
            )
        return "\n".join(lines)


@dataclass
class ResourceBudget:
    """What the user is willing to spend.  Exceeding it is an error, not a hint."""

    max_ram_bytes: float = 8 * GIB
    max_disk_bytes: float = 200 * GIB
    max_cost_class: CostClass = CostClass.HIGH


def check_budget(estimate: CostEstimate, budget: ResourceBudget | None = None,
                 alternatives: list[str] | None = None) -> CostEstimate:
    """Raise :class:`InfeasibleExperiment` if the estimate exceeds the budget.

    The exception carries the alternatives the user may choose between.
    Nothing is changed automatically: which scientific assumption to give
    up is the user's decision, not the software's.
    """
    budget = budget or ResourceBudget()
    order = list(CostClass)
    problems = []
    if estimate.ram_bytes > budget.max_ram_bytes:
        problems.append(
            f"peak RAM {estimate.ram_bytes / GIB:.2f} GiB exceeds the "
            f"{budget.max_ram_bytes / GIB:.2f} GiB budget"
        )
    if estimate.disk_bytes > budget.max_disk_bytes:
        problems.append(
            f"peak disk {estimate.disk_bytes / GIB:.2f} GiB exceeds the "
            f"{budget.max_disk_bytes / GIB:.2f} GiB budget"
        )
    if order.index(estimate.cost_class) > order.index(budget.max_cost_class):
        problems.append(
            f"cost class {estimate.cost_class.value} exceeds the accepted "
            f"{budget.max_cost_class.value}"
        )
    if problems:
        raise InfeasibleExperiment(
            "This experiment is outside the resource budget:\n"
            + "\n".join(f"  * {p}" for p in problems),
            alternatives if alternatives is not None else default_alternatives(estimate),
        )
    return estimate


def default_alternatives(estimate: CostEstimate) -> list[str]:
    """The standard menu of section 8, phrased against this estimate."""
    return [
        "reduce Fmax, which allows a coarser grid at the same dispersion tolerance",
        "crop the propagation domain to a target window around the wells "
        f"(currently {estimate.grid.extent[0]:.0f} x {estimate.grid.extent[1]:.0f} x "
        f"{estimate.grid.extent[2]:.0f} m)",
        "increase the grid spacing, after checking the dispersion criterion "
        "with fdscheme.recommend_spacing",
        f"reduce the number of sources (currently {estimate.n_sources}) or use a "
        "sparser acquisition",
        "shorten the recording duration if the target reflection arrives earlier",
        "switch to Preview Mode (fast convolution) for screening, then run "
        "full-wave only on the selected cases",
        "restrict modelling to a selected reservoir region",
        "defer this run until a GPU backend is available",
    ]


def estimate_experiment(grid: Grid3D, n_time_steps: int, n_sources: int, n_receivers: int,
                        solver_state_bytes: int, wavefield_snapshots: int = 0,
                        propagations_per_shot: int = 1, dtype_bytes: int = 4) -> CostEstimate:
    """Build a :class:`CostEstimate` for a survey."""
    return CostEstimate(
        grid=grid, n_time_steps=int(n_time_steps), n_sources=int(n_sources),
        n_receivers=int(n_receivers), dtype_bytes=int(dtype_bytes),
        propagations_per_shot=int(propagations_per_shot),
        wavefield_snapshots=int(wavefield_snapshots),
        solver_state_bytes=int(solver_state_bytes),
    )


def measure_throughput(shape: tuple[int, int, int] = (64, 64, 64), n_steps: int = 40,
                       backend=None, dtype=np.float32) -> float:
    """Benchmark this machine in cell-steps per second.

    Runs a short homogeneous simulation so that any runtime figure quoted
    later rests on a measurement of the hardware actually in use.
    """
    import time

    from ..wave.acoustic import AcousticModel, AcousticSolver, SolverSettings
    from ..wave.cpml import PMLSettings
    from ..wave.sources import NoSource

    grid = Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), shape)
    model = AcousticModel(grid, np.full(shape, 2000.0), np.full(shape, 2200.0))
    settings = SolverSettings(pml=PMLSettings(n_nodes=8), dtype=dtype)
    solver = AcousticSolver(model, settings, f0=20.0, backend=backend)
    solver.run(NoSource(), 2)  # warm up any JIT compilation
    start = time.perf_counter()
    solver.run(NoSource(), n_steps)
    elapsed = time.perf_counter() - start
    return grid.n_cells * n_steps / elapsed
