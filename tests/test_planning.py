import numpy as np
import pytest

from sim3d.core.errors import InfeasibleExperiment
from sim3d.core.grid import Grid3D
from sim3d.core.planning import (
    CostClass, ResourceBudget, check_budget, estimate_experiment, measure_throughput,
)


@pytest.fixture
def estimate():
    grid = Grid3D.from_bounds(((0, 1800), (0, 1800), (0, 1600)), (10, 10, 10))
    return estimate_experiment(grid, n_time_steps=1100, n_sources=60, n_receivers=400,
                               solver_state_bytes=int(0.5 * 2**30),
                               wavefield_snapshots=275, propagations_per_shot=3)


def test_estimate_reports_every_quantity_the_spec_requires(estimate):
    text = estimate.describe()
    for field in ("cells", "time steps", "sources", "receivers", "shot data",
                  "wavefield store", "peak RAM", "peak disk", "classification"):
        assert field in text
    assert estimate.cell_steps == pytest.approx(
        estimate.n_cells * 1100 * 60 * 3, rel=1e-12
    )
    assert estimate.cost_class is CostClass.VERY_HIGH


def test_no_runtime_is_quoted_without_a_benchmark(estimate):
    assert "not predicted" in estimate.describe()
    assert "h at a measured" in estimate.describe(throughput=2.0e7)


def test_cost_class_rises_with_the_experiment():
    small = Grid3D((0.0,) * 3, (10.0,) * 3, (50, 50, 50))
    assert estimate_experiment(small, 200, 1, 10, 0).cost_class is CostClass.LOW
    big = Grid3D((0.0,) * 3, (5.0,) * 3, (600, 600, 600))
    assert estimate_experiment(big, 5000, 500, 2000, 0,
                               propagations_per_shot=3).cost_class is CostClass.IMPRACTICAL


def test_over_budget_raises_with_alternatives_instead_of_degrading(estimate):
    with pytest.raises(InfeasibleExperiment) as excinfo:
        check_budget(estimate, ResourceBudget(max_ram_bytes=0.1 * 2**30,
                                              max_cost_class=CostClass.MODERATE))
    message = str(excinfo.value)
    assert "peak RAM" in message and "cost class" in message
    assert "reduce Fmax" in message
    assert "crop the propagation domain" in message
    assert len(excinfo.value.alternatives) >= 5


def test_within_budget_returns_the_estimate_unchanged(estimate):
    budget = ResourceBudget(max_ram_bytes=64 * 2**30, max_disk_bytes=1024 * 2**30,
                            max_cost_class=CostClass.IMPRACTICAL)
    assert check_budget(estimate, budget) is estimate


@pytest.mark.slow
def test_throughput_benchmark_returns_a_positive_rate():
    assert measure_throughput(shape=(32, 32, 32), n_steps=10) > 0
