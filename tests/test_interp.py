import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.wave.interp import PointSet


@pytest.fixture
def grid():
    return Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (11, 11, 11))


def test_trilinear_gather_is_exact_for_a_linear_field(grid):
    x, y, z = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")
    f = 2.0 * x + 3.0 * y - 1.5 * z
    ps = PointSet(grid, [[13.0, 27.0, 41.0], [100.0, 100.0, 100.0], [0.0, 0.0, 0.0]])
    assert ps.gather(f) == pytest.approx([2 * 13 + 3 * 27 - 1.5 * 41, 350.0, 0.0])


def test_scatter_conserves_the_injected_amount(grid):
    field = np.zeros(grid.shape)
    PointSet(grid, [[13.0, 27.0, 41.0], [55.0, 55.0, 55.0]]).scatter(field, [2.0, 3.0])
    assert field.sum() == pytest.approx(5.0)


def test_scatter_and_gather_are_adjoint(grid):
    """<Sx, y> == <x, G y>: RTM's correctness depends on this."""
    rng = np.random.default_rng(1)
    ps = PointSet(grid, rng.uniform(5.0, 95.0, size=(7, 3)))
    values = rng.standard_normal(7)
    field = rng.standard_normal(grid.shape)
    scattered = np.zeros(grid.shape)
    ps.scatter(scattered, values)
    assert np.sum(scattered * field) == pytest.approx(np.dot(values, ps.gather(field)))


def test_points_outside_the_domain_are_refused_not_clipped(grid):
    with pytest.raises(ConfigError, match="will not"):
        PointSet(grid, [[500.0, 0.0, 0.0]], "receivers")
