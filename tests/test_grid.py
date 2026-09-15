import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import DomainSet, Grid3D


def test_geometry_and_bounds():
    g = Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (11, 21, 41))
    assert g.n_cells == 11 * 21 * 41
    assert g.extent == pytest.approx((100.0, 200.0, 200.0))
    assert g.bounds[2] == (0.0, 200.0)
    assert g.axis("z")[-1] == pytest.approx(200.0)


def test_from_bounds_never_truncates_the_requested_volume():
    g = Grid3D.from_bounds(((0.0, 95.0), (0.0, 100.0), (0.0, 100.0)), (10.0, 10.0, 10.0))
    assert g.bounds[0][1] >= 95.0


def test_points_outside_the_grid_raise_rather_than_snap():
    g = Grid3D((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (11, 11, 11))
    assert g.nearest_index((13.0, 0.0, 0.0)) == (1, 0, 0)
    with pytest.raises(ConfigError, match="outside grid bounds"):
        g.nearest_index((130.0, 0.0, 0.0))


def test_padding_keeps_spacing_and_grows_both_ends():
    g = Grid3D((100.0, 100.0, 100.0), (10.0, 10.0, 10.0), (11, 11, 11))
    p = g.padded(5)
    assert p.origin == (50.0, 50.0, 50.0)
    assert p.shape == (21, 21, 21)
    assert p.spacing == g.spacing


def test_domain_hierarchy_is_enforced():
    geo = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (10, 10, 10))
    prop = Grid3D.from_bounds(((600, 2400), (600, 2400), (200, 1800)), (10, 10, 10))
    target = Grid3D.from_bounds(((1000, 2000), (1000, 2000), (800, 1200)), (10, 10, 10))
    ds = DomainSet(geo, prop, target)
    assert "propagation" in ds.summary()

    with pytest.raises(ConfigError, match="not contained"):
        DomainSet(geo, prop, Grid3D.from_bounds(((0, 2000), (1000, 2000), (800, 1200)), (10, 10, 10)))
    with pytest.raises(ConfigError, match="not contained"):
        DomainSet(prop, geo, target)
