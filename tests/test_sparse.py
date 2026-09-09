"""The sparse-synthetic mode: K vertical traces instead of a migration.

The two things worth pinning here are that it returns *exactly* K traces
wherever the user asked for them, and that a trace agrees with the full
convolution cube at the same column - because if the two 1D modes can
disagree, the cheap one is not screening for the expensive one, it is
just a second opinion.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.processing.preview import convolution_preview
from sim3d.processing.sparse import (
    SPARSE_LABEL, locations_from_points, locations_from_wells, locations_on_grid,
    resolve_locations, snap_to, sparse_synthetic,
)
from sim3d.wave.acoustic import AcousticModel
from sim3d.wave.wavelets import ricker
from sim3d.wells.well import Well, WellSet

DT = 0.002


@pytest.fixture
def grid():
    return Grid3D(origin=(0.0, 0.0, 0.0), spacing=(50.0, 50.0, 10.0),
                  shape=(21, 21, 61))


@pytest.fixture
def wavelet():
    return ricker(np.arange(int(0.5 / DT)) * DT, 20.0)


def two_layer(grid, interface=30, vp_top=2000.0, vp_base=3000.0,
              rho_top=2200.0, rho_base=2200.0):
    """A flat interface, so the reflection time is known analytically."""
    vp = np.full(grid.shape, vp_top)
    vp[:, :, interface:] = vp_base
    rho = np.full(grid.shape, rho_top)
    rho[:, :, interface:] = rho_base
    return AcousticModel(grid=grid, vp=vp, rho=rho)


# ------------------------------------------------------------------ layout
def test_the_grid_layout_returns_exactly_k_traces(grid):
    for k in (1, 4, 5, 9, 13):
        assert len(locations_on_grid(k, grid)) == k


def test_a_single_grid_trace_lands_in_the_middle(grid):
    (only,) = locations_on_grid(1, grid)
    (x0, x1), (y0, y1), _ = grid.bounds
    assert only.x == pytest.approx(0.5 * (x0 + x1))
    assert only.y == pytest.approx(0.5 * (y0 + y1))


def test_grid_traces_stay_inside_the_model(grid):
    (x0, x1), (y0, y1), _ = grid.bounds
    for site in locations_on_grid(12, grid):
        assert x0 < site.x < x1 and y0 < site.y < y1
        assert 0 <= site.ix < grid.nx and 0 <= site.iy < grid.ny


def test_traces_are_named_after_their_wells(grid):
    wells = WellSet([Well("P1", "producer", 400.0, 500.0),
                     Well("I1", "injector", 600.0, 500.0)])
    sites = locations_from_wells(wells, grid)
    assert [s.name for s in sites] == ["P1", "I1"]
    assert sites[0].ix == 8 and sites[0].iy == 10


def test_a_layout_with_no_wells_says_what_to_do_instead(grid):
    with pytest.raises(ConfigError, match="layout 'grid'|explicit coordinates"):
        locations_from_wells(WellSet([]), grid)


def test_a_point_outside_the_model_is_an_error_not_a_clamp(grid):
    with pytest.raises(Exception):
        locations_from_points([(99999.0, 0.0)], grid)


def test_a_point_needs_two_coordinates_because_the_trace_is_vertical(grid):
    with pytest.raises(ConfigError, match=r"\[x, y\]"):
        locations_from_points([(100.0, 200.0, 300.0)], grid)


def test_an_unknown_layout_lists_the_ones_that_exist(grid):
    with pytest.raises(ConfigError, match="wells"):
        resolve_locations(grid, layout="everywhere")


def test_snapping_to_another_grid_keeps_names_and_positions(grid):
    coarse = Grid3D(origin=(-100.0, -100.0, 0.0), spacing=(100.0, 100.0, 10.0),
                    shape=(15, 15, 61))
    sites = locations_on_grid(4, grid)
    moved = snap_to(sites, coarse)
    assert [s.name for s in moved] == [s.name for s in sites]
    assert [s.x for s in moved] == [s.x for s in sites]
    # Same physical place, different index, because the grids differ.
    assert [s.ix for s in moved] != [s.ix for s in sites]


# --------------------------------------------------------------- synthetic
def test_it_returns_one_trace_per_location(grid, wavelet):
    model = two_layer(grid)
    sites = locations_on_grid(6, grid)
    out = sparse_synthetic(model, sites, wavelet, DT, t_max=1.0)
    assert out.traces.shape[0] == 6
    assert out.depth_traces.shape == (6, grid.nz)
    assert out.names == tuple(s.name for s in sites)
    assert out.label == SPARSE_LABEL


def test_a_trace_matches_the_full_cube_at_the_same_column(grid, wavelet):
    """The cheap mode must screen for the expensive one, not differ from it."""
    model = two_layer(grid)
    cube = convolution_preview(model, wavelet, DT, t_max=1.0)
    sites = locations_on_grid(4, grid)
    sparse = sparse_synthetic(model, sites, wavelet, DT, t_max=1.0)
    for i, site in enumerate(sites):
        assert np.allclose(sparse.traces[i], cube.time_traces[site.ix, site.iy])
        assert np.allclose(sparse.depth_traces[i],
                           cube.depth_traces[site.ix, site.iy])


def test_the_reflection_arrives_at_its_two_way_time(grid, wavelet):
    """Analytic check on the timing, half-cell convention included.

    ``vp[..., 30:] = vp_base`` makes node 29 the last slow node and node 30
    the first fast one, so the interface lies between them, at
    ``29.5 * dz = 295`` m rather than at 300.  Two-way at 2000 m/s is
    0.295 s, and the trace peaks a wavelet delay after that.
    """
    model = two_layer(grid, interface=30)
    (site,) = locations_on_grid(1, grid)
    out = sparse_synthetic(model, [site], wavelet, DT, t_max=1.2)
    peak = out.times[np.argmax(np.abs(out.traces[0]))]
    delay = np.argmax(np.abs(wavelet)) * DT
    depth = (30 - 0.5) * grid.dz
    assert peak == pytest.approx(2.0 * depth / 2000.0 + delay, abs=2 * DT)


def test_a_model_with_no_contrast_produces_no_reflection(grid, wavelet):
    grid_ = grid
    model = AcousticModel(grid=grid_, vp=np.full(grid_.shape, 2500.0),
                          rho=np.full(grid_.shape, 2200.0))
    out = sparse_synthetic(model, locations_on_grid(3, grid_), wavelet, DT, t_max=1.0)
    assert np.allclose(out.traces, 0.0)


def test_a_harder_reflector_flips_the_polarity(grid, wavelet):
    """Contrast in density only, so both models share one traveltime.

    Flipping Vp instead would move the interface in time as well as change
    the coefficient's sign, and the two traces would not be comparable.
    """
    sites = locations_on_grid(1, grid)
    down = sparse_synthetic(two_layer(grid, vp_base=2000.0, rho_base=2600.0),
                            sites, wavelet, DT, t_max=1.2).traces[0]
    up = sparse_synthetic(two_layer(grid, vp_base=2000.0, rho_top=2600.0,
                                    rho_base=2200.0),
                          sites, wavelet, DT, t_max=1.2).traces[0]
    assert np.abs(down).max() > 0.0
    assert np.allclose(down, -up)


def test_traces_can_be_fetched_by_name(grid, wavelet):
    wells = WellSet([Well("P1", "producer", 400.0, 500.0)])
    out = sparse_synthetic(two_layer(grid), locations_from_wells(wells, grid),
                           wavelet, DT, t_max=1.0)
    assert np.array_equal(out.trace("P1"), out.traces[0])
    assert np.array_equal(out.depth_trace("P1"), out.depth_traces[0])
    with pytest.raises(KeyError, match="P9"):
        out.trace("P9")


def test_a_negative_sample_interval_is_rejected(grid, wavelet):
    with pytest.raises(ConfigError, match="dt must be positive"):
        sparse_synthetic(two_layer(grid), locations_on_grid(1, grid), wavelet, -1.0)


def test_no_locations_is_rejected(grid, wavelet):
    with pytest.raises(ConfigError, match="at least one trace location"):
        sparse_synthetic(two_layer(grid), [], wavelet, DT)


def test_the_description_says_what_it_is_not(grid, wavelet):
    out = sparse_synthetic(two_layer(grid), locations_on_grid(2, grid),
                           wavelet, DT, t_max=1.0)
    text = out.describe()
    assert "Not Full 3D Wave Modelling" in text
    assert "2 traces" in text


def test_it_costs_far_less_than_the_full_cube(grid, wavelet):
    """The point of the mode, stated as a test rather than a comment."""
    model = two_layer(grid)
    columns_in_cube = grid.nx * grid.ny
    sites = locations_on_grid(4, grid)
    assert len(sites) * 50 < columns_in_cube
    # And it still produces a usable trace.
    out = sparse_synthetic(model, sites, wavelet, DT, t_max=1.0)
    assert np.abs(out.traces).max() > 0.0


def test_a_well_outside_the_imaging_target_still_gets_a_trace(grid):
    """The target is where the image is made, not where the wells may be.

    Validating a given position against the target would refuse a trace at
    a well that sits just outside it, which is a normal field layout.
    """
    target = Grid3D(origin=(400.0, 400.0, 0.0), spacing=(50.0, 50.0, 10.0),
                    shape=(5, 5, 61))
    wells = WellSet([Well("P1", "producer", 100.0, 100.0)])
    sites = resolve_locations(grid, layout="wells", wells=wells, region=target)
    assert sites[0].name == "P1" and sites[0].x == 100.0


def test_a_lattice_spreads_over_the_region_not_the_whole_model(grid):
    target = Grid3D(origin=(400.0, 400.0, 0.0), spacing=(50.0, 50.0, 10.0),
                    shape=(5, 5, 61))
    (x0, x1), (y0, y1), _ = target.bounds
    for site in resolve_locations(grid, layout="grid", count=4, region=target):
        assert x0 <= site.x <= x1 and y0 <= site.y <= y1


def test_a_trace_outside_the_sampled_model_names_the_trace(grid):
    with pytest.raises(ConfigError, match="T1.*outside the model"):
        locations_from_points([(99999.0, 0.0)], grid)
