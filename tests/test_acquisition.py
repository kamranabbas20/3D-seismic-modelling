import numpy as np
import pytest

from sim3d.acquisition import (
    Acquisition, OBNGeometry, azimuth_distribution, fold_map, offset_distribution,
)
from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D


@pytest.fixture(scope="module")
def survey():
    return OBNGeometry(centre=(1500.0, 1500.0), receiver_spacing=200.0,
                       receiver_extent=1600.0, source_spacing=150.0,
                       source_line_spacing=300.0, source_extent=1600.0).build()


def test_the_node_and_shot_grids_have_the_requested_spacing(survey):
    nodes = np.unique(survey.receivers[:, 0])
    assert np.allclose(np.diff(nodes), 200.0)
    assert survey.n_receivers == 9 * 9
    assert survey.n_traces == survey.n_sources * survey.n_receivers


def test_offsets_and_azimuths_are_consistent(survey):
    off = survey.offsets()
    assert off.shape == (survey.n_sources, survey.n_receivers)
    assert off.min() >= 0.0
    az = survey.azimuths()
    assert az.min() >= 0.0 and az.max() < 360.0


def test_obn_gives_full_azimuth_coverage(survey):
    """The reason OBN is the first geometry: every sector is populated."""
    _, counts = azimuth_distribution(survey, n_bins=12, min_offset=200.0)
    assert np.all(counts > 0)
    assert counts.max() / counts.min() < 2.0


def test_offset_distribution_spans_the_survey(survey):
    centres, counts = offset_distribution(survey, n_bins=10)
    assert counts.sum() == survey.n_traces
    assert centres[-1] > 1500.0


def test_fold_peaks_near_the_survey_centre(survey):
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (50, 50, 20))
    xs, ys, fold = fold_map(survey, grid, depth=1250.0, bin_size=100.0)
    peak = np.unravel_index(int(np.argmax(fold)), fold.shape)
    assert abs(xs[peak[0]] - 1500.0) < 250.0
    assert abs(ys[peak[1]] - 1500.0) < 250.0


def test_positions_outside_the_grid_are_reported_not_clipped(survey):
    small = Grid3D.from_bounds(((1200, 1800), (1200, 1800), (0, 1000)), (50, 50, 50))
    assert not survey.within(small)
    assert len(survey.outside(small)) > 0


def test_decimation_is_explicit(survey):
    thinned = survey.decimated(source_step=2)
    assert thinned.n_sources == (survey.n_sources + 1) // 2
    assert thinned.n_receivers == survey.n_receivers
    assert "decimated" in thinned.name
    with pytest.raises(ConfigError, match="decimation steps"):
        survey.decimated(source_step=0)


def test_bad_geometry_is_rejected():
    with pytest.raises(ConfigError, match="must be positive"):
        OBNGeometry(receiver_spacing=0.0)
    with pytest.raises(ConfigError, match=r"shape \(n, 3\)"):
        Acquisition(sources=np.zeros((4, 2)), receivers=np.zeros((4, 3)))
