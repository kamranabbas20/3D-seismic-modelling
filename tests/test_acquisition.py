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


# ------------------------------------------------- geometry diagnostics
def _square_survey(extent=1000.0, n=5, depth=540.0):
    from sim3d.acquisition.geometry import Acquisition
    a = np.linspace(-extent / 2, extent / 2, n) + 1000.0
    X, Y = np.meshgrid(a, a, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, depth)])
    return Acquisition(sources=pts, receivers=pts)


def test_a_wider_spread_subtends_a_wider_aperture():
    """The number that decides whether a reflector images as an event or arcs."""
    from sim3d.acquisition.geometry import sampling_report
    narrow = sampling_report(_square_survey(500.0), 1200.0, 2400.0, 28.8)
    wide = sampling_report(_square_survey(1900.0), 1200.0, 2400.0, 28.8)
    assert wide.aperture_deg > narrow.aperture_deg
    assert narrow.aperture_deg == pytest.approx(
        np.degrees(np.arctan2(np.hypot(250.0, 250.0), 660.0)), abs=0.1)


def test_the_standoff_is_reported_in_wavelengths():
    """Below about two, the injection near-field overlaps the target."""
    from sim3d.acquisition.geometry import sampling_report
    r = sampling_report(_square_survey(depth=540.0), 1200.0, 2400.0, 12.0)
    assert r.standoff == pytest.approx(660.0)
    assert r.wavelength == pytest.approx(200.0)
    assert r.standoff_wavelengths == pytest.approx(3.3)


def test_the_operator_limit_tightens_with_frequency_and_aperture():
    """Raising the frequency for resolution makes the sampling demand harder,
    which is the trap: it goes as 1 / fmax."""
    from sim3d.acquisition.geometry import sampling_report
    low = sampling_report(_square_survey(), 1200.0, 2400.0, 20.0)
    high = sampling_report(_square_survey(), 1200.0, 2400.0, 40.0)
    assert high.operator_limit == pytest.approx(low.operator_limit / 2, rel=1e-6)
    wide = sampling_report(_square_survey(2000.0), 1200.0, 2400.0, 20.0)
    assert wide.operator_limit < low.operator_limit


def test_a_narrow_aperture_and_coarse_sampling_are_both_reported():
    from sim3d.acquisition.geometry import sampling_report
    r = sampling_report(_square_survey(300.0), 1200.0, 2400.0, 28.8,
                        source_spacing=400.0, receiver_spacing=400.0)
    notes = " ".join(r.notes())
    assert "aperture" in notes
    assert "source spacing" in notes and "receiver spacing" in notes


def test_a_healthy_geometry_reports_nothing():
    from sim3d.acquisition.geometry import sampling_report
    r = sampling_report(_square_survey(1900.0), 1200.0, 2400.0, 28.8,
                        source_spacing=20.0, receiver_spacing=20.0)
    assert r.notes() == []


def test_the_geometry_survives_an_unknown_velocity():
    """Aperture and standoff are geometry; they must not need a rock physics run.

    The acquisition page draws them before any flow simulation exists, so a
    NaN velocity has to fall through to NaN wavelengths rather than raise or,
    worse, report a comfortable 0x sampling factor.
    """
    from sim3d.acquisition.geometry import sampling_report
    report = sampling_report(_square_survey(1900.0), 1200.0, float("nan"), 28.8,
                             source_spacing=400.0, receiver_spacing=100.0)
    assert report.aperture_deg > 0.0 and np.isfinite(report.standoff)
    assert np.isnan(report.wavelength)
    assert np.isnan(report.operator_limit)
    assert np.isnan(report.factor(400.0))          # not 0.0: unknown, not fine
    assert report.notes() == []                    # nothing evaluable to warn about
