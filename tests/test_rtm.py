"""RTM validation (spec section 129): does migrated energy land where it should?"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.imaging.rtm import RTMSettings, laplacian, migrate_survey
from sim3d.imaging.store import WavefieldStore, safe_decimation
from sim3d.wave.acoustic import (
    AcousticModel, AcousticSolver, ShotRecord, SolverSettings, steps_for_duration,
)
from sim3d.wave.cpml import PMLSettings
from sim3d.wave.fdscheme import max_stable_dt
from sim3d.wave.sources import PointSource
from sim3d.wave.wavelets import ricker

V0, RHO0, F0, DX = 2000.0, 2200.0, 12.0, 14.0
SHAPE = (51, 51, 51)
Z_ACQ = 196.0


def _experiment(perturb):
    """Build background and perturbed models sharing one time step.

    Baseline and monitor must share ``dt`` or their gathers sit on
    different time axes and their difference is meaningless.
    """
    grid = Grid3D((0.0, 0.0, 0.0), (DX,) * 3, SHAPE)
    vp = np.full(SHAPE, V0)
    vp_pert = perturb(vp.copy(), grid)
    dt = max_stable_dt(grid.spacing, max(vp.max(), vp_pert.max()), 8, safety=0.9)
    settings = SolverSettings(pml=PMLSettings(n_nodes=10), dtype=np.float32, dt=dt)
    rho = np.full(SHAPE, RHO0)
    return (grid, settings,
            AcousticModel(grid, vp, rho, name="background"),
            AcousticModel(grid, vp_pert, rho, name="perturbed"))


def _scattered_survey(grid, settings, background, perturbed, sources, receivers, duration):
    """Model both earths and keep the difference: the scattered field only."""
    nt = steps_for_duration(duration, settings.dt)
    wavelet = ricker(np.arange(nt) * settings.dt, F0)
    bg_solver = AcousticSolver(background, settings, f0=F0)
    pt_solver = AcousticSolver(perturbed, settings, f0=F0)
    records = []
    for src in sources:
        a = bg_solver.run(PointSource(grid, src, wavelet), nt, receivers=receivers)
        b = pt_solver.run(PointSource(grid, src, wavelet), nt, receivers=receivers)
        records.append(ShotRecord(traces=b.traces - a.traces, dt=a.dt,
                                  receiver_positions=a.receiver_positions,
                                  source_position=src))
    return records, wavelet


def _acquisition():
    rx, ry = np.meshgrid(np.arange(175.0, 526.0, 70.0),
                         np.arange(175.0, 526.0, 70.0), indexing="ij")
    receivers = np.column_stack([rx.ravel(), ry.ravel(), np.full(rx.size, Z_ACQ)])
    sources = [(266.0, 266.0, Z_ACQ), (434.0, 434.0, Z_ACQ)]
    return sources, receivers


def test_safe_decimation_respects_the_correlation_nyquist():
    assert safe_decimation(1e-3, 50.0) == 5  # dt_save <= 1/(4*50) = 5 ms
    assert safe_decimation(2e-3, 250.0) == 1
    with pytest.raises(ConfigError):
        safe_decimation(0.0, 50.0)


def test_wavefield_store_round_trips_and_reports_its_backing():
    with WavefieldStore((4, 4, 4), 3, dtype=np.float32) as store:
        assert store.backing == "memory"
        field = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        store.save(1, field)
        assert np.array_equal(store[1], field)
    with WavefieldStore((8, 8, 8), 4, dtype=np.float32, max_ram_bytes=1) as store:
        assert store.backing == "memmap"
        store.save(0, np.ones((8, 8, 8), dtype=np.float32))
        store.flush()
        assert store[0].sum() == 8**3


def test_stores_sharing_a_directory_do_not_collide(tmp_path):
    """Several shots in one run share a workdir.

    The backing files must be distinct and must be gone once closed:
    Windows refuses to delete or reopen a file that is still mapped, so a
    collision or a leaked mapping would fail on the second shot rather than
    the first.
    """
    paths = []
    for value in range(3):
        with WavefieldStore((8, 8, 8), 4, dtype=np.float32, max_ram_bytes=1,
                            directory=str(tmp_path)) as store:
            assert store.backing == "memmap"
            store.save(0, np.full((8, 8, 8), float(value), dtype=np.float32))
            store.flush()
            assert store[0][0, 0, 0] == value
            paths.append(store.path)
    assert len(set(paths)) == 3
    assert list(tmp_path.iterdir()) == []


def test_a_temporary_store_removes_its_own_directory():
    store = WavefieldStore((8, 8, 8), 4, dtype=np.float32, max_ram_bytes=1)
    path = store.path
    assert path.exists()
    store.close()
    assert not path.exists()
    store.close()  # idempotent


def test_laplacian_matches_the_analytic_second_derivative():
    """Second-order accurate, so the error must sit within the ``d^2/12`` bound."""
    n, d = 40, 0.5
    x = np.arange(n) * d
    field = np.sin(x)[:, None, None] * np.ones((n, 3, 3))
    lap = laplacian(field, (d, d, d))
    error = np.max(np.abs(lap[5:-5, 1, 1] + np.sin(x)[5:-5]))
    assert error <= 1.05 * d**2 / 12.0
    assert error > 0.5 * d**2 / 12.0  # and it really is second order, not exact


def test_point_diffractor_energy_collapses_to_its_true_position():
    """The mandatory RTM benchmark: a point scatterer must image at a point."""
    target = (350.0, 350.0, 406.0)

    def perturb(vp, grid):
        i, j, k = grid.nearest_index(target)
        vp[i:i + 2, j:j + 2, k:k + 2] *= 1.10
        return vp

    grid, settings, background, perturbed = _experiment(perturb)
    sources, receivers = _acquisition()
    records, wavelet = _scattered_survey(grid, settings, background, perturbed,
                                         sources, receivers, duration=0.62)

    result = migrate_survey(records, background, wavelet, sources,
                            solver_settings=settings,
                            rtm=RTMSettings(imaging_condition="crosscorrelation",
                                            laplacian_filter=False),
                            fmax=3.0 * F0, f0=F0)

    energy = result.image.astype(np.float64) ** 2
    # Exclude the boundary layer and the band around the acquisition plane,
    # where the injection points are singular by construction.
    mask = np.zeros_like(energy, dtype=bool)
    k_below = int((Z_ACQ + 100.0) / DX)
    mask[12:-12, 12:-12, k_below:-12] = True

    peak = np.unravel_index(int(np.argmax(np.where(mask, energy, 0.0))), energy.shape)
    located = np.array([grid.origin[a] + peak[a] * grid.spacing[a] for a in range(3)])
    wavelength = V0 / F0
    assert np.linalg.norm(located - np.array(target)) < 0.35 * wavelength

    # Focused, not smeared: a large share of the energy inside a small share
    # of the volume.  A sphere of 0.3 wavelengths holds about 1.4% of the
    # analysed volume, so an unfocused image would put ~1.4% of the energy there.
    x, y, z = np.meshgrid(*[grid.axis(a) for a in range(3)], indexing="ij")
    radius = np.sqrt((x - target[0]) ** 2 + (y - target[1]) ** 2 + (z - target[2]) ** 2)
    near = mask & (radius < 0.3 * wavelength)
    energy_fraction = energy[near].sum() / energy[mask].sum()
    volume_fraction = near.sum() / mask.sum()
    assert energy_fraction > 0.40
    assert energy_fraction / volume_fraction > 15.0


def test_flat_reflector_images_at_the_correct_depth():
    z_reflector = 406.0

    def perturb(vp, grid):
        vp[:, :, grid.axis(2) >= z_reflector] *= 1.10
        return vp

    grid, settings, background, perturbed = _experiment(perturb)
    sources, receivers = _acquisition()
    records, wavelet = _scattered_survey(grid, settings, background, perturbed,
                                         sources, receivers, duration=0.62)

    result = migrate_survey(records, background, wavelet, sources,
                            solver_settings=settings,
                            rtm=RTMSettings(imaging_condition="source_normalized"),
                            fmax=3.0 * F0, f0=F0)

    core = result.image[16:-16, 16:-16, :].astype(np.float64) ** 2
    k_below = int((Z_ACQ + 100.0) / DX)
    profile = core.sum(axis=(0, 1))
    profile[:k_below] = 0.0
    profile[-12:] = 0.0
    imaged_depth = grid.axis(2)[int(np.argmax(profile))]
    assert imaged_depth == pytest.approx(z_reflector, abs=0.25 * V0 / F0)


def test_a_coarser_imaging_stride_is_refused_rather_than_aliased():
    grid, settings, background, _ = _experiment(lambda vp, g: vp)
    sources, receivers = _acquisition()
    nt = 40
    wavelet = ricker(np.arange(nt) * settings.dt, F0)
    record = ShotRecord(traces=np.zeros((len(receivers), nt), dtype=np.float32),
                        dt=settings.dt, receiver_positions=receivers,
                        source_position=sources[0])
    with pytest.raises(ConfigError, match="aliases the imaging condition"):
        migrate_survey([record], background, wavelet, sources[:1],
                       solver_settings=settings,
                       rtm=RTMSettings(time_decimation=999), fmax=3.0 * F0, f0=F0)


def test_a_shot_recorded_at_a_different_dt_is_refused():
    grid, settings, background, _ = _experiment(lambda vp, g: vp)
    sources, receivers = _acquisition()
    record = ShotRecord(traces=np.zeros((len(receivers), 20), dtype=np.float32),
                        dt=settings.dt * 1.3, receiver_positions=receivers,
                        source_position=sources[0])
    with pytest.raises(ConfigError, match="must share a time step"):
        migrate_survey([record], background, ricker(np.arange(20) * settings.dt, F0),
                       sources[:1], solver_settings=settings, fmax=3.0 * F0, f0=F0)


# ------------------------------------------------- image polarity and taper
def test_the_artefact_filter_keeps_reflectivity_polarity():
    """A Laplacian inverts a band-limited peak: its second derivative is a
    trough. Filtering with the bare operator gives an image in which every
    hard event reads soft, which on the three-layer model showed up as a
    correlation of -0.71 against the band-limited reference.
    """
    from sim3d.imaging.rtm import RTMSettings, filtered_image

    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(10.0, 10.0, 10.0),
                  shape=(9, 9, 41))
    z = np.arange(41) * 10.0
    lobe = np.exp(-((z - 200.0) / 40.0) ** 2)          # a positive reflector
    image = np.zeros(grid.shape) + lobe
    out, notes = filtered_image(image, grid, RTMSettings(taper_wavelengths=0.0))
    assert out[4, 4, 20] > 0.0
    assert any("polarity" in n for n in notes)


def test_the_taper_is_zero_at_an_acquisition_point_and_one_beyond_it():
    from sim3d.imaging.rtm import acquisition_taper

    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(10.0, 10.0, 10.0),
                  shape=(21, 21, 21))
    taper = acquisition_taper(grid, [(100.0, 100.0, 100.0)], radius=50.0)
    assert taper[10, 10, 10] == pytest.approx(0.0)
    assert taper[0, 0, 0] == pytest.approx(1.0)
    assert np.all((taper >= 0.0) & (taper <= 1.0))
    # monotonic away from the point along a row
    row = taper[10:, 10, 10]
    assert np.all(np.diff(row[:6]) >= -1e-12)


def test_a_zero_radius_taper_leaves_the_image_alone():
    from sim3d.imaging.rtm import acquisition_taper

    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(10.0, 10.0, 10.0),
                  shape=(5, 5, 5))
    assert np.all(acquisition_taper(grid, [(20.0, 20.0, 20.0)], 0.0) == 1.0)


def test_the_taper_radius_comes_from_wavelengths_or_an_override():
    from sim3d.imaging.rtm import RTMSettings

    assert RTMSettings(taper_wavelengths=1.5).acquisition_taper_radius(100.0) == 150.0
    assert RTMSettings(taper_radius=42.0).acquisition_taper_radius(100.0) == 42.0
    assert RTMSettings(taper_wavelengths=0.0).acquisition_taper_radius(100.0) == 0.0
    with pytest.raises(ConfigError, match="taper_wavelengths"):
        RTMSettings(taper_wavelengths=-1.0)


def test_the_taper_suppresses_the_acquisition_zone_without_touching_the_target():
    """The point of the taper: it must remove the injection near-field and
    leave a deeper reflector untouched."""
    from sim3d.imaging.rtm import RTMSettings, filtered_image

    grid = Grid3D(origin=(0.0, 0.0, 0.0), spacing=(20.0, 20.0, 20.0),
                  shape=(11, 11, 41))
    image = np.zeros(grid.shape)
    image[5, 5, 5] = 50.0            # near-field spike at the source, z = 100 m
    image[:, :, 30] = 1.0            # a reflector at z = 600 m
    out, notes = filtered_image(
        image, grid, RTMSettings(laplacian_filter=False, taper_radius=100.0),
        points=[(100.0, 100.0, 100.0)], wavelength=None)
    assert out[5, 5, 5] == pytest.approx(0.0)
    assert out[5, 5, 30] == pytest.approx(1.0)
    assert any("tapered" in n for n in notes)


def test_the_correlation_window_starts_when_the_target_can_first_reflect():
    """Correlating before that can only accumulate injection near-field."""
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline

    cfg = ExperimentConfig()
    cfg.domains.geology_bounds = [[0, 1200], [0, 1200], [0, 1400]]
    cfg.domains.geology_spacing = [50.0, 50.0, 25.0]
    cfg.domains.propagation_bounds = [[100, 1100], [100, 1100], [400, 1400]]
    cfg.domains.propagation_spacing = [50.0, 50.0, 50.0]
    cfg.domains.target_bounds = [[400, 800], [400, 800], [900, 1100]]
    pipe = Pipeline(cfg)

    class FakeAcq:
        sources = np.array([[500.0, 500.0, 700.0]])

    class FakeModel:
        vp = np.full((2, 2, 2), 2000.0)

    # (900 - 700) * 2 / 2000 = 0.2 s
    assert pipe.correlation_start_time(FakeModel(), FakeAcq()) == pytest.approx(0.2)

    cfg.imaging.correlation_start_time = 0.0
    assert Pipeline(cfg).correlation_start_time(FakeModel(), FakeAcq()) == 0.0
    cfg.imaging.correlation_start_time = 0.42
    assert Pipeline(cfg).correlation_start_time(FakeModel(), FakeAcq()) == pytest.approx(0.42)


def test_a_negative_correlation_start_is_rejected():
    from sim3d.imaging.rtm import RTMSettings
    with pytest.raises(ConfigError, match="correlation_start_time"):
        RTMSettings(correlation_start_time=-0.1)
