import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.wave.wavelets import (
    amplitude_spectrum, dominant_frequency, ormsby, ricker, ricker_fmax,
)


def test_ricker_peaks_at_its_nominal_frequency():
    dt = 1e-3
    t = np.arange(0.0, 1.0, dt)
    assert dominant_frequency(ricker(t, 20.0), dt) == pytest.approx(20.0, abs=1.0)


def test_ricker_starts_near_zero_with_the_default_delay():
    t = np.arange(0.0, 0.5, 1e-3)
    assert abs(ricker(t, 20.0)[0]) < 1e-3


def test_ricker_matches_its_closed_form():
    t = np.array([0.05])
    a = (np.pi * 20.0 * (0.05 - 0.05)) ** 2
    assert ricker(t, 20.0, t0=0.05)[0] == pytest.approx((1 - 2 * a) * np.exp(-a))


def test_ricker_bandwidth_is_well_above_the_peak_frequency():
    """A 20 Hz Ricker still carries energy near 48 Hz, which is what the
    grid must be sampled for - not for 20 Hz."""
    assert ricker_fmax(20.0, 0.05) == pytest.approx(47.9, abs=1.0)
    assert ricker_fmax(20.0) > 2.0 * 20.0


def test_ormsby_passband_is_flat_between_its_inner_corners():
    dt = 1e-3
    t = np.arange(0.0, 2.0, dt)
    f, s = amplitude_spectrum(ormsby(t, 5, 10, 40, 50), dt)
    s = s / s.max()
    band = s[(f >= 12) & (f <= 38)]
    assert band.min() > 0.95
    assert s[np.argmin(abs(f - 60))] < 0.02
    assert s[np.argmin(abs(f - 2))] < 0.02


def test_bad_frequencies_are_rejected():
    with pytest.raises(ConfigError):
        ricker(np.array([0.0]), -1.0)
    with pytest.raises(ConfigError):
        ormsby(np.array([0.0]), 10, 5, 40, 50)


# ------------------------------------------------------- selecting one
def test_the_configured_wavelet_is_the_one_that_gets_built():
    from sim3d.wave.wavelets import build_wavelet
    t = np.arange(500) * 0.002
    assert not np.allclose(build_wavelet(t, "ricker", frequency=25.0),
                           build_wavelet(t, "ormsby", corners=[5, 10, 40, 50]))


def test_an_ormsby_is_delayed_enough_to_contain_its_tail():
    """Its tails decay only as 1/tau^2. Left at a wavelet's own midpoint
    default - on an array as long as a whole record - the shot would fire
    half a second late; cut too close, and a step at t=0 rings across the
    whole passband.
    """
    from sim3d.wave.wavelets import build_wavelet, ormsby_delay
    dt = 0.002
    t = np.arange(int(1.0 / dt)) * dt
    w = build_wavelet(t, "ormsby", corners=[5, 10, 40, 50])
    assert t[int(np.argmax(np.abs(w)))] == pytest.approx(ormsby_delay(5.0), abs=2 * dt)
    assert abs(w[0]) < 0.02 * np.abs(w).max()


def test_an_ormsby_fmax_is_its_top_corner_exactly():
    """It is band-limited by construction, so asking where a trapezoid has
    fallen to 5% of its peak is asking a question it already answered."""
    from sim3d.wave.wavelets import wavelet_fmax
    assert wavelet_fmax("ormsby", corners=[5, 10, 40, 50]) == 50.0
    # A Ricker is not band-limited at its peak, so its answer is far above it.
    assert wavelet_fmax("ricker", frequency=20.0) > 40.0


def test_an_unknown_wavelet_lists_the_ones_that_exist():
    from sim3d.core.errors import ConfigError
    from sim3d.wave.wavelets import build_wavelet, wavelet_fmax
    with pytest.raises(ConfigError, match="ormsby"):
        build_wavelet(np.arange(10) * 0.002, "klauder")
    with pytest.raises(ConfigError, match="ormsby"):
        wavelet_fmax("klauder")


def test_ormsby_corners_are_validated_where_they_are_configured():
    from sim3d.core.errors import ConfigError
    from sim3d.wave.wavelets import build_wavelet
    t = np.arange(100) * 0.002
    with pytest.raises(ConfigError, match="four corner frequencies"):
        build_wavelet(t, "ormsby", corners=[5, 10, 40])
    with pytest.raises(ConfigError, match="must increase"):
        build_wavelet(t, "ormsby", corners=[5, 40, 10, 50])


def test_choosing_an_ormsby_tightens_the_sample_interval():
    """Fmax feeds the grid and time sampling, so the choice has to reach
    them - a wavelet setting that changed only the trace would be a lie."""
    from sim3d.core.config import ExperimentConfig
    from sim3d.experiments.pipeline import Pipeline
    config = ExperimentConfig.load("examples/configs/three_layer_4d.yaml")
    ricker_fmax_value = Pipeline(config).fmax
    config.source.type = "ormsby"
    config.source.corners = [8.0, 14.0, 45.0, 60.0]
    assert Pipeline(config).fmax == 60.0
    assert Pipeline(config).fmax > ricker_fmax_value
