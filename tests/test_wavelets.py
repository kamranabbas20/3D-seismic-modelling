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
