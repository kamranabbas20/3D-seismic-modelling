"""4D time shifts: the true one, the measurable one, and the alignment.

A shifted wavelet minus itself is a derivative-shaped difference that can
dwarf the amplitude change it gets mistaken for, so the numbers that
matter here are how much NRMS a pure shift manufactures and how much of
it alignment takes back.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.fourd.metrics import nrms
from sim3d.fourd.timeshift import (
    align, estimated_time_shift, resample_shift, true_time_shift,
)
from sim3d.wave.wavelets import ricker

DT = 0.002
NT = 600


@pytest.fixture
def traces():
    """A reflectivity series convolved to a wavelet, as a small cube."""
    rng = np.random.default_rng(1)
    rc = np.zeros((6, 6, NT))
    rc[..., ::37] = rng.standard_normal(rc[..., ::37].shape)
    w = ricker(np.arange(200) * DT, 25.0)
    return np.apply_along_axis(lambda x: np.convolve(x, w, "same"), -1, rc)


@pytest.fixture
def ramp():
    """Zero above 0.5 s, ramping to 6 ms by 0.8 s, flat below - the shape a
    softening reservoir puts on everything beneath it."""
    t = np.arange(NT) * DT
    return np.clip((t - 0.5) / 0.3, 0.0, 1.0) * 0.006


def shifted(cube, shift):
    t = np.arange(cube.shape[-1]) * DT
    out = np.empty_like(cube)
    for i in np.ndindex(cube.shape[:-1]):
        out[i] = np.interp(t - shift, t, cube[i])
    return out


# ------------------------------------------------------------------- true
def test_the_true_shift_comes_straight_from_the_two_way_times():
    """A forward model knows the answer; estimating it would be a choice to
    be less accurate."""
    nz, dz = 200, 10.0
    base_vp = np.full((2, 2, nz), 2000.0)
    mon_vp = base_vp.copy()
    mon_vp[..., 100:] = 1900.0                       # slower below 1000 m
    from sim3d.processing.preview import time_from_depth
    twt_b = time_from_depth(base_vp, dz)
    twt_m = time_from_depth(mon_vp, dz)
    times = np.arange(0.0, 2.2, DT)
    shift = true_time_shift(twt_b, twt_m, times)
    assert shift[..., times < 0.9].max() == pytest.approx(0.0, abs=1e-6)
    # 100 layers of 10 m at 1900 instead of 2000: 2*1000*(1/1900 - 1/2000)
    expected = 2.0 * 1000.0 * (1.0 / 1900.0 - 1.0 / 2000.0)
    assert shift[..., -1].mean() == pytest.approx(expected, rel=0.02)


def test_a_shift_is_positive_when_the_monitor_is_late():
    """Positive means slowdown, the classic softening signature - a sign
    slip here inverts every interpretation downstream."""
    nz, dz = 100, 10.0
    fast = np.full((1, 1, nz), 2000.0)
    slow = np.full((1, 1, nz), 1800.0)
    from sim3d.processing.preview import time_from_depth
    times = np.arange(0.0, 1.5, DT)
    shift = true_time_shift(time_from_depth(fast, dz),
                            time_from_depth(slow, dz), times)
    assert shift[..., -1] > 0.0


def test_mismatched_shapes_are_refused():
    with pytest.raises(ConfigError, match="shapes must match"):
        true_time_shift(np.zeros((2, 2, 10)), np.zeros((2, 2, 9)), np.zeros(5))


# -------------------------------------------------------------- estimated
def test_the_estimate_recovers_a_known_ramp(traces, ramp):
    t = np.arange(NT) * DT
    monitor = shifted(traces, ramp)
    shifts, centres = estimated_time_shift(traces, monitor, DT, window=0.12,
                                           step=0.02, max_shift=0.02)
    want = np.interp(centres, t, ramp)
    got = shifts.mean(axis=(0, 1))
    # Only where a window has something to correlate. The last window sits
    # past the final reflector, and the estimator reports zero there by
    # design - grading it on that would be grading it on data it never had.
    amplitude = np.abs(traces).mean(axis=(0, 1))
    live = np.interp(centres, t, amplitude) > 0.05 * amplitude.max()
    assert live.sum() > 5
    assert np.abs(got - want)[live].max() < 0.001     # within half a sample
    assert got[live][-1] == pytest.approx(0.006, abs=0.0005)


def test_the_estimate_is_time_varying_not_one_number_per_trace(traces, ramp):
    """A whole-trace correlation cannot represent the thing being measured:
    the shift accumulates with depth and is flat above the change."""
    shifts, centres = estimated_time_shift(traces, shifted(traces, ramp), DT)
    profile = shifts.mean(axis=(0, 1))
    assert centres.size > 10
    assert profile.max() - profile.min() > 0.004


def test_a_dead_window_measures_nothing_rather_than_guessing():
    """The time axis runs from the surface and the model does not, so every
    trace has dead ends; left to guess they return the search range."""
    cube = np.zeros((2, 2, NT))
    cube[..., 200:260] = ricker(np.arange(60) * DT, 25.0)
    shifts, centres = estimated_time_shift(cube, cube, DT, max_shift=0.03)
    assert np.allclose(shifts[..., centres < 0.2], 0.0)


def test_identical_volumes_have_no_shift(traces):
    shifts, _ = estimated_time_shift(traces, traces, DT)
    assert np.abs(shifts).max() < 1e-9


# ---------------------------------------------------------------- align
def test_a_pure_time_shift_manufactures_nrms_that_alignment_takes_back(traces, ramp):
    """The number that justifies the whole module: 6 ms of shift, no
    amplitude change at all, and the raw difference is enormous."""
    t = np.arange(NT) * DT
    monitor = shifted(traces, ramp)
    raw = nrms(monitor, traces)
    assert raw > 40.0                                 # a shift alone does this

    shifts, centres = estimated_time_shift(traces, monitor, DT, max_shift=0.02)
    full = resample_shift(shifts, centres, t)
    assert nrms(align(monitor, full, DT), traces) < 0.2 * raw


def test_aligning_with_the_true_shift_is_at_least_as_good(traces, ramp):
    t = np.arange(NT) * DT
    monitor = shifted(traces, ramp)
    exact = np.broadcast_to(ramp, monitor.shape).copy()
    assert nrms(align(monitor, exact, DT), traces) < nrms(monitor, traces)


def test_aligning_by_nothing_changes_nothing(traces):
    zero = np.zeros_like(traces)
    assert np.allclose(align(traces, zero, DT), traces)


def test_align_needs_a_shift_per_sample(traces):
    with pytest.raises(ConfigError, match="a shift per sample"):
        align(traces, np.zeros(traces.shape[:-1]), DT)
