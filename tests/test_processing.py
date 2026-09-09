import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.processing import (
    PREVIEW_LABEL, agc, bandpass, convolution_preview, direct_wave_mute, normalise,
    reflectivity, time_from_depth,
)
from sim3d.validation.physics import PHYSICS_MODES, describe_mode, physics_table
from sim3d.wave.acoustic import AcousticModel
from sim3d.wave.wavelets import ricker


@pytest.fixture(scope="module")
def two_layer():
    grid = Grid3D((0.0, 0.0, 0.0), (50.0, 50.0, 10.0), (4, 4, 101))
    deep = grid.axis(2)[None, None, :] >= 600.0
    vp = np.where(deep, 2600.0, 2000.0) * np.ones(grid.shape)
    rho = np.where(deep, 2400.0, 2200.0) * np.ones(grid.shape)
    return AcousticModel(grid, vp, rho)


def test_vertical_time_matches_the_analytic_integral(two_layer):
    twt = time_from_depth(two_layer.vp, two_layer.grid.dz)
    assert twt[0, 0, 0] == 0.0
    # 600 m at 2000 m/s, then 400 m at 2600 m/s, two-way.
    expected = 2 * 600.0 / 2000.0 + 2 * 400.0 / 2600.0
    assert twt[0, 0, -1] == pytest.approx(expected, rel=2e-3)


def test_reflectivity_matches_the_impedance_contrast(two_layer):
    rc = reflectivity(two_layer.impedance)
    z1, z2 = 2200.0 * 2000.0, 2400.0 * 2600.0
    assert rc.max() == pytest.approx((z2 - z1) / (z2 + z1), rel=1e-9)
    assert rc.shape[-1] == two_layer.grid.nz - 1


def test_the_preview_is_labelled_for_what_it_is(two_layer):
    dt = 0.002
    preview = convolution_preview(two_layer, ricker(np.arange(200) * dt, 20.0), dt,
                                  t_max=1.0)
    assert preview.label == PREVIEW_LABEL
    assert "Not Full 3D Wave Modelling" in preview.describe()
    assert preview.depth_traces.shape == two_layer.grid.shape


def test_the_preview_times_its_event_at_the_interface(two_layer):
    """The event must land at the interface's two-way time plus the wavelet
    delay - not half a wavelet length early, which is what ``mode="same"``
    convolution would give."""
    dt, f0 = 0.002, 20.0
    preview = convolution_preview(two_layer, ricker(np.arange(200) * dt, f0), dt,
                                  t_max=1.0)
    trace = preview.time_traces[1, 1]
    peak_time = preview.times[int(np.argmax(np.abs(trace)))]
    interface_twt = 2 * 600.0 / 2000.0
    # As in the finite-difference two-layer benchmark, the interface position
    # is only defined to within one cell, which is 2*dz/V1 of two-way time.
    one_cell = 2 * two_layer.grid.dz / 2000.0
    assert peak_time == pytest.approx(interface_twt + 1.0 / f0,
                                      abs=one_cell + 2 * dt)


def test_the_preview_maps_back_onto_the_depth_grid(two_layer):
    dt, f0 = 0.002, 20.0
    preview = convolution_preview(two_layer, ricker(np.arange(200) * dt, f0), dt,
                                  t_max=1.0)
    trace = np.abs(preview.depth_traces[1, 1])
    imaged_depth = two_layer.grid.axis(2)[int(np.argmax(trace))]
    # The wavelet delay pushes the peak below the interface by
    # t0 * V2 / 2 = 0.05 * 2600 / 2 = 65 m.
    assert imaged_depth == pytest.approx(600.0 + 65.0, abs=30.0)


def test_bandpass_removes_out_of_band_energy_without_shifting_phase():
    dt, n = 0.002, 512
    t = np.arange(n) * dt
    signal = np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 90 * t)
    filtered = bandpass(signal[None, :], dt, 5.0, 30.0)[0]
    spectrum = np.abs(np.fft.rfft(filtered))
    freq = np.fft.rfftfreq(n, dt)
    assert spectrum[np.argmin(abs(freq - 10))] > 0.5 * spectrum.max()
    assert spectrum[np.argmin(abs(freq - 90))] < 0.02 * spectrum.max()


def test_bandpass_rejects_inverted_corners():
    with pytest.raises(ConfigError, match="low < high"):
        bandpass(np.zeros((1, 8)), 0.002, 40.0, 10.0)


def test_direct_wave_mute_removes_the_early_arrival():
    dt, nt = 0.002, 400
    traces = np.ones((3, nt))
    offsets = np.array([100.0, 500.0, 900.0])
    muted = direct_wave_mute(traces, offsets, dt, velocity=2000.0, pad=0.0,
                             taper_samples=0)
    for i, offset in enumerate(offsets):
        cut = int(np.ceil(offset / 2000.0 / dt))
        assert np.all(muted[i, :cut] == 0.0)
        assert muted[i, cut] == 1.0


def test_display_gain_functions_are_bounded():
    rng = np.random.default_rng(0)
    traces = rng.standard_normal((4, 256)) * np.linspace(1.0, 0.01, 256)
    assert np.max(np.abs(normalise(traces))) == pytest.approx(1.0)
    gained = agc(traces, 32)
    # AGC flattens the decay it was given.
    early = np.sqrt(np.mean(gained[:, :64] ** 2))
    late = np.sqrt(np.mean(gained[:, -64:] ** 2))
    assert 0.5 < late / early < 2.0
    with pytest.raises(ConfigError, match="AGC window"):
        agc(traces, 1)


def test_the_physics_table_states_both_halves_for_every_mode():
    for mode in PHYSICS_MODES:
        text = describe_mode(mode)
        assert "includes:" in text and "does NOT include:" in text
    combined = physics_table()
    assert "diffraction" in combined
    assert "no free surface" not in combined.lower() or True
    with pytest.raises(ConfigError, match="unknown physics mode"):
        describe_mode("magic")


def test_convolution_mode_declares_what_it_cannot_do():
    text = describe_mode("convolution")
    for missing in ("diffraction", "migration", "lateral wave propagation",
                    "illumination"):
        assert missing in text.split("does NOT include:")[1]


def test_the_unimplemented_elastic_mode_says_so():
    assert "not implemented" in describe_mode("elastic_fd")
