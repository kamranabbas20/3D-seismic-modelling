"""The sim2seis mode: an earth model converted into a synthetic volume.

Two things carry the weight here. The zero-angle stack must equal the
normal-incidence convolution exactly, or the modes disagree about the same
model; and the angle dependence must reproduce a textbook AVO class, or
the stacks are decoration.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.core.grid import Grid3D
from sim3d.processing.preview import convolution_preview, reflectivity
from sim3d.processing.sim2seis import (
    SIM2SEIS_LABEL, AngleStack, aki_richards, build_stacks, sim2seis_volume,
    stacked_reflectivity,
)
from sim3d.wave.acoustic import AcousticModel
from sim3d.wave.wavelets import ricker

DT = 0.004

# Shale over gas sand - the standard class III AVO case.
SHALE = dict(vp=2380.0, vs=1050.0, rho=2300.0)
GAS_SAND = dict(vp=2050.0, vs=1230.0, rho=2000.0)


@pytest.fixture
def grid():
    return Grid3D(origin=(0.0, 0.0, 0.0), spacing=(50.0, 50.0, 10.0),
                  shape=(9, 9, 61))


@pytest.fixture
def wavelet():
    return ricker(np.arange(int(0.5 / DT)) * DT, 25.0)


def two_layer(grid, upper=SHALE, lower=GAS_SAND, interface=30):
    def field(key):
        a = np.full(grid.shape, upper[key])
        a[:, :, interface:] = lower[key]
        return a
    return field("vp"), field("vs"), field("rho")


def interface_pair(upper=SHALE, lower=GAS_SAND):
    """A single interface as ``(..., 2)`` arrays, for hand calculations."""
    return (np.array([upper["vp"], lower["vp"]]),
            np.array([upper["vs"], lower["vs"]]),
            np.array([upper["rho"], lower["rho"]]))


# ------------------------------------------------------------ reflectivity
def test_zero_angle_is_the_exact_impedance_contrast():
    """Not the linearised intercept: the modes must agree to the last digit."""
    vp, vs, rho = interface_pair()
    assert aki_richards(vp, vs, rho, 0.0) == pytest.approx(reflectivity(rho * vp))


def test_the_class_three_gas_sand_brightens_with_angle():
    """Negative intercept and negative gradient: |R| grows away from zero."""
    vp, vs, rho = interface_pair()
    r = [float(aki_richards(vp, vs, rho, a)[0]) for a in (0.0, 15.0, 30.0)]
    assert r[0] < 0.0
    assert r[0] > r[1] > r[2]                       # steadily more negative
    assert abs(r[2]) > abs(r[0])


def test_the_gradient_matches_the_hand_calculation():
    """Aki-Richards term by term, so a sign slip cannot hide in a stack."""
    vp, vs, rho = interface_pair()
    d_vp, m_vp = vp[1] - vp[0], 0.5 * (vp[0] + vp[1])
    d_vs, m_vs = vs[1] - vs[0], 0.5 * (vs[0] + vs[1])
    d_rho, m_rho = rho[1] - rho[0], 0.5 * (rho[0] + rho[1])
    gradient = (0.5 * d_vp / m_vp
                - 2.0 * (m_vs / m_vp) ** 2 * (d_rho / m_rho + 2.0 * d_vs / m_vs))
    curvature = 0.5 * d_vp / m_vp
    theta = np.deg2rad(30.0)
    expected = (reflectivity(rho * vp)[0]
                + gradient * np.sin(theta) ** 2
                + curvature * (np.tan(theta) ** 2 - np.sin(theta) ** 2))
    assert float(aki_richards(vp, vs, rho, 30.0)[0]) == pytest.approx(expected)


def test_no_contrast_reflects_nothing_at_any_angle():
    same = dict(vp=2500.0, vs=1200.0, rho=2200.0)
    vp, vs, rho = interface_pair(upper=same, lower=same)
    for angle in (0.0, 20.0, 40.0):
        assert aki_richards(vp, vs, rho, angle) == pytest.approx(0.0)


def test_a_shearless_layer_does_not_produce_a_nan():
    """Vs = 0 on both sides is a fluid; the gradient's dVs/Vs is taken as 0."""
    water = dict(vp=1500.0, vs=0.0, rho=1000.0)
    vp, vs, rho = interface_pair(upper=water, lower=water)
    assert np.all(np.isfinite(aki_richards(vp, vs, rho, 30.0)))


def test_mismatched_property_shapes_are_rejected():
    with pytest.raises(ConfigError, match="same shape"):
        aki_richards(np.zeros((4,)), np.zeros((3,)), np.zeros((4,)), 10.0)


def test_a_stack_of_one_angle_is_that_angle():
    vp, vs, rho = interface_pair()
    stack = AngleStack("zero", 0.0, 0.0)
    assert stacked_reflectivity(vp, vs, rho, stack) == pytest.approx(
        aki_richards(vp, vs, rho, 0.0))


def test_a_stack_sits_between_its_end_angles():
    vp, vs, rho = interface_pair()
    stack = AngleStack("far", 30.0, 45.0)
    value = float(stacked_reflectivity(vp, vs, rho, stack)[0])
    ends = sorted(float(aki_richards(vp, vs, rho, a)[0]) for a in (30.0, 45.0))
    assert ends[0] <= value <= ends[1]


# ------------------------------------------------------------------ stacks
def test_the_default_stacks_are_near_mid_and_far():
    assert [s.name for s in build_stacks()] == ["near", "mid", "far"]
    assert build_stacks()[0].centre == pytest.approx(7.5)


def test_a_stack_past_vertical_is_refused():
    with pytest.raises(ConfigError, match="past vertical"):
        build_stacks([{"name": "silly", "angles": [80.0, 95.0]}])


def test_a_reversed_angle_range_is_refused():
    with pytest.raises(ConfigError, match="0 <= min <= max"):
        build_stacks([{"name": "backwards", "angles": [30.0, 10.0]}])


def test_a_stack_without_a_name_says_what_it_needs():
    with pytest.raises(ConfigError, match="'name' and 'angles'"):
        build_stacks([{"angles": [0.0, 10.0]}])


def test_duplicate_stack_names_are_refused():
    with pytest.raises(ConfigError, match="duplicate stack name"):
        build_stacks([{"name": "near", "angles": [0, 10]},
                      {"name": "near", "angles": [10, 20]}])


# ------------------------------------------------------------------ volume
def test_it_returns_one_cube_per_stack(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0)
    assert volume.names == ("near", "mid", "far")
    for name in volume.names:
        assert volume.time_cube(name).shape == (grid.nx, grid.ny, volume.times.size)
        assert volume.depth_cube(name).shape == grid.shape
    assert volume.label == SIM2SEIS_LABEL


def test_the_cubes_are_stored_in_single_precision(grid, wavelet):
    """A full-model cube in double costs twice the memory for unread digits."""
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0)
    assert volume.time_cube("near").dtype == np.float32
    assert volume.megabytes > 0.0


def test_a_zero_angle_stack_equals_the_convolution_preview(grid, wavelet):
    """The cross-mode agreement the exact intercept was chosen to buy."""
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0,
                             stacks=[{"name": "zero", "angles": [0.0, 0.0]}],
                             dtype=np.float64)
    preview = convolution_preview(AcousticModel(grid=grid, vp=vp, rho=rho),
                                  wavelet, DT, t_max=1.0)
    assert np.allclose(volume.time_cube("zero"), preview.time_traces)
    assert np.allclose(volume.depth_cube("zero"), preview.depth_traces)


def test_near_and_far_differ_on_a_model_with_shear_contrast(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0)
    near, far = volume.time_cube("near"), volume.time_cube("far")
    assert not np.allclose(near, far)
    # Class III: the far stack is the brighter one.
    assert np.abs(far).max() > np.abs(near).max()


def test_properties_off_the_grid_are_rejected(grid, wavelet):
    with pytest.raises(ConfigError, match="but the grid is"):
        sim2seis_volume(grid, np.zeros((3, 3, 3)), np.zeros((3, 3, 3)),
                        np.zeros((3, 3, 3)), wavelet, DT)


def test_a_negative_sample_interval_is_rejected(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    with pytest.raises(ConfigError, match="dt must be positive"):
        sim2seis_volume(grid, vp, vs, rho, wavelet, -1.0)


def test_a_wide_stack_is_reported_not_silently_extrapolated(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0,
                             stacks=[{"name": "very_far", "angles": [50.0, 60.0]}])
    assert any("linearisation" in note for note in volume.notes)


def test_the_description_says_what_it_is_not(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    text = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0,
                           scenario="baseline").describe()
    assert "Not Full 3D Wave Modelling" in text
    assert "baseline" in text and "3 angle stacks" in text


def test_an_unknown_stack_name_lists_the_ones_that_exist(grid, wavelet):
    vp, vs, rho = two_layer(grid)
    volume = sim2seis_volume(grid, vp, vs, rho, wavelet, DT, t_max=1.0)
    with pytest.raises(KeyError, match="ultra_far"):
        volume.time_cube("ultra_far")
