"""Survey noise and 4D non-repeatability.

The whole value of the model is that the NRMS floor it implies is
predictable, because a floor nobody can compute is a floor nobody
subtracts. So the arithmetic relating level and repeatability to NRMS is
tested against measurement, not just asserted in a docstring.
"""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError
from sim3d.fourd.metrics import nrms, rms
from sim3d.fourd.noise import NoiseModel, add_survey_noise

DT = 0.004


@pytest.fixture
def signal():
    rng = np.random.default_rng(0)
    return rng.standard_normal((16, 16, 256))


def test_the_nrms_floor_predicts_what_is_measured(signal):
    """100 * level * sqrt(2(1-r)), against two surveys of identical signal."""
    for level in (0.05, 0.10, 0.20):
        for r in (0.0, 0.5, 0.9):
            model = NoiseModel(level=level, repeatability=r, band=(5.0, 60.0), seed=3)
            out = add_survey_noise({"a": signal, "b": signal}, DT, model)
            assert nrms(out["a"], out["b"]) == pytest.approx(
                model.nrms_floor, rel=0.05)


def test_the_floor_is_in_percent_like_every_other_metric():
    """Two conventions for one quantity is how a 14% floor gets compared
    against a 0.14 measurement."""
    assert NoiseModel(level=0.1, repeatability=0.0).nrms_floor == pytest.approx(14.14, rel=1e-3)


def test_perfect_repeatability_cancels_in_the_difference(signal):
    model = NoiseModel(level=0.3, repeatability=1.0, seed=1)
    out = add_survey_noise({"a": signal, "b": signal}, DT, model)
    assert np.allclose(out["a"], out["b"])
    assert model.nrms_floor == pytest.approx(0.0)


def test_each_survey_carries_the_same_noise_whatever_repeats(signal):
    """Only the *difference* should depend on r: a survey's own noise level
    is `level` either way, or the knob would secretly be two knobs."""
    levels = []
    for r in (0.0, 0.5, 1.0):
        out = add_survey_noise({"a": signal}, DT,
                               NoiseModel(level=0.2, repeatability=r, seed=7))
        levels.append(rms(out["a"] - signal))
    assert levels[0] == pytest.approx(levels[1], rel=0.05)
    assert levels[1] == pytest.approx(levels[2], rel=0.05)


def test_zero_level_is_exactly_the_input(signal):
    out = add_survey_noise({"a": signal}, DT, NoiseModel(level=0.0))
    assert np.array_equal(out["a"], signal)


def test_the_noise_is_band_limited_not_white(signal):
    """White noise is not what a seismic volume contains and is trivially
    removable, so adding it would understate the problem."""
    model = NoiseModel(level=1.0, repeatability=0.0, band=(10.0, 40.0), seed=2)
    added = add_survey_noise({"a": np.zeros_like(signal)}, DT, model)["a"]
    # A zero-signal volume gets zero noise (nothing to scale against), so
    # scale against a real one and isolate the added part.
    added = add_survey_noise({"a": signal}, DT, model)["a"] - signal
    spectrum = np.abs(np.fft.rfft(added, axis=-1)).mean(axis=(0, 1))
    freq = np.fft.rfftfreq(signal.shape[-1], d=DT)
    inside = spectrum[(freq > 15) & (freq < 35)].mean()
    outside = spectrum[freq > 60].mean()
    assert outside < 0.05 * inside


def test_every_survey_shares_one_realisation_of_the_repeatable_part(signal):
    """Drawing independently per survey would make repeatability a setting
    with no effect, which is the failure this guards."""
    model = NoiseModel(level=0.2, repeatability=0.95, seed=4)
    out = add_survey_noise({"a": signal, "b": signal, "c": signal}, DT, model)
    assert nrms(out["a"], out["b"]) == pytest.approx(model.nrms_floor, rel=0.1)
    assert nrms(out["a"], out["c"]) == pytest.approx(model.nrms_floor, rel=0.1)


def test_surveys_on_different_axes_are_refused(signal):
    with pytest.raises(ConfigError, match="same axes"):
        add_survey_noise({"a": signal, "b": signal[..., :-1]}, DT,
                         NoiseModel(level=0.1))


def test_a_negative_level_and_a_silly_repeatability_are_refused():
    with pytest.raises(ConfigError, match="must not be negative"):
        NoiseModel(level=-0.1)
    with pytest.raises(ConfigError, match="fraction from 0 to 1"):
        NoiseModel(level=0.1, repeatability=1.5)


def test_the_description_says_what_it_costs():
    assert "no survey noise" in NoiseModel().describe()
    assert "NRMS floor" in NoiseModel(level=0.1).describe()
