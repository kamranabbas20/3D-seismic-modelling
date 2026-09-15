"""Survey noise and 4D non-repeatability for the forward model.

A noise-free synthetic pair has an NRMS of zero, which is not a good
result - it is a meaningless one.  Field 4D NRMS runs from about 5% on an
excellent permanent installation to 30% or worse on a repeated streamer
survey, and most of that is *non-repeatability*: noise that differs
between the two surveys and therefore survives the difference.  Without a
term for it every 4D number this tool reports is incomparable with a real
one, and the anomaly-detectability question - can this flood be seen? -
has no meaning at all.

The model has two knobs and one honest relationship between them.

``level``
    Noise RMS as a fraction of the volume's own signal RMS.
``repeatability``
    How much of that noise is *shared* between surveys, from 0 (nothing
    repeats; every survey is independently noisy) to 1 (perfectly
    repeatable; the noise is identical and cancels in the difference).

Each survey gets ``sqrt(r) * shared + sqrt(1 - r) * its own``, so the
noise level of any single survey is ``level`` regardless of ``r`` - only
the difference changes.  That makes the NRMS floor predictable:

.. math:: \\mathrm{NRMS}_{\\text{noise}} = 100 \\,\\text{level} \\sqrt{2 (1 - r)}

so 10% noise with no repeatability lands at about 14% NRMS before the
reservoir has changed at all.  Anything the 4D reports below that floor is
noise, and the floor is reported alongside the metric rather than left for
the reader to work out.

The noise is band-limited to the signal's own passband.  White noise is
not what a seismic volume contains and it is trivially removable, so
adding it would understate the problem rather than model it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..processing.filters import bandpass


@dataclass(frozen=True)
class NoiseModel:
    """Survey noise as a fraction of signal, and how much of it repeats."""

    #: Noise RMS as a fraction of the volume's signal RMS.  0 disables it.
    level: float = 0.0
    #: Shared fraction, 0 (independent every survey) to 1 (identical).
    repeatability: float = 0.0
    #: Passband of the noise, Hz.  ``None`` takes the source bandwidth.
    band: tuple[float, float] | None = None
    seed: int = 1

    def __post_init__(self) -> None:
        if self.level < 0.0:
            raise ConfigError(f"noise level must not be negative, got {self.level}")
        if not 0.0 <= self.repeatability <= 1.0:
            raise ConfigError(
                f"repeatability is a fraction from 0 to 1, got {self.repeatability}")
        if self.band is not None and not 0 <= self.band[0] < self.band[1]:
            raise ConfigError(f"noise band must be (low, high), got {self.band}")

    @property
    def active(self) -> bool:
        return self.level > 0.0

    @property
    def nrms_floor(self) -> float:
        """NRMS between two surveys of identical signal, from noise alone.

        The number every 4D measurement in the experiment has to beat to
        mean anything.  In percent, matching :func:`sim3d.fourd.metrics.nrms`
        - two conventions for one quantity in one codebase is how a 14%
        floor gets compared against a 0.14 measurement.
        """
        return float(100.0 * self.level * np.sqrt(2.0 * (1.0 - self.repeatability)))

    def describe(self) -> str:
        if not self.active:
            return "no survey noise; NRMS has no floor and is not comparable to field data"
        return (f"noise {100 * self.level:.0f}% of signal RMS, repeatability "
                f"{100 * self.repeatability:.0f}% → NRMS floor "
                f"{self.nrms_floor:.1f}%")


def _band_limited(shape, dt: float, band, rng) -> np.ndarray:
    noise = rng.standard_normal(shape)
    if band is not None:
        noise = bandpass(noise, dt, float(band[0]), float(band[1]))
    rms = float(np.sqrt(np.mean(noise**2)))
    return noise / rms if rms > 0 else noise


def add_survey_noise(cubes: dict, dt: float, model: NoiseModel,
                     band: tuple[float, float] | None = None) -> dict:
    """Add correlated-between-surveys noise to one cube per survey.

    ``cubes`` maps a survey name to an array whose last axis is time.  The
    scale is set by the *pooled* signal RMS rather than each survey's own,
    so a monitor that happens to be dimmer does not quietly receive less
    noise and flatter itself in the difference.
    """
    if not model.active or not cubes:
        return {name: np.asarray(cube) for name, cube in cubes.items()}
    arrays = {name: np.asarray(cube, dtype=float) for name, cube in cubes.items()}
    shapes = {a.shape for a in arrays.values()}
    if len(shapes) != 1:
        raise ConfigError(
            f"every survey must be on the same axes to share a noise "
            f"realisation, got {sorted(shapes)}")
    shape = shapes.pop()

    signal_rms = float(np.sqrt(np.mean(
        np.concatenate([a.ravel() for a in arrays.values()])**2)))
    if signal_rms == 0.0:
        return arrays
    sigma = model.level * signal_rms
    band = model.band if model.band is not None else band

    rng = np.random.default_rng(model.seed)
    shared = _band_limited(shape, dt, band, rng)
    r = model.repeatability
    out = {}
    for name, array in arrays.items():
        own = _band_limited(shape, dt, band, rng)
        out[name] = array + sigma * (np.sqrt(r) * shared + np.sqrt(1.0 - r) * own)
    return out
