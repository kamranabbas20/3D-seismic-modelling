r"""Correlated random property fields (spec section 25).

Properties are never restricted to constants.  Fields here are generated
by the spectral method: build the target covariance on the grid, take its
Fourier transform to get the power spectrum, colour white noise with its
square root, and transform back.  The result has the requested covariance
up to the periodic wrap-around inherent in an FFT, which is why the
correlation lengths should stay well below the domain size.

Supported covariance models, with lag :math:`h` normalised by the
anisotropic correlation lengths:

``gaussian``
    :math:`C(h) = \sigma^2 e^{-h^2}` - very smooth, suitable for gentle
    facies or porosity trends.
``exponential``
    :math:`C(h) = \sigma^2 e^{-h}` - rougher, closer to what well logs show.
``spherical``
    :math:`C(h) = \sigma^2 (1 - 1.5h + 0.5h^3)` for :math:`h<1`, else 0 -
    the classical geostatistical variogram with a finite range.

Every realisation is reproducible from its seed, which is stored with the
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.errors import ConfigError
from ..core.geometry import to_principal

COVARIANCE_MODELS = ("gaussian", "exponential", "spherical")


@dataclass
class HeterogeneitySpec:
    """Controls for one correlated random field."""

    mean: float = 0.0
    std: float = 1.0
    #: Horizontal correlation length along the ``azimuth`` direction, metres.
    correlation_major: float = 400.0
    #: Horizontal correlation length perpendicular to it, metres.
    correlation_minor: float = 400.0
    correlation_vertical: float = 20.0
    #: Bearing of the major axis, degrees clockwise from +y (map north).
    azimuth: float = 0.0
    model: str = "exponential"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.model not in COVARIANCE_MODELS:
            raise ConfigError(
                f"unknown covariance model {self.model!r}; choose from {COVARIANCE_MODELS}"
            )
        for name in ("correlation_major", "correlation_minor", "correlation_vertical"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive, got {getattr(self, name)}")
        if self.std < 0:
            raise ConfigError(f"std must be non-negative, got {self.std}")

    @property
    def anisotropy_ratio(self) -> float:
        """Horizontal-to-vertical correlation ratio."""
        return max(self.correlation_major, self.correlation_minor) / self.correlation_vertical


def _covariance(h: np.ndarray, model: str) -> np.ndarray:
    if model == "gaussian":
        return np.exp(-(h**2))
    if model == "exponential":
        return np.exp(-h)
    return np.where(h < 1.0, 1.0 - 1.5 * h + 0.5 * h**3, 0.0)


def random_field(grid, spec: HeterogeneitySpec) -> np.ndarray:
    """A correlated Gaussian field on ``grid`` with mean and standard deviation.

    The output is standardised to the requested mean and standard deviation
    after generation, so the statistics are exact regardless of how much
    variance the periodic embedding loses.
    """
    if spec.std == 0.0:
        return np.full(grid.shape, spec.mean, dtype=float)

    nx, ny, nz = grid.shape
    dx, dy, dz = grid.spacing
    # Minimum-image lags, so the covariance is periodic and its transform real.
    lx = np.minimum(np.arange(nx), nx - np.arange(nx)) * dx
    ly = np.minimum(np.arange(ny), ny - np.arange(ny)) * dy
    lz = np.minimum(np.arange(nz), nz - np.arange(nz)) * dz
    gx, gy, gz = np.meshgrid(lx, ly, lz, indexing="ij")

    major, minor = to_principal(gx, gy, spec.azimuth)
    h = np.sqrt((major / spec.correlation_major) ** 2
                + (minor / spec.correlation_minor) ** 2
                + (gz / spec.correlation_vertical) ** 2)

    spectrum = np.real(np.fft.fftn(_covariance(h, spec.model)))
    spectrum = np.clip(spectrum, 0.0, None)

    rng = np.random.default_rng(spec.seed)
    noise = np.fft.fftn(rng.standard_normal(grid.shape))
    field = np.real(np.fft.ifftn(noise * np.sqrt(spectrum)))

    field -= field.mean()
    scale = field.std()
    if scale > 0:
        field /= scale
    return spec.mean + spec.std * field


class GaussianField:
    """Convenience wrapper that caches one realisation per grid and spec."""

    def __init__(self, spec: HeterogeneitySpec):
        self.spec = spec
        self._cache: dict[tuple, np.ndarray] = {}

    def __call__(self, grid) -> np.ndarray:
        key = (grid.origin, grid.spacing, grid.shape)
        if key not in self._cache:
            self._cache[key] = random_field(grid, self.spec)
        return self._cache[key]


def clipped_field(grid, spec: HeterogeneitySpec, low: float, high: float) -> np.ndarray:
    """A correlated field truncated to ``[low, high]``.

    Truncation changes the distribution, so the realised mean and standard
    deviation will differ from the requested ones - deliberately, since a
    porosity field must stay physical.
    """
    if low >= high:
        raise ConfigError(f"clip bounds must increase, got ({low}, {high})")
    return np.clip(random_field(grid, spec), low, high)
