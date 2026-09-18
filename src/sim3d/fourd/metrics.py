r"""4D metrics and well-centred analysis (spec sections 88-92).

Amplitude difference and time shift are kept separate throughout
(section 89): a reservoir change that slows the overburden shifts events
in time, and a change in reflectivity changes their amplitude, and
conflating the two is how a pressure signal gets misread as a saturation
one.

NRMS follows the standard definition

.. math:: \mathrm{NRMS} = \frac{200\,\mathrm{RMS}(a-b)}
                               {\mathrm{RMS}(a) + \mathrm{RMS}(b)}

in percent, so 0 is perfect repeatability and 200 is anticorrelation.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def rms(values, mask=None) -> float:
    """Root-mean-square of an array, optionally within a mask."""
    a = np.asarray(values, dtype=float)
    a = a[mask] if mask is not None else a
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a**2)))


def nrms(a, b, mask=None) -> float:
    """Normalised RMS difference in percent."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ConfigError(f"shapes must match, got {x.shape} and {y.shape}")
    denominator = rms(x, mask) + rms(y, mask)
    if denominator == 0.0:
        return 0.0
    return float(200.0 * rms(x - y, mask) / denominator)


def cross_correlation(a, b, mask=None) -> float:
    """Zero-lag normalised cross-correlation, in ``[-1, 1]``."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if mask is not None:
        x, y = x[mask], y[mask]
    x, y = x.ravel(), y.ravel()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def predictability(a, b, mask=None) -> float:
    """Squared correlation - the fraction of variance one cube explains of the other."""
    return cross_correlation(a, b, mask) ** 2


def difference_energy(a, b, mask=None) -> float:
    """Total squared difference within the mask."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[mask] if mask is not None else d
    return float(np.sum(d**2))


def local_time_shift(baseline_trace, monitor_trace, sample_interval: float,
                     max_lag: int = 20) -> float:
    r"""Sub-sample time shift between two traces by cross-correlation.

    The integer lag maximising the cross-correlation is refined by fitting
    a parabola through it and its neighbours, giving a shift resolved to a
    fraction of a sample.  Positive means the monitor event arrives *later*
    - a slowdown, the classic pressure-increase signature.
    """
    a = np.asarray(baseline_trace, dtype=float)
    b = np.asarray(monitor_trace, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ConfigError("both traces must be 1D and the same length")
    if np.allclose(a, 0) or np.allclose(b, 0):
        return 0.0
    lags = np.arange(-max_lag, max_lag + 1)
    scores = np.array([_shifted_correlation(a, b, int(lag)) for lag in lags])
    peak = int(np.argmax(scores))
    if 0 < peak < len(scores) - 1:
        y0, y1, y2 = scores[peak - 1], scores[peak], scores[peak + 1]
        denominator = y0 - 2.0 * y1 + y2
        offset = 0.5 * (y0 - y2) / denominator if denominator != 0 else 0.0
    else:
        offset = 0.0
    return float((lags[peak] + offset) * sample_interval)


def _shifted_correlation(a: np.ndarray, b: np.ndarray, lag: int) -> float:
    if lag > 0:
        x, y = a[:-lag], b[lag:]
    elif lag < 0:
        x, y = a[-lag:], b[:lag]
    else:
        x, y = a, b
    if x.size == 0:
        return -np.inf
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else -np.inf


def time_shift_volume(baseline, monitor, sample_interval: float,
                      max_lag: int = 20) -> np.ndarray:
    """Trace-by-trace time shift over a cube whose last axis is time or depth."""
    base = np.asarray(baseline, dtype=float)
    mon = np.asarray(monitor, dtype=float)
    if base.shape != mon.shape:
        raise ConfigError(f"shapes must match, got {base.shape} and {mon.shape}")
    flat_base = base.reshape(-1, base.shape[-1])
    flat_mon = mon.reshape(-1, mon.shape[-1])
    shifts = np.array([
        local_time_shift(flat_base[i], flat_mon[i], sample_interval, max_lag)
        for i in range(flat_base.shape[0])
    ])
    return shifts.reshape(base.shape[:-1])


def well_centred_statistics(grid, well, values, radii=(100.0, 250.0, 500.0),
                            mask=None, statistic="mean") -> dict[float, float]:
    """Average a volume within cylinders of increasing radius about a well.

    Spec section 90.  Returns ``{radius: value}``.  The cylinder is vertical
    and spans the well's perforated interval, which is the interval a 4D
    response around that well should be attributed to.
    """
    values = np.asarray(values, dtype=float)
    if values.shape != grid.shape:
        raise ConfigError(f"values shape {values.shape} does not match grid {grid.shape}")
    x, y, z = np.meshgrid(grid.axis(0), grid.axis(1), grid.axis(2), indexing="ij")
    distance = np.hypot(x - well.x, y - well.y)
    in_perforation = well.perforation_mask(z)
    reducers = {"mean": np.mean, "max": np.max, "min": np.min, "sum": np.sum,
                "rms": lambda a: float(np.sqrt(np.mean(a**2)))}
    if statistic not in reducers:
        raise ConfigError(f"unknown statistic {statistic!r}; choose from {sorted(reducers)}")

    out = {}
    for radius in radii:
        sel = (distance <= radius) & in_perforation
        if mask is not None:
            sel &= mask
        out[float(radius)] = float(reducers[statistic](values[sel])) if np.any(sel) else float("nan")
    return out


def radial_profile(grid, well, values, bin_width: float = 50.0,
                   max_radius: float = 1000.0, mask=None):
    """Mean of ``values`` in annuli about a well (spec section 91).

    Returns ``(radius_centres, mean_values, counts)``.  Plotting a property
    change and a seismic difference against the same radius axis is the most
    direct way to see whether the seismic anomaly tracks the pressure halo
    or the saturation front.
    """
    values = np.asarray(values, dtype=float)
    x, y, z = np.meshgrid(grid.axis(0), grid.axis(1), grid.axis(2), indexing="ij")
    distance = np.hypot(x - well.x, y - well.y)
    sel = well.perforation_mask(z)
    if mask is not None:
        sel = sel & mask
    edges = np.arange(0.0, max_radius + bin_width, bin_width)
    centres = 0.5 * (edges[:-1] + edges[1:])
    means, counts = np.zeros(centres.size), np.zeros(centres.size, dtype=int)
    for i in range(centres.size):
        ring = sel & (distance >= edges[i]) & (distance < edges[i + 1])
        counts[i] = int(ring.sum())
        means[i] = float(values[ring].mean()) if counts[i] else np.nan
    return centres, means, counts


def affected_volume(grid, values, threshold: float, mask=None) -> float:
    """Volume in m^3 where ``|values|`` exceeds a threshold (spec section 92)."""
    sel = np.abs(np.asarray(values, dtype=float)) > threshold
    if mask is not None:
        sel &= mask
    cell = grid.dx * grid.dy * grid.dz
    return float(sel.sum() * cell)


def overlap_volume(grid, a, threshold_a: float, b, threshold_b: float, mask=None) -> float:
    """Volume where both quantities exceed their thresholds."""
    sel = (np.abs(np.asarray(a)) > threshold_a) & (np.abs(np.asarray(b)) > threshold_b)
    if mask is not None:
        sel &= mask
    return float(sel.sum() * grid.dx * grid.dy * grid.dz)
