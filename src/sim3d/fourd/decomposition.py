r"""Pressure / saturation decomposition and the interaction term.

Spec sections 53 and 55.  The same two functions serve property space and
seismic space, because the algebra is identical and the distinction that
matters is what was fed in:

.. math::
    \Delta X_{pressure}    &= X_{pressure} - X_{baseline} \\
    \Delta X_{saturation}  &= X_{saturation} - X_{baseline} \\
    \Delta X_{combined}    &= X_{combined} - X_{baseline} \\
    \Delta X_{interaction} &= \Delta X_{combined}
                              - (\Delta X_{pressure} + \Delta X_{saturation})

In property space the interaction measures the nonlinearity of the rock
physics alone: Gassmann's response to a fluid change depends on the frame,
and the frame depends on effective stress, so the two effects do not add.

In seismic space it measures that *plus* everything the wave equation and
the imaging chain add - finite-frequency interference, tuning, illumination,
and the migration operator.  Comparing the two is what tells a study
whether an observed nonlinearity is rock physics or wave physics.

A near-zero interaction term is the signature of a small perturbation; it
growing with amplitude is the signature that superposition has stopped
being a safe assumption for that reservoir.
"""

from __future__ import annotations

import numpy as np

from ..core.errors import ConfigError


def decompose(baseline, pressure, saturation, combined) -> dict[str, np.ndarray]:
    """Differences and the interaction term for one quantity.

    Works on any array of matching shape: a property volume, a migrated
    seismic cube, or a single trace.

    Returns
    -------
    dict
        ``d_pressure``, ``d_saturation``, ``d_combined``, ``d_interaction``
        and ``d_sum`` (the linear superposition, for direct comparison).
    """
    arrays = [np.asarray(a, dtype=float)
              for a in (baseline, pressure, saturation, combined)]
    shapes = {a.shape for a in arrays}
    if len(shapes) != 1:
        raise ConfigError(
            f"all four inputs must have the same shape; got {sorted(shapes)}"
        )
    base, pres, sat, comb = arrays
    d_pressure = pres - base
    d_saturation = sat - base
    d_combined = comb - base
    d_sum = d_pressure + d_saturation
    return {
        "d_pressure": d_pressure,
        "d_saturation": d_saturation,
        "d_combined": d_combined,
        "d_sum": d_sum,
        "d_interaction": d_combined - d_sum,
    }


def interaction(baseline, pressure, saturation, combined) -> np.ndarray:
    """Just the interaction term of :func:`decompose`."""
    return decompose(baseline, pressure, saturation, combined)["d_interaction"]


def nonlinearity_ratio(parts: dict[str, np.ndarray], mask=None) -> float:
    """RMS interaction as a fraction of the RMS combined response.

    A useful single number for section 134: near zero means the pressure and
    saturation responses superpose, and a growing value means they do not.
    """
    combined = parts["d_combined"]
    inter = parts["d_interaction"]
    if mask is not None:
        combined, inter = combined[mask], inter[mask]
    denominator = float(np.sqrt(np.mean(combined**2)))
    if denominator == 0.0:
        return 0.0
    return float(np.sqrt(np.mean(inter**2)) / denominator)


def describe(parts: dict[str, np.ndarray], label: str = "quantity",
             scale: float = 1.0, unit: str = "", mask=None) -> str:
    """Human-readable summary of a decomposition."""
    lines = [f"Decomposition of {label}:"]
    for key in ("d_pressure", "d_saturation", "d_sum", "d_combined", "d_interaction"):
        arr = parts[key]
        arr = arr[mask] if mask is not None else arr
        lines.append(
            f"  {key:14s} min {np.min(arr) / scale:+11.5f}  "
            f"max {np.max(arr) / scale:+11.5f}  "
            f"rms {np.sqrt(np.mean(arr**2)) / scale:11.5f} {unit}"
        )
    lines.append(f"  interaction / combined (RMS): "
                 f"{100 * nonlinearity_ratio(parts, mask):.2f}%")
    return "\n".join(lines)
