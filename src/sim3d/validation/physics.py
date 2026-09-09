"""What each simulation mode does and does not represent (spec section 83).

Displayed before every run.  The point of the table is that a user should
never have to infer from an image whether diffractions were modelled.
"""

from __future__ import annotations

from ..core.errors import ConfigError

PHYSICS_MODES: dict[str, dict[str, object]] = {
    "convolution": {
        "label": "Fast 1D/Convolution Approximation - Not Full 3D Wave Modelling",
        "includes": [
            "vertical normal-incidence reflectivity",
            "source bandwidth",
            "tuning between closely spaced interfaces",
        ],
        "excludes": [
            "lateral wave propagation",
            "diffraction",
            "refraction and head waves",
            "transmission loss and geometric spreading",
            "acquisition geometry and illumination",
            "migration",
            "S waves and mode conversion",
        ],
    },
    "sparse_synthetic": {
        "label": ("Sparse 1D Synthetic Traces - Not Full 3D Wave Modelling "
                  "or Migration"),
        "includes": [
            "vertical normal-incidence reflectivity",
            "source bandwidth",
            "tuning between closely spaced interfaces",
            "the 4D signal as the rock physics put it into the impedance profile",
        ],
        "excludes": [
            "everything away from the K sampled columns",
            "lateral wave propagation",
            "diffraction",
            "refraction and head waves",
            "transmission loss and geometric spreading",
            "acquisition geometry, offset, azimuth and illumination",
            "migration",
            "S waves and mode conversion",
        ],
    },
    "acoustic_fd": {
        "label": "3D acoustic finite-difference propagation",
        "includes": [
            "3D wave propagation through the full heterogeneous medium",
            "primary reflections and transmissions",
            "refraction and head waves",
            "diffraction",
            "geometric spreading",
            "finite-frequency interference and tuning",
            "structural focusing and defocusing",
            "internal multiples",
            "acquisition geometry and illumination",
        ],
        "excludes": [
            "S waves and mode conversion",
            "anisotropy",
            "intrinsic attenuation (Q)",
            "free-surface multiples and ghosts (every face is absorbing)",
        ],
    },
    "rtm": {
        "label": "3D acoustic reverse time migration",
        "includes": [
            "two-way wave-equation imaging",
            "correct positioning of dipping and faulted reflectors",
            "diffraction collapse",
            "illumination imprint of the acquisition",
        ],
        "excludes": [
            "amplitude-preserving (least-squares) imaging",
            "angle-domain gathers",
            "elastic and converted-wave imaging",
        ],
    },
    "elastic_fd": {
        "label": "3D elastic finite-difference propagation (not implemented)",
        "includes": ["P waves", "S waves", "converted waves"],
        "excludes": ["everything - this mode is not implemented yet"],
    },
}


def describe_mode(mode: str) -> str:
    """Human-readable physics statement for one mode."""
    try:
        entry = PHYSICS_MODES[mode]
    except KeyError:
        raise ConfigError(
            f"unknown physics mode {mode!r}; choose from {sorted(PHYSICS_MODES)}"
        ) from None
    lines = [str(entry["label"]), "  includes:"]
    lines += [f"    + {item}" for item in entry["includes"]]
    lines.append("  does NOT include:")
    lines += [f"    - {item}" for item in entry["excludes"]]
    return "\n".join(lines)


def physics_table(modes=None) -> str:
    """The full transparency table for the modes in use."""
    modes = list(modes or ("convolution", "sparse_synthetic", "acoustic_fd", "rtm"))
    return "\n\n".join(describe_mode(m) for m in modes)
