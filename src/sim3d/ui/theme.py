"""Colour and layout constants for the GUI.

The palette is the validated default of the project's data-visualisation
method, applied by the job each colour does rather than by taste:

* **sequential** (one hue, light to dark) for magnitudes that are only ever
  positive - porosity, Vp, impedance, illumination;
* **diverging** (two hues, neutral grey midpoint, symmetric limits) for
  anything signed - a 4D difference, a seismic amplitude, a pressure change.
  A signed field on a sequential ramp hides the sign, which is the single
  most consequential display mistake this application could make;
* **categorical**, in fixed slot order and never cycled, only for series
  identity in line charts.

There is deliberately no rainbow anywhere.  A rainbow ramp invents
boundaries where the data has none - exactly the failure mode a mechanistic
study is trying to avoid - and is not colourblind-safe.

The four categorical slots were validated against the light chart surface
(`#fcfcfb`): worst adjacent CVD Delta E 9.1, worst adjacent normal-vision
Delta E 22.9, both clear. Two slots sit below 3:1 contrast on that surface,
so every chart using them ships a legend *and* direct end-labels rather than
relying on colour alone.

The app pins Streamlit to its light theme because that is the surface the
palette was validated against. A dark palette is a set of steps chosen for
the dark surface, not an automatic inversion of these, so shipping one
without validating it would be worse than not shipping it.
"""

from __future__ import annotations

from ..core.units import PSI

# --- surfaces and ink -----------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

# --- categorical: identity only, fixed order, never cycled ----------------
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

#: Fixed colour per 4D scenario, so a scenario keeps its identity across
#: every chart in the app even when a filter changes which are shown.
SCENARIO_COLOUR = {
    "baseline": INK_SECONDARY,
    "pressure_only": SERIES[0],
    "saturation_only": SERIES[1],
    "combined": SERIES[2],
    "interaction": SERIES[3],
}

# --- well type ------------------------------------------------------------
#: The application-wide convention: producers green, injectors blue, filled
#: circles, in map view, sections, the 3D scene and every selection list.
#:
#: The pair separates well for protanopia and deuteranopia (Delta E 26.5) but
#: only marginally for tritanopia (7.6, inside the 6-8 band), so well type is
#: never carried by colour alone: every symbol is drawn with its well name
#: beside it and the two roles use different marker outlines.
WELL_COLOUR = {
    "producer": "#008300",
    "injector": "#2a78d6",
    "observation": INK_SECONDARY,
}
WELL_SYMBOL_2D = {"producer": "circle", "injector": "circle", "observation": "circle-open"}
WELL_SYMBOL_3D = {"producer": "circle", "injector": "circle", "observation": "circle-open"}

# --- status ---------------------------------------------------------------
STATUS = {"PASS": "#0ca30c", "WARNING": "#fab219", "FAIL": "#d03b3b"}
STATUS_ICON = {"PASS": "✓", "WARNING": "!", "FAIL": "✕"}

# --- sequential: one hue, light to dark -----------------------------------
_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b")

#: Red arm of the diverging pair, generated at the blue arm's own OKLCH
#: lightness at every step (matched to within 0.001) so neither pole reads
#: as heavier than the other. Equal step count per arm.
_RED = ("#fad6d2", "#f4c2be", "#f1aea8", "#ea9a93", "#e4857e", "#dd716a",
        "#d75853", "#c74845", "#b13f3c", "#9e3432", "#892b2a", "#762221",
        "#621b1a")

DIVERGING_MID = "#f0efec"


def _scale(colours):
    n = len(colours) - 1
    return [[i / n, c] for i, c in enumerate(colours)]


#: Magnitude ramp, light (near zero) to dark.
SEQUENTIAL = _scale(_BLUE)

#: Signed ramp: red for negative, neutral grey at zero, blue for positive.
#: Used only with symmetric limits, so the midpoint always means "no change".
DIVERGING = _scale((*reversed(_RED), DIVERGING_MID, *_BLUE))

#: Seismic amplitude uses the same diverging pair, which keeps a migrated
#: image and a 4D difference readable on one screen without a mental switch.
SEISMIC = DIVERGING


def plotly_layout(**overrides) -> dict:
    """Recessive chrome: hairline grid, muted axes, no chart-junk."""
    layout = dict(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif',
                  size=12, color=INK_PRIMARY),
        margin=dict(l=56, r=16, t=44, b=44),
        xaxis=dict(gridcolor=GRIDLINE, linecolor=AXIS, zeroline=False,
                   tickfont=dict(color=INK_MUTED)),
        yaxis=dict(gridcolor=GRIDLINE, linecolor=AXIS, zeroline=False,
                   tickfont=dict(color=INK_MUTED)),
        hoverlabel=dict(bgcolor=SURFACE, font_size=12,
                        bordercolor="rgba(11,11,11,0.10)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_SECONDARY)),
    )
    layout.update(overrides)
    return layout


#: Display scaling per quantity: (factor, unit, colour job).
DISPLAY = {
    "vp": (1.0, "m/s", "sequential"),
    "vs": (1.0, "m/s", "sequential"),
    "rho": (1.0, "kg/m³", "sequential"),
    "ai": (1e6, "10⁶ kg/m²/s", "sequential"),
    "porosity": (1.0, "fraction", "sequential"),
    "vsh": (1.0, "fraction", "sequential"),
    "ntg": (1.0, "fraction", "sequential"),
    "pressure": (PSI, "psi", "sequential"),
    "sw": (1.0, "fraction", "sequential"),
    "so": (1.0, "fraction", "sequential"),
    "sg": (1.0, "fraction", "sequential"),
    "dP": (PSI, "psi", "diverging"),
    "dSw": (1.0, "fraction", "diverging"),
    "dSo": (1.0, "fraction", "diverging"),
    "dSg": (1.0, "fraction", "diverging"),
    "dvp": (1.0, "m/s", "diverging"),
    "drho": (1.0, "kg/m³", "diverging"),
    "dai": (1e6, "10⁶ kg/m²/s", "diverging"),
    "image": (1.0, "amplitude", "diverging"),
}
