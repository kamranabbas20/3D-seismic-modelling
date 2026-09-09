"""SI unit handling and explicit display conversions (spec section 115).

Internally every quantity in sim3d is SI:

======================  =========================
quantity                unit
======================  =========================
length                  m
time                    s
velocity                m/s
density                 kg/m^3
pressure / modulus      Pa
frequency               Hz
temperature             degC (see note)
======================  =========================

Temperature is the one deliberate exception: Batzle-Wang correlations are
published in degrees Celsius, so temperature is carried in degC throughout
and the unit string says so.  Nothing is converted silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import UnitsError

# --- conversion factors, all "multiply the field unit by this to get SI" ---
BAR = 1.0e5          # Pa per bar
MPA = 1.0e6          # Pa per MPa
PSI = 6894.757293168  # Pa per psi
GCC = 1.0e3          # kg/m^3 per g/cm^3
FT = 0.3048          # m per ft
KM = 1.0e3           # m per km
MS = 1.0e-3          # s per ms

#: Canonical SI unit string for each physical quantity sim3d stores.
SI_UNITS: dict[str, str] = {
    "x": "m", "y": "m", "z": "m",
    "time": "s",
    "vp": "m/s", "vs": "m/s",
    "rho": "kg/m3",
    "k": "Pa", "mu": "Pa", "kdry": "Pa", "kfluid": "Pa", "kmin": "Pa",
    "pressure": "Pa", "p_pore": "Pa", "p_conf": "Pa", "p_eff": "Pa",
    "ai": "kg/m2/s", "si": "kg/m2/s",
    "phi": "1", "ntg": "1", "vsh": "1",
    "sw": "1", "so": "1", "sg": "1",
    "frequency": "Hz",
    "temperature": "degC",
    "salinity": "ppm",
    "gor": "L/L",
    "api": "degAPI",
}


def bar_to_pa(value):
    """Convert bar to Pa."""
    return value * BAR


def pa_to_bar(value):
    """Convert Pa to bar."""
    return value / BAR


def mpa_to_pa(value):
    """Convert MPa to Pa."""
    return value * MPA


def pa_to_mpa(value):
    """Convert Pa to MPa."""
    return value / MPA


def gcc_to_kgm3(value):
    """Convert g/cm^3 to kg/m^3."""
    return value * GCC


def kgm3_to_gcc(value):
    """Convert kg/m^3 to g/cm^3."""
    return value / GCC


def ms_to_s(value):
    """Convert milliseconds to seconds."""
    return value * MS


def s_to_ms(value):
    """Convert seconds to milliseconds."""
    return value / MS


#: Quantity name -> (display unit, factor such that ``si_value / factor`` is
#: the displayed number).  Used only by the UI and reporting layers.
DISPLAY_UNITS: dict[str, tuple[str, float]] = {
    "pressure": ("bar", BAR),
    "p_pore": ("bar", BAR),
    "p_conf": ("bar", BAR),
    "p_eff": ("bar", BAR),
    "rho": ("g/cc", GCC),
    "k": ("GPa", 1.0e9),
    "mu": ("GPa", 1.0e9),
    "kdry": ("GPa", 1.0e9),
    "kfluid": ("GPa", 1.0e9),
    "kmin": ("GPa", 1.0e9),
    "time": ("ms", MS),
}


def to_display(quantity: str, si_value):
    """Return ``(value_in_display_unit, unit_string)`` for a SI quantity."""
    unit, factor = DISPLAY_UNITS.get(quantity, (SI_UNITS.get(quantity, "1"), 1.0))
    return si_value / factor, unit


@dataclass(frozen=True)
class Quantity:
    """A number with an explicit unit string.

    Used at configuration boundaries so that a YAML file can say
    ``dp_max: {value: -50, unit: bar}`` and be converted once, loudly,
    on ingest.
    """

    value: float
    unit: str

    def to_si(self, quantity: str) -> float:
        """Convert to the SI unit sim3d stores ``quantity`` in.

        Raises :class:`UnitsError` if the unit is not a recognised
        alternative for that quantity.
        """
        target = SI_UNITS.get(quantity)
        if target is None:
            raise UnitsError(f"unknown quantity {quantity!r}; no SI unit declared")
        unit = self.unit.strip()
        if unit == target:
            return float(self.value)
        table = _ALTERNATIVES.get(target, {})
        if unit not in table:
            known = ", ".join(sorted({target, *table}))
            raise UnitsError(
                f"cannot express {quantity!r} in {unit!r}; known units: {known}"
            )
        return float(self.value) * table[unit]


#: Accepted non-SI input units per SI target unit, with their SI factors.
_ALTERNATIVES: dict[str, dict[str, float]] = {
    "m": {"km": KM, "ft": FT},
    "s": {"ms": MS},
    "m/s": {"km/s": KM, "ft/s": FT},
    "kg/m3": {"g/cc": GCC, "g/cm3": GCC},
    "Pa": {"bar": BAR, "MPa": MPA, "GPa": 1.0e9, "psi": PSI, "kPa": 1.0e3},
    "Hz": {"kHz": 1.0e3},
}
