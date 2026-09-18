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

**Pressure is always shown to the user in psi**, and volumes and rates in
the oilfield units an engineer reads without converting - STB, STB/day,
Mscf/day, mD, days.  The solver stays in SI; the conversion happens once, at
the display boundary, and is never something the user has to do in their
head.
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
DAY = 86400.0        # s per day
STB = 0.158987294928  # m^3 per stock-tank barrel
SCF = 0.0283168466    # m^3 per standard cubic foot
MSCF = 1.0e3 * SCF    # m^3 per thousand standard cubic feet
MMSCF = 1.0e6 * SCF   # m^3 per million standard cubic feet
MILLIDARCY = 9.869232667160128e-16   # m^2 per mD
CENTIPOISE = 1.0e-3   # Pa.s per cP

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
    "permeability": "m2",
    "viscosity": "Pa.s",
    "rate": "m3/s",
    "volume": "m3",
    "duration": "s",
    "bhp": "Pa",
    "dp": "Pa",
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


def psi_to_pa(value):
    """Convert psi to Pa."""
    return value * PSI


def pa_to_psi(value):
    """Convert Pa to psi - the unit every pressure is shown in."""
    return value / PSI


def days_to_seconds(value):
    """Convert days to seconds.  Simulation time is stored in days."""
    return value * DAY


def seconds_to_days(value):
    """Convert seconds to days."""
    return value / DAY


def stb_per_day_to_si(value):
    """Convert STB/day to m^3/s."""
    return value * STB / DAY


def si_to_stb_per_day(value):
    """Convert m^3/s to STB/day."""
    return value * DAY / STB


def mscf_per_day_to_si(value):
    """Convert Mscf/day to m^3/s."""
    return value * MSCF / DAY


def si_to_mscf_per_day(value):
    """Convert m^3/s to Mscf/day."""
    return value * DAY / MSCF


def md_to_si(value):
    """Convert millidarcy to m^2."""
    return value * MILLIDARCY


def si_to_md(value):
    """Convert m^2 to millidarcy."""
    return value / MILLIDARCY


#: Quantity name -> (display unit, factor such that ``si_value / factor`` is
#: the displayed number).  Used only by the UI and reporting layers.
#:
#: Every pressure is psi.  There is no per-page choice and no toggle: a study
#: that quotes depletion in bar on one screen and psi on another is a study
#: whose numbers cannot be compared by eye.
DISPLAY_UNITS: dict[str, tuple[str, float]] = {
    "pressure": ("psi", PSI),
    "p_pore": ("psi", PSI),
    "p_conf": ("psi", PSI),
    "p_eff": ("psi", PSI),
    "bhp": ("psi", PSI),
    "dp": ("psi", PSI),
    "permeability": ("mD", MILLIDARCY),
    "viscosity": ("cP", CENTIPOISE),
    "liquid_rate": ("STB/day", STB / DAY),
    "water_rate": ("STB/day", STB / DAY),
    "oil_rate": ("STB/day", STB / DAY),
    "gas_rate": ("Mscf/day", MSCF / DAY),
    "cumulative": ("STB", STB),
    "duration": ("days", DAY),
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
    "s": {"ms": MS, "days": DAY, "day": DAY, "years": 365.25 * DAY},
    "m/s": {"km/s": KM, "ft/s": FT},
    "m2": {"mD": MILLIDARCY, "D": 1000 * MILLIDARCY},
    "Pa.s": {"cP": CENTIPOISE},
    "m3/s": {"STB/day": STB / DAY, "Mscf/day": MSCF / DAY, "MMscf/day": MMSCF / DAY},
    "m3": {"STB": STB, "Mscf": MSCF, "MMscf": MMSCF},
    "kg/m3": {"g/cc": GCC, "g/cm3": GCC},
    "Pa": {"bar": BAR, "MPa": MPA, "GPa": 1.0e9, "psi": PSI, "kPa": 1.0e3,
           "psia": PSI},
    "Hz": {"kHz": 1.0e3},
}
