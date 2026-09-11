import pytest

from sim3d.core.errors import UnitsError
from sim3d.core.units import (
    Quantity, md_to_si, pa_to_bar, pa_to_psi, psi_to_pa, seconds_to_days,
    si_to_md, si_to_stb_per_day, stb_per_day_to_si, to_display,
)


def test_pressure_round_trip():
    assert Quantity(-50.0, "bar").to_si("pressure") == pytest.approx(-5.0e6)
    assert pa_to_bar(-5.0e6) == pytest.approx(-50.0)


def test_si_unit_passes_through():
    assert Quantity(2350.0, "kg/m3").to_si("rho") == pytest.approx(2350.0)


def test_density_and_length_alternatives():
    assert Quantity(2.35, "g/cc").to_si("rho") == pytest.approx(2350.0)
    assert Quantity(1.5, "km").to_si("x") == pytest.approx(1500.0)


def test_unknown_unit_is_rejected_not_guessed():
    with pytest.raises(UnitsError, match="known units"):
        Quantity(50.0, "furlongs").to_si("pressure")


def test_unknown_quantity_is_rejected():
    with pytest.raises(UnitsError):
        Quantity(1.0, "m").to_si("not_a_quantity")


def test_pressure_is_always_displayed_in_psi():
    """Requirement 8: one pressure unit everywhere, with no toggle.

    A study that quotes depletion in bar on one screen and psi on another
    is a study whose numbers cannot be compared by eye.
    """
    value, unit = to_display("pressure", -5.0e6)
    assert unit == "psi"
    assert value == pytest.approx(-725.19, abs=0.01)
    for quantity in ("pressure", "p_pore", "p_conf", "p_eff", "bhp", "dp"):
        assert to_display(quantity, 1.0e7)[1] == "psi"


def test_psi_round_trips():
    assert pa_to_psi(30e6) == pytest.approx(4351.13, abs=0.01)
    assert psi_to_pa(2500.0) == pytest.approx(1.7236893e7, rel=1e-7)
    assert pa_to_psi(psi_to_pa(1234.5)) == pytest.approx(1234.5)


def test_oilfield_rate_and_permeability_units():
    # 1 STB is 0.158987 m^3, so 5,000 STB/day is about 795 m^3/day.
    assert stb_per_day_to_si(5000.0) * 86400.0 == pytest.approx(794.94, abs=0.01)
    assert si_to_stb_per_day(stb_per_day_to_si(1234.0)) == pytest.approx(1234.0)
    assert si_to_md(md_to_si(250.0)) == pytest.approx(250.0)
    assert md_to_si(1000.0) == pytest.approx(9.8692e-13, rel=1e-4)
    assert seconds_to_days(86400.0) == pytest.approx(1.0)


def test_oilfield_units_are_accepted_at_the_configuration_boundary():
    assert Quantity(2500.0, "psi").to_si("pressure") == pytest.approx(psi_to_pa(2500.0))
    assert Quantity(200.0, "mD").to_si("permeability") == pytest.approx(md_to_si(200.0))
    assert Quantity(5.0, "days").to_si("time") == pytest.approx(432000.0)
    assert Quantity(1000.0, "STB/day").to_si("rate") == pytest.approx(
        stb_per_day_to_si(1000.0))


def test_bar_conversion_still_exists_for_internal_use():
    assert pa_to_bar(-5.0e6) == pytest.approx(-50.0)


def test_qc_reports_pore_pressure_in_psi():
    """Requirement 8 reaches the QC panel too, which is user-facing text."""
    import numpy as np
    from types import SimpleNamespace
    from sim3d.validation.qc import check_state

    ones = np.ones((2, 2, 2))
    state = SimpleNamespace(sw=0.3 * ones, so=0.7 * ones, sg=0.0 * ones,
                            porosity=0.2 * ones, pressure=2.0e7 * ones)
    line = next(c.message for c in check_state(state).checks
                if "pore pressure" in c.message)
    assert "psi" in line and "bar" not in line
    assert "2,901" in line          # 20 MPa
