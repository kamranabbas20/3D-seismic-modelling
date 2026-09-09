import pytest

from sim3d.core.errors import UnitsError
from sim3d.core.units import Quantity, pa_to_bar, to_display


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


def test_display_conversion_reports_its_unit():
    value, unit = to_display("pressure", -5.0e6)
    assert value == pytest.approx(-50.0)
    assert unit == "bar"
