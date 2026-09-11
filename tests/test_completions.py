"""Completions by geological unit, and physically-grounded rate defaults."""

import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.core.grid import Grid3D
from sim3d.core.units import si_to_stb_per_day
from sim3d.geology import build_geology, template
from sim3d.wells import Well, WellSet
from sim3d.wells.completion import (
    Completion, completion_mask, completion_summary, default_completions,
    layer_intersections, resolve_completions,
)
from sim3d.wells.controls import (
    ControlMode, WellControl, check_rates, inflow_properties, pore_volume,
    suggest_control,
)


@pytest.fixture(scope="module")
def dipping():
    """A dipping model, so layer depths genuinely differ between wells."""
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (50, 50, 10))
    layers, faults = template("dipping", z_reservoir=1200.0, gross=150.0, dip=6.0)
    return build_geology(grid, layers, faults)


@pytest.fixture(scope="module")
def faulted():
    grid = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (50, 50, 10))
    layers, faults = template("fault_compartment", throw=60.0)
    return build_geology(grid, layers, faults)


# ------------------------------------------------------------ intersections
def test_a_well_finds_every_unit_it_passes_through(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    units = layer_intersections(well, dipping)
    assert [u.name for u in units] == [layer.name for layer in dipping.layers]
    assert all(u.base > u.top for u in units)
    # Contiguous: each unit starts where the previous one ended.
    for a, b in zip(units, units[1:]):
        assert b.top == pytest.approx(a.base)


def test_layer_depths_follow_the_structure(dipping):
    """The whole point of naming a unit instead of typing a depth."""
    shallow = Well("A", "producer", 300.0, 1500.0, (1000.0, 1100.0))
    deep = Well("B", "producer", 2700.0, 1500.0, (1000.0, 1100.0))
    tops = []
    for well in (shallow, deep):
        units = {u.name: u for u in layer_intersections(well, dipping)}
        tops.append(units["reservoir"].top)
    # 6 degrees over 2400 m is about 250 m of relief.
    assert abs(tops[1] - tops[0]) > 150.0


def test_a_fault_offsets_the_completion_depth(faulted):
    across = []
    for x in (900.0, 2100.0):
        well = Well("W", "producer", x, 1500.0, (1200.0, 1350.0))
        units = {u.name: u for u in layer_intersections(well, faulted)}
        across.append(units["reservoir"].top)
    assert abs(across[1] - across[0]) > 20.0


def test_default_completions_open_the_reservoir_only(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    completions = default_completions(well, dipping)
    assert [c.layer for c in completions] == ["reservoir"]
    assert all(c.open for c in completions)


def test_resolved_intervals_use_the_local_depths(dipping):
    well = Well("P1", "producer", 900.0, 1500.0, (1200.0, 1350.0))
    units = {u.name: u for u in layer_intersections(well, dipping)}
    intervals = resolve_completions(well, dipping, [Completion("reservoir")])
    top, base, unit = intervals[0]
    assert top == pytest.approx(units["reservoir"].top)
    assert base == pytest.approx(units["reservoir"].base)
    assert unit.name == "reservoir"


def test_a_manual_interval_must_stay_inside_its_unit(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    units = {u.name: u for u in layer_intersections(well, dipping)}
    reservoir = units["reservoir"]

    inside = Completion("reservoir", top=reservoir.top + 20.0,
                        base=reservoir.base - 20.0)
    top, base = inside.resolve(reservoir)
    assert top > reservoir.top and base < reservoir.base

    with pytest.raises(ValidationError, match="outside that unit"):
        Completion("reservoir", top=reservoir.top - 200.0).resolve(reservoir)
    with pytest.raises(ConfigError, match="at or above"):
        Completion("reservoir", top=reservoir.base, base=reservoir.top).resolve(reservoir)


def test_completing_a_unit_the_well_does_not_reach_is_refused(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    with pytest.raises(ValidationError, match="does not intersect"):
        resolve_completions(well, dipping, [Completion("atlantis")])


def test_a_well_outside_the_model_is_refused(dipping):
    with pytest.raises(ValidationError, match="outside the geological model"):
        layer_intersections(Well("X", "producer", 9999.0, 1500.0), dipping)


def test_the_completion_mask_is_one_column_inside_the_intervals(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    mask = completion_mask(well, dipping, default_completions(well, dipping))
    assert mask.any()
    columns = np.unique(np.argwhere(mask)[:, :2], axis=0)
    assert len(columns) == 1                      # vertical well, one column
    assert np.all(dipping.reservoir_mask[mask])   # and only in the reservoir


def test_the_summary_states_which_units_are_open(dipping):
    well = Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0))
    text = completion_summary(well, dipping, default_completions(well, dipping))
    assert "OPEN " in text and "shut " in text
    assert "net completed thickness" in text


# ------------------------------------------------------------------- rates
def test_a_suggested_rate_is_physically_reasonable(dipping):
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0)),
                     Well("P1", "producer", 1800.0, 1500.0, (1200.0, 1350.0))])
    pressure = 1.3e7
    for well in wells:
        control = suggest_control(well, dipping, default_completions(well, dipping),
                                  list(wells), pressure)
        rate = si_to_stb_per_day(control.target)
        assert 100.0 < rate < 60_000.0, f"{well.name} suggested {rate:,.0f} STB/day"
        assert control.valid_for(well.role)
        assert "deliverability" in control.provenance
        assert "pattern scale" in control.provenance


def test_deliverability_alone_would_be_absurd(dipping):
    """A thick, high-permeability completion can flow far more than a field
    is ever developed at; the pattern-scale limit is what makes the default
    sane, so it must be the binding one here."""
    wells = WellSet([Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0)),
                     Well("P2", "producer", 2000.0, 1500.0, (1200.0, 1350.0))])
    control = suggest_control(wells[0], dipping,
                              default_completions(wells[0], dipping),
                              list(wells), 1.3e7)
    assert "bound by pattern scale" in control.provenance


def test_a_thinner_completion_gets_a_smaller_rate(dipping):
    wells = WellSet([Well("P1", "producer", 1500.0, 1500.0, (1200.0, 1350.0)),
                     Well("P2", "producer", 2000.0, 1500.0, (1200.0, 1350.0))])
    units = {u.name: u for u in layer_intersections(wells[0], dipping)}
    reservoir = units["reservoir"]
    full = suggest_control(wells[0], dipping, [Completion("reservoir")],
                           list(wells), 1.3e7)
    partial = suggest_control(
        wells[0], dipping,
        [Completion("reservoir", top=reservoir.top,
                    base=reservoir.top + 0.25 * reservoir.thickness)],
        list(wells), 1.3e7)
    assert partial.target < 0.5 * full.target


def test_an_injector_is_capped_at_the_fracture_gradient(dipping):
    wells = WellSet([Well("I1", "injector", 1500.0, 1500.0, (1200.0, 1350.0)),
                     Well("P1", "producer", 2000.0, 1500.0, (1200.0, 1350.0))])
    control = suggest_control(wells[0], dipping,
                              default_completions(wells[0], dipping),
                              list(wells), 1.3e7)
    depth = 0.5 * sum(dipping.grid.bounds[2])
    from sim3d.wells.controls import FRACTURE_GRADIENT
    assert control.bhp_limit <= FRACTURE_GRADIENT * depth * 1.000001


def test_schedules_open_and_close_wells():
    control = WellControl(ControlMode.LIQUID_RATE, 1.0, start_day=100.0, end_day=200.0)
    assert not control.active(50.0)
    assert control.active(150.0)
    assert not control.active(250.0)
    with pytest.raises(ConfigError, match="at or before its start"):
        WellControl(ControlMode.LIQUID_RATE, 1.0, start_day=100.0, end_day=50.0)


def test_control_modes_are_checked_against_the_role():
    assert WellControl(ControlMode.LIQUID_RATE, 1.0).valid_for("producer")
    assert not WellControl(ControlMode.LIQUID_RATE, 1.0).valid_for("injector")
    assert WellControl(ControlMode.BHP, 1e7).valid_for("injector")


# -------------------------------------------------------------- sanity checks
def test_a_sensible_pattern_raises_no_warnings(dipping):
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0)),
                     Well("P1", "producer", 1800.0, 1500.0, (1200.0, 1350.0))])
    pressure = 1.3e7
    controls = {w.name: suggest_control(w, dipping, default_completions(w, dipping),
                                        list(wells), pressure) for w in wells}
    assert check_rates(list(wells), controls, dipping, pressure, 1825.0) == []


def test_an_unbalanced_voidage_ratio_is_flagged(dipping):
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0)),
                     Well("P1", "producer", 1800.0, 1500.0, (1200.0, 1350.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 3.0e-2),
                "P1": WellControl(ControlMode.LIQUID_RATE, 3.0e-3)}
    warnings = check_rates(list(wells), controls, dipping, 1.3e7, 365.0)
    assert any("voidage replacement" in w for w in warnings)


def test_injection_with_no_production_is_flagged(dipping):
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 1.0e-3)}
    warnings = check_rates(list(wells), controls, dipping, 1.3e7, 365.0)
    assert any("without limit" in w for w in warnings)


def test_a_wrong_control_mode_for_the_role_is_flagged(dipping):
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0))])
    controls = {"I1": WellControl(ControlMode.OIL_RATE, 1.0e-3)}
    assert any("not a injector control mode" in w
               for w in check_rates(list(wells), controls, dipping, 1.3e7, 365.0))


def test_pore_volume_is_net_not_gross(dipping):
    grid = dipping.grid
    gross = float(dipping.reservoir_mask.sum()) * grid.dx * grid.dy * grid.dz
    assert 0 < pore_volume(dipping) < gross


def test_warnings_explain_rather_than_reject(dipping):
    """Every check returns text; none of them raises."""
    wells = WellSet([Well("I1", "injector", 1200.0, 1500.0, (1200.0, 1350.0))])
    controls = {"I1": WellControl(ControlMode.WATER_RATE, 10.0)}
    warnings = check_rates(list(wells), controls, dipping, 1.3e7, 3650.0)
    assert warnings and all(isinstance(w, str) and len(w) > 40 for w in warnings)
