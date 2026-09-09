import numpy as np
import pytest

from sim3d.core.errors import ConfigError, ValidationError
from sim3d.core.grid import Grid3D
from sim3d.geology import (
    FACIES, Fault, FaultSet, HeterogeneitySpec, Layer, build_geology, random_field, template,
)
from sim3d.geology.surfaces import Anticline, Dipping, Flat, Relief, Syncline
from sim3d.geology.templates import TEMPLATES


@pytest.fixture(scope="module")
def grid():
    return Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (100, 100, 20))


# ------------------------------------------------------------------ surfaces
def test_flat_surface_is_constant(grid):
    assert np.allclose(Flat(1200.0).on_grid(grid), 1200.0)


def test_dip_matches_its_trigonometry(grid):
    z = Dipping(1200.0, dip=3.0, azimuth=90.0).on_grid(grid)
    assert z.max() - z.min() == pytest.approx(3000.0 * np.tan(np.radians(3.0)), rel=1e-6)


def test_dip_azimuth_points_where_it_says(grid):
    """Azimuth is clockwise from +y, so 90 degrees deepens towards +x."""
    z = Dipping(1200.0, dip=3.0, azimuth=90.0).on_grid(grid)
    assert z[-1, 0] > z[0, 0]
    assert z[0, -1] == pytest.approx(z[0, 0])
    north = Dipping(1200.0, dip=3.0, azimuth=0.0).on_grid(grid)
    assert north[0, -1] > north[0, 0]


def test_anticline_shallows_at_its_crest_by_its_amplitude(grid):
    fold = Anticline(1200.0, amplitude=80.0, centre=(1500.0, 1500.0), radius=(900.0, 1400.0))
    z = fold.on_grid(grid)
    assert z.min() == pytest.approx(1120.0, abs=1.0)
    assert z.max() < 1200.0 + 1e-9


def test_syncline_mirrors_the_anticline(grid):
    kwargs = dict(z0=1200.0, amplitude=80.0, centre=(1500.0, 1500.0), radius=(900.0, 1400.0))
    up, down = Anticline(**kwargs).on_grid(grid), Syncline(**kwargs).on_grid(grid)
    assert np.allclose(up + down, 2 * 1200.0)


def test_relief_lets_structures_add_without_double_counting_the_datum(grid):
    combined = (Flat(1200.0) + Relief(Anticline(0.0, amplitude=80.0,
                                                centre=(1500.0, 1500.0),
                                                radius=(900.0, 1400.0)))).on_grid(grid)
    # The Gaussian dome has not quite decayed to zero at the far corner, so
    # the deepest point is a metre or two above the flat datum.
    assert combined.max() == pytest.approx(1200.0, abs=2.0)
    assert combined.min() == pytest.approx(1120.0, abs=1.0)


def test_impossible_dip_is_rejected(grid):
    with pytest.raises(ConfigError, match="dip"):
        Dipping(1200.0, dip=95.0).on_grid(grid)


# ------------------------------------------------------------ heterogeneity
def test_random_fields_hit_their_requested_statistics(grid):
    f = random_field(grid, HeterogeneitySpec(mean=0.22, std=0.04, seed=1))
    assert f.mean() == pytest.approx(0.22, abs=1e-9)
    assert f.std() == pytest.approx(0.04, rel=1e-9)


def test_random_fields_are_reproducible_from_their_seed(grid):
    spec = HeterogeneitySpec(std=1.0, seed=7)
    assert np.array_equal(random_field(grid, spec), random_field(grid, spec))
    other = HeterogeneitySpec(std=1.0, seed=8)
    assert not np.array_equal(random_field(grid, spec), random_field(grid, other))


@pytest.mark.parametrize("model,expected", [("gaussian", np.exp(-1.0)),
                                            ("exponential", np.exp(-1.0))])
def test_realised_correlation_matches_the_covariance_model_at_one_range(grid, model, expected):
    """At a lag of one correlation length the empirical correlation must match C(1)."""
    spec = HeterogeneitySpec(std=1.0, correlation_major=600.0, correlation_minor=600.0,
                             correlation_vertical=100.0, model=model, seed=3)
    f = random_field(grid, spec)
    lag = int(600.0 / grid.dx)
    realised = np.corrcoef(f[:-lag].ravel(), f[lag:].ravel())[0, 1]
    assert realised == pytest.approx(expected, abs=0.10)


def test_spherical_covariance_has_a_finite_range(grid):
    spec = HeterogeneitySpec(std=1.0, correlation_major=600.0, correlation_minor=600.0,
                             correlation_vertical=100.0, model="spherical", seed=3)
    f = random_field(grid, spec)
    lag = int(600.0 / grid.dx)
    assert abs(np.corrcoef(f[:-lag].ravel(), f[lag:].ravel())[0, 1]) < 0.12


@pytest.mark.parametrize("azimuth,long_axis", [(0.0, "y"), (90.0, "x")])
def test_anisotropy_points_where_the_azimuth_says(grid, azimuth, long_axis):
    """Azimuth is the bearing of the major axis, clockwise from +y."""
    spec = HeterogeneitySpec(std=1.0, correlation_major=1200.0, correlation_minor=200.0,
                             correlation_vertical=100.0, azimuth=azimuth, seed=5)
    f = random_field(grid, spec)
    lag = 4  # 400 m at this grid spacing
    along_x = np.corrcoef(f[:-lag].ravel(), f[lag:].ravel())[0, 1]
    along_y = np.corrcoef(f[:, :-lag].ravel(), f[:, lag:].ravel())[0, 1]
    if long_axis == "x":
        assert along_x > along_y + 0.2
    else:
        assert along_y > along_x + 0.2


def test_bad_heterogeneity_settings_are_rejected():
    with pytest.raises(ConfigError):
        HeterogeneitySpec(correlation_major=-1.0)
    with pytest.raises(ConfigError, match="covariance model"):
        HeterogeneitySpec(model="wishful")


# ---------------------------------------------------------------------- faults
def test_fault_normal_is_a_unit_vector_perpendicular_to_strike_and_dip():
    f = Fault(strike=37.0, dip=64.0)
    assert np.linalg.norm(f.normal) == pytest.approx(1.0)
    assert np.dot(f.normal, f.strike_vector) == pytest.approx(0.0, abs=1e-12)
    assert np.dot(f.normal, f.dip_vector) == pytest.approx(0.0, abs=1e-12)


def test_a_vertical_fault_splits_the_model_at_its_origin():
    f = Fault(origin=(1500.0, 1500.0, 1200.0), strike=0.0, dip=90.0, throw=40.0,
              zone_width=1.0)
    x = np.array([1400.0, 1600.0])
    frac = f.hanging_wall_fraction(x, np.full(2, 1500.0), np.full(2, 1200.0))
    assert frac[0] < 0.01 and frac[1] > 0.99


@pytest.mark.parametrize("throw,sense", [(40.0, "normal"), (-40.0, "reverse")])
def test_restoring_through_a_fault_offsets_by_the_throw(throw, sense):
    """A positive throw drops the hanging wall: horizons there sit deeper."""
    f = Fault(origin=(1500.0, 1500.0, 1200.0), strike=0.0, dip=90.0, throw=throw,
              zone_width=1.0)
    z = np.full(2, 1200.0)
    restored = f.restore(np.array([1300.0, 1700.0]), np.full(2, 1500.0), z)
    assert restored[1] - restored[0] == pytest.approx(-throw, abs=0.5)
    if sense == "normal":
        assert restored[1] < restored[0]   # hanging wall reaches a shallower unit
    else:
        assert restored[1] > restored[0]


def test_a_sealing_fault_blocks_transport_and_a_transmissive_one_does_not():
    sealing = Fault(transmissibility=0.0, zone_width=50.0)
    open_fault = Fault(transmissibility=1.0, zone_width=50.0)
    at_plane = (np.array([1500.0]), np.array([1500.0]), np.array([1500.0]))
    assert float(sealing.transmissibility_multiplier(*at_plane)[0]) == pytest.approx(0.0)
    assert float(open_fault.transmissibility_multiplier(*at_plane)[0]) == pytest.approx(1.0)


def test_a_finite_fault_does_nothing_beyond_its_extent():
    f = Fault(origin=(1500.0, 1500.0, 1200.0), strike=0.0, dip=90.0, throw=40.0,
              strike_extent=200.0, zone_width=1.0)
    far = f.hanging_wall_fraction(np.array([1700.0]), np.array([2500.0]),
                                  np.array([1200.0]))
    assert float(far[0]) == pytest.approx(0.0)


def test_fault_transmissibility_is_bounded():
    with pytest.raises(ConfigError, match="transmissibility"):
        Fault(transmissibility=1.5)


# --------------------------------------------------------------------- builder
def test_layers_are_stacked_in_depth_order(grid):
    layers = [Layer("a", Flat(0.0), "shale"), Layer("b", Flat(800.0), "sandy_shale"),
              Layer("c", Flat(1600.0), "clean_sandstone")]
    model = build_geology(grid, layers)
    z = grid.axis(2)[None, None, :] * np.ones(grid.shape)
    assert np.all(model.layer_index[z < 800.0] == 0)
    assert np.all(model.layer_index[(z >= 800.0) & (z < 1600.0)] == 1)
    assert np.all(model.layer_index[z >= 1600.0] == 2)


def test_crossing_horizons_are_refused(grid):
    layers = [Layer("a", Flat(1000.0), "shale"), Layer("b", Flat(600.0), "shale")]
    with pytest.raises(ValidationError, match="above layer"):
        build_geology(grid, layers)


def test_composition_honours_shale_volume_and_sums_to_one(grid):
    layers, _ = template("anticline")
    model = build_geology(grid, layers)
    comp = model.composition
    assert np.allclose(sum(comp.values()), 1.0)
    assert np.allclose(comp["clay"], model.vsh)


def test_a_fault_offsets_the_reservoir_top_by_its_throw():
    """Compare against the same structure without the fault, so the fold's
    own relief cancels and only the throw is left."""
    fine = Grid3D.from_bounds(((0, 3000), (0, 3000), (0, 2200)), (50, 50, 10))
    throw = 45.0
    layers, faults = template("fault_compartment", throw=throw)
    faulted = build_geology(fine, layers, faults)
    unfaulted = build_geology(fine, layers)

    def reservoir_top(model):
        reservoir = model.layer_index == 3
        iy = fine.ny // 2
        return np.array([
            fine.axis(2)[np.where(reservoir[ix, iy])[0][0]]
            if reservoir[ix, iy].any() else np.nan
            for ix in range(fine.nx)
        ])

    offset = reservoir_top(faulted) - reservoir_top(unfaulted)
    hanging_wall = offset[offset > 1.0]
    footwall = offset[np.abs(offset) < 1.0]
    assert hanging_wall.size > 5 and footwall.size > 5
    assert np.nanmedian(hanging_wall) == pytest.approx(throw, abs=fine.dz)


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_every_template_builds_a_model_with_reservoir_cells(grid, name):
    layers, faults = template(name)
    model = build_geology(grid, layers, faults)
    assert model.reservoir_mask.any()
    assert np.all(model.porosity > 0) and np.all(model.porosity < 0.5)
    assert np.all((model.vsh >= 0) & (model.vsh <= 1))
    assert "layers" in model.summary()


def test_unknown_template_is_named():
    with pytest.raises(ConfigError, match="unknown geological template"):
        template("atlantis")


def test_unknown_facies_is_named(grid):
    with pytest.raises(ConfigError, match="unknown facies"):
        build_geology(grid, [Layer("a", Flat(0.0), "unobtainium")])


def test_facies_catalogue_is_self_consistent():
    for name, facies in FACIES.items():
        assert abs(sum(facies.composition.values()) - 1.0) < 1e-9
        assert facies.porosity[0] <= facies.porosity[1]
