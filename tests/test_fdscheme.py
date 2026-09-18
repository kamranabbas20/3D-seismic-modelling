import numpy as np
import pytest

from sim3d.core.errors import ConfigError, StabilityError
from sim3d.wave import operators as ops
from sim3d.wave.fdscheme import (
    STAGGERED_COEFFS, cells_per_wavelength, check_dt, coefficients,
    effective_wavenumber, max_courant, max_stable_dt, phase_velocity_error,
    recommend_spacing, required_ppw,
)


@pytest.mark.parametrize("order", sorted(STAGGERED_COEFFS))
def test_coefficients_reproduce_the_exact_derivative_to_their_order(order):
    """Taylor condition: sum_k c_k (k - 1/2)^(2m+1) = delta_{m,0} / 2."""
    c = coefficients(order)
    k = np.arange(1, len(c) + 1) - 0.5
    assert np.sum(c * k) == pytest.approx(0.5, rel=1e-12)
    for m in range(1, len(c)):
        assert np.sum(c * k ** (2 * m + 1)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("order,expected", [(2, 2.0), (4, 4.0), (6, 6.0)])
def test_operator_converges_at_its_nominal_order(order, expected):
    errs = []
    for n in (64, 128, 256):
        dx = 2 * np.pi / n
        x = np.arange(n) * dx
        f = np.sin(x)[:, None, None]
        d = ops.forward_diff(f, 0, dx, order)
        exact = np.cos((np.arange(n - 1) + 0.5) * dx)
        h = order // 2
        errs.append(np.max(np.abs(d[h:-h, 0, 0] - exact[h:-h])))
    rate = np.log2(errs[0] / errs[1])
    assert rate == pytest.approx(expected, abs=0.15)


def test_effective_wavenumber_approaches_the_true_one_as_sampling_improves():
    for order in (2, 4, 8):
        errs = [
            abs(effective_wavenumber(2 * np.pi / ppw, order) / (2 * np.pi / ppw) - 1.0)
            for ppw in (4.0, 8.0, 40.0)
        ]
        assert errs[0] > errs[1] > errs[2]
        assert errs[-1] < 2e-3  # 2nd order at 40 cells/wavelength: (k dx)^2/24


def test_higher_order_is_more_accurate_at_the_same_sampling():
    k_dx = 2 * np.pi / 6.0
    errs = [abs(effective_wavenumber(k_dx, o) / k_dx - 1.0) for o in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_courant_limit_matches_the_published_value_for_8th_order():
    # 8th-order staggered, isotropic 3D: dt <= 0.449 dx / Vmax.
    dt = max_stable_dt((10.0, 10.0, 10.0), 2000.0, 8, safety=1.0)
    assert dt * 2000.0 / 10.0 == pytest.approx(0.4489, abs=1e-3)
    assert max_courant(2) == pytest.approx(1.0)


def test_unstable_dt_raises_and_is_not_corrected():
    with pytest.raises(StabilityError, match="will not"):
        check_dt(5e-3, (7.5, 7.5, 7.5), 4500.0, 8)


def test_stable_dt_reports_the_numbers_the_user_must_see():
    r = check_dt(1e-3, (10.0, 10.0, 10.0), 2000.0, 8)
    assert 0 < r.cfl_ratio < 1
    assert "CFL" in r.describe()


def test_higher_spatial_order_needs_fewer_cells_per_wavelength():
    spatial_only = [required_ppw(o, 0.01, courant=1e-4) for o in (2, 4, 8)]
    assert spatial_only[0] > spatial_only[1] > spatial_only[2]
    assert spatial_only[2] < 4.0  # 8th order: about 3.3 cells per wavelength


def test_at_the_stability_limit_time_error_sets_the_floor():
    """Raising the spatial order past 4th buys little unless dt also drops.

    This is the diagnostic that replaces the folklore '8 cells per
    wavelength': the requirement depends on the Courant number, not only
    on the operator.
    """
    at_limit = [required_ppw(o, 0.01) for o in (4, 8)]
    assert at_limit[1] > 4.5
    assert abs(at_limit[0] - at_limit[1]) < 1.0
    assert required_ppw(8, 0.01, courant=0.2) < required_ppw(8, 0.01)


def test_dispersion_error_falls_as_sampling_improves():
    errs = [phase_velocity_error(g, 8, 0.7) for g in (4.0, 8.0, 16.0)]
    assert errs[0] > errs[1] > errs[2]


def test_recommended_spacing_follows_the_wavelength():
    r = recommend_spacing(1800.0, 30.0, order=8, tolerance=0.01)
    assert r.min_wavelength == pytest.approx(60.0)
    assert cells_per_wavelength(r.max_spacing, 1800.0, 30.0) == pytest.approx(
        r.points_per_wavelength, rel=1e-9
    )
    # Halving Fmax must roughly double the allowed spacing.
    coarse = recommend_spacing(1800.0, 15.0, order=8, tolerance=0.01)
    assert coarse.max_spacing == pytest.approx(2 * r.max_spacing, rel=1e-9)


def test_unknown_order_is_rejected():
    with pytest.raises(ConfigError, match="not available"):
        coefficients(7)
