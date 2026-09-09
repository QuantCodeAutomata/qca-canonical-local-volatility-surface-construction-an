"""
Tests for src/transforms.py — Laplace transform ingredients.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HestonNandiParams
from src.transforms import (
    compute_d, laplace_ingredients, Lt, Mt, Lt_and_Mt,
    A_func, s_star, x_star
)


@pytest.fixture
def default_params():
    return HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)


class TestDerivedParameters:
    def test_beta_formula(self, default_params):
        """beta = kappa + xi/2."""
        p = default_params
        assert abs(p.beta - (p.kappa + p.xi / 2)) < 1e-12

    def test_a_formula(self, default_params):
        """a = kappa * theta."""
        p = default_params
        assert abs(p.a - p.kappa * p.theta) < 1e-12

    def test_b_formula(self, default_params):
        """b = -kappa."""
        p = default_params
        assert abs(p.b - (-p.kappa)) < 1e-12

    def test_c_formula(self, default_params):
        """c = xi."""
        p = default_params
        assert abs(p.c - p.xi) < 1e-12

    def test_rho_fixed(self):
        """rho must be -1."""
        with pytest.raises(AssertionError):
            HestonNandiParams(rho=0.0)

    def test_beta_positive(self, default_params):
        """beta = kappa + xi/2 > 0 always (required by paper)."""
        assert default_params.beta > 0


class TestComputeD:
    def test_principal_branch_real_positive(self, default_params):
        """d(z) has non-negative real part for positive real z."""
        z = np.array([1.0 + 0j, 2.0 + 0j, 5.0 + 0j])
        d = compute_d(z, default_params)
        assert np.all(d.real >= 0)

    def test_principal_branch_complex(self, default_params):
        """d(z) has non-negative real part on Bromwich contour."""
        z = np.array([5.0 + 1j, 5.0 + 2j, 5.0 - 1j])
        d = compute_d(z, default_params)
        assert np.all(d.real >= -1e-12)

    def test_d_squared_equals_discriminant(self, default_params):
        """Verify d^2 = b^2 + 2*beta*c^2*z."""
        p = default_params
        z = np.array([1.0 + 0.5j, 2.0 - 0.5j])
        d = compute_d(z, p)
        discriminant = p.b**2 + 2 * p.beta * p.c**2 * z
        np.testing.assert_allclose(d**2, discriminant, rtol=1e-10)

    def test_d_at_zero(self, default_params):
        """d(0) = |b| = kappa."""
        p = default_params
        z = np.array([0.0 + 0j])
        d = compute_d(z, p)
        assert abs(d[0].real - abs(p.b)) < 1e-10


class TestLaplaceIngredients:
    def test_ingredients_shapes(self, default_params):
        """All ingredients have same shape as input z."""
        z = np.array([1.0 + 2j, 3.0 - 1j, 5.0 + 0j])
        ingr = laplace_ingredients(z, t=1.0, params=default_params)
        for key in ['d', 'r_plus', 'r_minus', 'h', 'B_t', 'Acal_t', 'Hcal_t', 'Jcal_t']:
            assert ingr[key].shape == z.shape, f"Shape mismatch for {key}"

    def test_Hcal_t_positive_real_z(self, default_params):
        """Hcal_t(z) should be real and in (0,1) for large positive real z."""
        z = np.array([10.0 + 0j])
        ingr = laplace_ingredients(z, t=1.0, params=default_params)
        # Hcal_t = e^{-dt} * ((1-h)/(1-h*e^{-dt}))^2
        # For large real z and t>0, Hcal_t -> 0
        assert ingr['Hcal_t'][0].imag < 1e-8

    def test_Lt_real_for_real_positive_z(self, default_params):
        """L_t(z) is real for real positive z (it's a Laplace transform)."""
        z = np.array([1.0 + 0j, 2.0 + 0j])
        L = Lt(z, t=1.0, params=default_params)
        np.testing.assert_allclose(L.imag, 0, atol=1e-10)

    def test_Lt_between_0_and_1(self, default_params):
        """L_t(z) = E[e^{-z Q_t}] should be in (0, 1] for z >= 0 (real)."""
        z = np.array([0.01 + 0j, 0.5 + 0j, 1.0 + 0j, 2.0 + 0j, 5.0 + 0j])
        L = Lt(z, t=1.0, params=default_params)
        assert np.all(L.real > 0), "L_t should be positive"
        assert np.all(L.real <= 1.0 + 1e-10), "L_t should be <= 1"

    def test_Mt_nonneg_for_real_z(self, default_params):
        """M_t(z) >= 0 for real positive z (it's E[v_t e^{-z Q_t}])."""
        z = np.array([0.5 + 0j, 1.0 + 0j, 2.0 + 0j])
        M = Mt(z, t=1.0, params=default_params)
        assert np.all(M.real >= -1e-10)

    def test_Lt_and_Mt_consistent(self, default_params):
        """Lt_and_Mt returns same values as Lt and Mt separately."""
        z = np.array([1.0 + 0.5j, 2.0 - 0.3j])
        L, M = Lt_and_Mt(z, t=1.0, params=default_params)
        L_sep = Lt(z, t=1.0, params=default_params)
        M_sep = Mt(z, t=1.0, params=default_params)
        np.testing.assert_allclose(L, L_sep, rtol=1e-12)
        np.testing.assert_allclose(M, M_sep, rtol=1e-12)

    def test_M_less_than_v0_times_L(self, default_params):
        """M_t(z) = E[v_t e^{-z Q_t}] <= v0 * E[e^{-z Q_t}] = v0 * L_t(z) approximately for small t."""
        # At t=0+, v_t ≈ v0, so M ≈ v0 * L
        z = np.array([1.0 + 0j])
        L, M = Lt_and_Mt(z, t=0.001, params=default_params)
        assert M.real <= default_params.v0 * L.real + 1e-4


class TestAFuncAndBarrier:
    def test_A_func_at_S0(self, default_params):
        """A(t, S0) = v0 + kappa*theta*t."""
        p = default_params
        t = 1.0
        s = np.array([p.S0])
        A = A_func(t, s, p)
        expected = p.v0 + p.kappa * p.theta * t
        np.testing.assert_allclose(A, expected, rtol=1e-12)

    def test_A_func_monotone_in_s(self, default_params):
        """A(t,s) is strictly decreasing in s (since -xi*log(s/S0) decreases as s increases)."""
        p = default_params
        t = 1.0
        s = np.linspace(10.0, 500.0, 20)
        A = A_func(t, s, p)
        assert np.all(np.diff(A) < 0), "A(t,s) should be strictly decreasing in s"

    def test_A_zero_at_barrier(self, default_params):
        """A(t, s*(t)) = 0."""
        p = default_params
        t = 1.0
        sstar = s_star(t, p)
        A = A_func(t, np.array([sstar]), p)
        np.testing.assert_allclose(A, 0.0, atol=1e-12)

    def test_A_positive_below_barrier(self, default_params):
        """A(t,s) > 0 for s < s*(t)."""
        p = default_params
        t = 1.0
        sstar = s_star(t, p)
        s_below = np.linspace(0.01 * sstar, 0.99 * sstar, 10)
        A = A_func(t, s_below, p)
        assert np.all(A > 0)

    def test_A_negative_above_barrier(self, default_params):
        """A(t,s) < 0 for s > s*(t)."""
        p = default_params
        t = 1.0
        sstar = s_star(t, p)
        s_above = np.linspace(1.01 * sstar, 3.0 * sstar, 10)
        A = A_func(t, s_above, p)
        assert np.all(A < 0)

    def test_s_star_formula(self, default_params):
        """s*(t) = S0 * exp((v0 + kappa*theta*t) / xi)."""
        p = default_params
        t = 1.5
        expected = p.S0 * np.exp((p.v0 + p.kappa * p.theta * t) / p.xi)
        assert abs(s_star(t, p) - expected) < 1e-10

    def test_s_star_increasing_in_t(self, default_params):
        """s*(t) is increasing in t (since v0 + kappa*theta*t increases)."""
        p = default_params
        t_grid = np.linspace(0.1, 3.0, 20)
        sstars = np.array([s_star(t, p) for t in t_grid])
        assert np.all(np.diff(sstars) > 0)

    def test_x_star_equals_log_s_star(self, default_params):
        """x*(t) = log(s*(t))."""
        p = default_params
        t = 1.0
        assert abs(x_star(t, p) - np.log(s_star(t, p))) < 1e-12
