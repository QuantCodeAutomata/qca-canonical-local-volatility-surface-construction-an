"""
Tests for src/inversion.py — inverse Laplace and phi computation.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HestonNandiParams
from src.transforms import Lt, Mt
from src.inversion import (
    euler_invert_array, compute_gt_mt, phi_asymptotic, compute_phi
)


@pytest.fixture
def default_params():
    return HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)


class TestEulerInversion:
    def test_simple_exponential(self):
        """Test inversion of F(s) = 1/(s+a), which inverts to f(t) = e^{-at}."""
        a = 2.0
        t = 0.5
        F = lambda s: 1.0 / (s + a)
        result = euler_invert_array(F, t, M=15, A=23.0)
        expected = np.exp(-a * t)
        assert abs(result - expected) < 1e-5, f"Expected {expected:.6f}, got {result:.6f}"

    def test_constant_transform(self):
        """F(s) = 1/s inverts to f(t) = 1."""
        F = lambda s: 1.0 / s
        t = 1.0
        result = euler_invert_array(F, t, M=15, A=23.0)
        assert abs(result - 1.0) < 1e-5

    def test_laplace_of_cdf(self):
        """F(s) = 1/(s*(s+a)) inverts to (1 - e^{-at})/a."""
        a = 1.5
        t = 0.8
        F = lambda s: 1.0 / (s * (s + a))
        result = euler_invert_array(F, t, M=15, A=23.0)
        expected = (1 - np.exp(-a * t)) / a
        assert abs(result - expected) < 1e-4


class TestComputeGtMt:
    def test_gt_positive(self, default_params):
        """g_t(q) should be non-negative for q > 0.

        Euler inversion has absolute accuracy ~e^{-A}; for q in the far tail of Q_t
        the density is near zero and can appear as a small negative number within
        numerical tolerance.  We allow -1e-6 as the practical floor.
        """
        q = np.array([0.1, 0.5, 1.0])
        t = 1.0
        g, m = compute_gt_mt(q, t, default_params, M=12, A=20.0)
        assert np.all(g > -1e-6), f"g_t should be non-negative (within tolerance), got {g}"

    def test_mt_positive(self, default_params):
        """m_t(q) should be non-negative for q > 0 (since v_t >= 0 a.s.).

        Same Euler noise caveat as test_gt_positive; allow -1e-7 tolerance.
        """
        q = np.array([0.1, 0.5, 1.0])
        t = 1.0
        g, m = compute_gt_mt(q, t, default_params, M=12, A=20.0)
        assert np.all(m > -1e-7), f"m_t should be nonneg (within tolerance), got {m}"

    def test_mt_leq_v0_times_gt(self, default_params):
        """m_t(q) = E[v_t | Q_t=q] * g_t(q) and E[v_t|Q_t=q] <= Q_t = q."""
        # phi_t(q) = m_t(q)/g_t(q) = E[v_t | Q_t=q] <= q (since v_t <= Q_t = v_t + beta*I)
        q = np.array([0.1, 0.5, 1.0, 2.0])
        t = 1.0
        g, m_val = compute_gt_mt(q, t, default_params, M=12, A=20.0)
        valid = g > 1e-10
        phi = np.where(valid, m_val / g, 0.0)
        # phi_t(q) = E[v_t | Q_t = q] <= q because v_t <= Q_t
        assert np.all(phi[valid] <= q[valid] + 1e-4), f"phi should <= q, got {phi}"

    def test_gt_zero_for_nonpositive_q(self, default_params):
        """g_t(q) should be 0 for q <= 0."""
        q = np.array([-1.0, 0.0])
        t = 1.0
        g, m = compute_gt_mt(q, t, default_params)
        assert np.all(g == 0.0)


class TestPhiAsymptotic:
    def test_asymptotic_formula(self, default_params):
        """phi_asymptotic(q) = 4a/(beta*(v0+a*t)^2) * q^2."""
        p = default_params
        t = 1.0
        q = np.array([0.01, 0.02])
        result = phi_asymptotic(q, t, p)
        a = p.a
        beta = p.beta
        v0_at = p.v0 + a * t
        coeff = 4 * a / (beta * v0_at**2)
        expected = coeff * q**2
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_asymptotic_nonneg(self, default_params):
        """phi_asymptotic >= 0 for q >= 0."""
        q = np.linspace(0, 0.1, 20)
        result = phi_asymptotic(q, 1.0, default_params)
        assert np.all(result >= 0)

    def test_asymptotic_at_zero(self, default_params):
        """phi_asymptotic(0) = 0."""
        result = phi_asymptotic(np.array([0.0]), 1.0, default_params)
        assert abs(result[0]) < 1e-15

    def test_quadratic_scaling(self, default_params):
        """phi_asymptotic(2q)/phi_asymptotic(q) = 4."""
        q = np.array([0.01])
        phi_q = phi_asymptotic(q, 1.0, default_params)
        phi_2q = phi_asymptotic(2 * q, 1.0, default_params)
        np.testing.assert_allclose(phi_2q / phi_q, 4.0, rtol=1e-10)


class TestComputePhi:
    def test_phi_nonneg(self, default_params):
        """phi_t(q) >= 0 for q > 0."""
        q = np.linspace(0.001, 3.0, 30)
        phi = compute_phi(q, 1.0, default_params, q_small=1e-3, M=12, A=20.0)
        assert np.all(phi >= -1e-8), f"phi should be nonneg, min={phi.min():.4e}"

    def test_phi_zero_for_nonpositive_q(self, default_params):
        """phi_t(q) = 0 for q <= 0."""
        q = np.array([-0.5, 0.0, -1.0])
        phi = compute_phi(q, 1.0, default_params)
        assert np.all(phi == 0.0)

    def test_phi_uses_asymptotic_near_zero(self, default_params):
        """phi_t(q) for q < q_small uses Proposition 11 asymptotic."""
        p = default_params
        t = 1.0
        q_small_test = 0.01
        q = np.array([q_small_test / 2])  # below threshold
        phi = compute_phi(q, t, p, q_small=q_small_test, M=12, A=20.0)
        expected = phi_asymptotic(q, t, p)
        np.testing.assert_allclose(phi, expected, rtol=1e-10)

    def test_phi_bounded_by_q(self, default_params):
        """phi_t(q) = E[v_t|Q_t=q] <= q because v_t <= Q_t."""
        q = np.array([0.1, 0.5, 1.0, 2.0])
        phi = compute_phi(q, 1.0, default_params, q_small=1e-4, M=12, A=20.0)
        assert np.all(phi <= q + 1e-4)
