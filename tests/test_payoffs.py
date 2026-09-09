"""
Tests for src/payoffs.py — payoff functions and diagnostics.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.payoffs import (
    call_payoff, spot_call_price, integrated_var_call_price,
    vol_swap_expectation, soft_call_payoff, mc_confidence_interval
)


class TestCallPayoff:
    def test_itm(self):
        """(I - K)^+ = I - K when I > K."""
        assert abs(call_payoff(np.array([1.0]), 0.5)[0] - 0.5) < 1e-12

    def test_otm(self):
        """(I - K)^+ = 0 when I <= K."""
        assert call_payoff(np.array([0.3]), 0.5)[0] == 0.0

    def test_at_money(self):
        """(K - K)^+ = 0."""
        assert call_payoff(np.array([0.5]), 0.5)[0] == 0.0

    def test_shape_preserved(self):
        """Output shape matches input."""
        I = np.random.rand(100)
        out = call_payoff(I, 0.5)
        assert out.shape == I.shape

    def test_nonneg(self):
        """Call payoff is always nonneg."""
        I = np.random.randn(200)
        out = call_payoff(I, 0.3)
        assert np.all(out >= 0)


class TestSpotCallPrice:
    def test_itm_spot_call(self):
        """Deep ITM spot call ≈ S_T - K (since almost all paths above K)."""
        S_T = np.full(10000, 200.0)  # all paths at 200
        price = spot_call_price(S_T, K=100.0)
        assert abs(price - 100.0) < 0.1

    def test_otm_spot_call_is_zero(self):
        """Deep OTM spot call ≈ 0."""
        S_T = np.full(10000, 50.0)  # all paths at 50
        price = spot_call_price(S_T, K=100.0)
        assert abs(price) < 0.01

    def test_call_nonneg(self):
        """Spot call price is always nonneg."""
        S_T = np.random.lognormal(0, 0.2, 1000)
        K_grid = np.linspace(0.5, 2.0, 10)
        for K in K_grid:
            price = spot_call_price(S_T, K)
            assert price >= 0


class TestIntegratedVarCallPrice:
    def test_positive_for_atm(self):
        """ATM integrated var call price > 0 when some paths exceed mean."""
        I_T = np.random.exponential(0.1, 10000)
        K = float(np.mean(I_T))
        price = integrated_var_call_price(I_T, K)
        assert price > 0

    def test_zero_for_deeply_otm(self):
        """Deep OTM integrated var call = 0."""
        I_T = np.full(1000, 0.05)
        price = integrated_var_call_price(I_T, K=1.0)
        assert abs(price) < 1e-10

    def test_equal_mean_for_K_leq_0(self):
        """E[(I - K)^+] = E[I] - K for K <= 0 (since I >= 0 a.s.)."""
        I_T = np.abs(np.random.randn(10000)) * 0.1 + 0.05
        K = -0.01
        price = integrated_var_call_price(I_T, K)
        expected = float(np.mean(I_T)) - K
        assert abs(price - expected) < 0.01


class TestVolSwapExpectation:
    def test_nonneg(self):
        """Volatility swap expectation is nonneg."""
        I_T = np.random.exponential(0.1, 1000)
        T = 1.0
        vs = vol_swap_expectation(I_T, T)
        assert vs >= 0

    def test_zero_for_zero_variance(self):
        """sqrt(0/T) = 0."""
        I_T = np.zeros(100)
        assert vol_swap_expectation(I_T, T=1.0) == 0.0

    def test_jensen_inequality(self):
        """By Jensen, E[sqrt(I/T)] <= sqrt(E[I/T])."""
        I_T = np.random.exponential(0.1, 10000)
        T = 1.0
        vs = vol_swap_expectation(I_T, T)
        mean_iv = float(np.mean(I_T))
        assert vs <= np.sqrt(mean_iv / T) + 1e-10


class TestSoftCallPayoff:
    def test_convergence_to_hard_call(self):
        """Soft call converges to hard call as m -> infinity."""
        i = np.array([0.5])
        K = 0.3
        hard = call_payoff(i, K)[0]
        m_vals = [5, 20, 100, 1000]
        errors = [abs(soft_call_payoff(i, K, m)[0] - hard) for m in m_vals]
        # Errors should decrease as m increases
        assert all(errors[i] > errors[i+1] for i in range(len(errors)-1)), (
            f"Soft call not converging: errors = {errors}")

    def test_nonneg(self):
        """Soft call payoff is always positive."""
        i = np.random.randn(100)
        K = 0.0
        m = 10
        out = soft_call_payoff(i, K, m)
        assert np.all(out > 0)

    def test_smooth(self):
        """Soft call is smooth (no discontinuity at K)."""
        i = np.linspace(-1, 1, 1000)
        K = 0.0
        m = 10
        out = soft_call_payoff(i, K, m)
        # Finite differences should be bounded
        diffs = np.abs(np.diff(out))
        assert np.all(diffs < 0.01), f"Max diff={diffs.max():.4e}"

    def test_approx_linear_far_itm(self):
        """Far ITM: soft_call(i, K, m) ≈ i - K."""
        i = np.array([10.0])
        K = 0.0
        m = 50
        out = soft_call_payoff(i, K, m)
        assert abs(out[0] - (i[0] - K)) < 1e-5

    def test_approx_zero_far_otm(self):
        """Far OTM: soft_call(i, K, m) ≈ 0."""
        i = np.array([-10.0])
        K = 0.0
        m = 50
        out = soft_call_payoff(i, K, m)
        assert abs(out[0]) < 1e-5


class TestMonteCarloCI:
    def test_ci_width_decreases_with_n(self):
        """CI width decreases as n increases (mean = 0.5, std = std)."""
        rng = np.random.default_rng(42)
        x_small = rng.uniform(0, 1, 100)
        x_large = rng.uniform(0, 1, 10000)
        mean_s, se_s, lo_s, hi_s = mc_confidence_interval(x_small, alpha=0.05)
        mean_l, se_l, lo_l, hi_l = mc_confidence_interval(x_large, alpha=0.05)
        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_ci_contains_true_mean(self):
        """95% CI should contain true mean for normal samples."""
        rng = np.random.default_rng(42)
        true_mean = 5.0
        x = rng.normal(loc=true_mean, scale=1.0, size=10000)
        mean, se, lo, hi = mc_confidence_interval(x, alpha=0.05)
        assert lo < true_mean < hi, (
            f"True mean {true_mean} not in CI [{lo:.4f}, {hi:.4f}]")

    def test_se_formula(self):
        """SE = std / sqrt(n)."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, se, lo, hi = mc_confidence_interval(x, alpha=0.05)
        expected_se = np.std(x, ddof=1) / np.sqrt(len(x))
        assert abs(se - expected_se) < 1e-12
