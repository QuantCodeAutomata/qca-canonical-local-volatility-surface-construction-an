"""
Tests for src/heston_nandi.py — Heston-Nandi simulation.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HestonNandiParams
from src.heston_nandi import (
    simulate_cir_exact, integrate_cir_trapezoidal,
    simulate_heston_nandi, terminal_spot, terminal_integrated_variance
)
from src.transforms import s_star, A_func


@pytest.fixture
def default_params():
    return HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)


class TestCIRExact:
    def test_output_shapes(self, default_params):
        """CIR simulation returns correct shape."""
        p = default_params
        rng = np.random.default_rng(42)
        t_grid = np.array([0.25, 0.5, 0.75, 1.0])
        v = simulate_cir_exact(p.v0, p.kappa, p.theta, p.xi, t_grid, n_paths=100, rng=rng)
        assert v.shape == (100, 4)

    def test_variance_nonneg(self, default_params):
        """CIR variance process is nonneg."""
        p = default_params
        rng = np.random.default_rng(123)
        t_grid = np.linspace(0.1, 2.0, 20)
        v = simulate_cir_exact(p.v0, p.kappa, p.theta, p.xi, t_grid, n_paths=1000, rng=rng)
        assert np.all(v >= 0), f"CIR variance should be nonneg, min={v.min():.4e}"

    def test_long_run_mean(self, default_params):
        """Long-run mean of CIR process is theta."""
        p = default_params
        rng = np.random.default_rng(999)
        t_grid = np.array([10.0])  # large t -> E[v_t] -> theta
        v = simulate_cir_exact(p.v0, p.kappa, p.theta, p.xi, t_grid, n_paths=10000, rng=rng)
        mean_v = float(np.mean(v[:, 0]))
        assert abs(mean_v - p.theta) < 0.003, (
            f"Long-run mean {mean_v:.5f} not close to theta={p.theta}")

    def test_short_run_mean(self, default_params):
        """At t=0+, E[v_t] ≈ v0."""
        p = default_params
        rng = np.random.default_rng(77)
        t_grid = np.array([0.001])
        v = simulate_cir_exact(p.v0, p.kappa, p.theta, p.xi, t_grid, n_paths=5000, rng=rng)
        mean_v = float(np.mean(v[:, 0]))
        # E[v_dt] = v0*exp(-kappa*dt) + theta*(1-exp(-kappa*dt)) ≈ v0
        expected = p.v0 * np.exp(-p.kappa * 0.001) + p.theta * (1 - np.exp(-p.kappa * 0.001))
        assert abs(mean_v - expected) < 0.001


class TestHestonNandiSimulation:
    def test_output_structure(self, default_params):
        """simulate_heston_nandi returns all expected keys."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=100, seed=42)
        for key in ['t_grid', 'v', 'I', 'S', 'log_S', 'pathwise_residual']:
            assert key in result, f"Missing key: {key}"

    def test_shapes(self, default_params):
        """Output arrays have correct shapes."""
        n_paths, n_steps = 200, 50
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=n_steps,
                                       n_paths=n_paths, seed=42)
        assert result['v'].shape == (n_paths, n_steps)
        assert result['I'].shape == (n_paths, n_steps)
        assert result['S'].shape == (n_paths, n_steps)

    def test_variance_nonneg(self, default_params):
        """Variance paths are nonneg."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=500, seed=42)
        assert np.all(result['v'] >= 0)

    def test_integrated_variance_nonneg_increasing(self, default_params):
        """Integrated variance I_t is nonneg and increasing in t."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=200, seed=42)
        I = result['I']
        assert np.all(I >= 0)
        # Trapezoidal integral should be non-decreasing
        assert np.all(np.diff(I, axis=1) >= -1e-10)

    def test_spot_positive(self, default_params):
        """Spot paths are strictly positive."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=500, seed=42)
        assert np.all(result['S'] > 0)

    def test_pathwise_identity_residual_small(self, default_params):
        """Pathwise identity residual is small (discretization error only)."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=200, n_paths=1000, seed=42)
        residuals = result['pathwise_residual']
        rmse = float(np.sqrt(np.mean(residuals**2)))
        # Should be small due to consistent construction via pathwise identity
        assert rmse < 0.05, f"Pathwise residual RMSE too large: {rmse:.4e}"

    def test_barrier_condition(self, default_params):
        """S_t < s*(t) almost surely."""
        p = default_params
        result = simulate_heston_nandi(p, T=1.0, n_steps=100, n_paths=500, seed=42)
        t_grid = result['t_grid']
        S_all = result['S']
        for j, t in enumerate(t_grid):
            sstar = s_star(t, p)
            violations = int(np.sum(S_all[:, j] >= sstar))
            assert violations == 0, f"Barrier violated at t={t:.3f}: {violations} paths"

    def test_terminal_spot_extraction(self, default_params):
        """terminal_spot returns last column of S."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=100, seed=42)
        S_T = terminal_spot(result)
        np.testing.assert_array_equal(S_T, result['S'][:, -1])

    def test_terminal_iv_extraction(self, default_params):
        """terminal_integrated_variance returns last column of I."""
        result = simulate_heston_nandi(default_params, T=1.0, n_steps=50, n_paths=100, seed=42)
        I_T = terminal_integrated_variance(result)
        np.testing.assert_array_equal(I_T, result['I'][:, -1])

    def test_reproducibility(self, default_params):
        """Same seed gives same result."""
        r1 = simulate_heston_nandi(default_params, T=0.5, n_steps=50, n_paths=100, seed=99)
        r2 = simulate_heston_nandi(default_params, T=0.5, n_steps=50, n_paths=100, seed=99)
        np.testing.assert_array_equal(r1['v'], r2['v'])
        np.testing.assert_array_equal(r1['S'], r2['S'])
