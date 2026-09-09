"""
Tests for src/localvol_sim.py — Local-vol simulation.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HestonNandiParams, NumericalConfig
from src.localvol_surface import LocalVolSurface
from src.localvol_sim import simulate_localvol, terminal_spot_lv, terminal_integrated_variance_lv
from src.transforms import s_star, x_star


@pytest.fixture(scope="module")
def built_surface():
    params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    num_cfg = NumericalConfig(
        t_min=0.01, t_max=2.1, n_time=8,
        q_min=1e-5, q_max=3.0, n_q=60,
        q_small=1e-3, euler_M=12, euler_A=20.0
    )
    surface = LocalVolSurface(params, num_cfg)
    surface.build(verbose=False)
    return surface


@pytest.fixture
def default_params():
    return HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)


class TestLocalVolSimulation:
    def test_output_structure(self, built_surface, default_params):
        """simulate_localvol returns expected keys."""
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=50, n_paths=100, seed=42)
        for key in ['t_grid', 'X', 'S', 'I']:
            assert key in result

    def test_shapes(self, built_surface, default_params):
        """Output arrays have correct shapes."""
        n_paths, n_steps = 200, 50
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=n_steps, n_paths=n_paths, seed=42)
        assert result['X'].shape == (n_paths, n_steps)
        assert result['S'].shape == (n_paths, n_steps)
        assert result['I'].shape == (n_paths, n_steps)

    def test_spot_positive(self, built_surface, default_params):
        """Local-vol spot paths S_t > 0."""
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=50, n_paths=500, seed=42)
        assert np.all(result['S'] > 0)

    def test_integrated_variance_nonneg(self, built_surface, default_params):
        """Integrated variance I_t >= 0."""
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=50, n_paths=200, seed=42)
        assert np.all(result['I'] >= 0)

    def test_spot_below_barrier(self, built_surface, default_params):
        """Local-vol spot X_t <= x*(t) (barrier enforced)."""
        p = default_params
        result = simulate_localvol(built_surface, p, T=1.0, n_steps=100,
                                   n_paths=200, seed=42)
        t_grid = result['t_grid']
        X = result['X']
        for j, t in enumerate(t_grid):
            xstar = x_star(t, p)
            above = X[:, j] > xstar + 1e-9
            assert not above.any(), (
                f"X_t exceeds x*(t) at t={t:.3f}: {above.sum()} violations")

    def test_integrated_variance_nondecreasing(self, built_surface, default_params):
        """I_t is nondecreasing in t."""
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=50, n_paths=200, seed=42)
        I = result['I']
        assert np.all(np.diff(I, axis=1) >= -1e-12)

    def test_terminal_extractions(self, built_surface, default_params):
        """terminal_spot_lv and terminal_integrated_variance_lv return correct columns."""
        result = simulate_localvol(built_surface, default_params, T=1.0,
                                   n_steps=50, n_paths=100, seed=42)
        np.testing.assert_array_equal(terminal_spot_lv(result), result['S'][:, -1])
        np.testing.assert_array_equal(
            terminal_integrated_variance_lv(result), result['I'][:, -1])

    def test_reproducibility(self, built_surface, default_params):
        """Same seed gives same paths."""
        r1 = simulate_localvol(built_surface, default_params, T=0.5,
                               n_steps=30, n_paths=50, seed=77)
        r2 = simulate_localvol(built_surface, default_params, T=0.5,
                               n_steps=30, n_paths=50, seed=77)
        np.testing.assert_array_equal(r1['X'], r2['X'])
