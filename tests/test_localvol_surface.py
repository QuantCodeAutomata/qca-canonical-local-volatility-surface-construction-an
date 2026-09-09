"""
Tests for src/localvol_surface.py — LocalVolSurface construction.
"""
import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HestonNandiParams, NumericalConfig
from src.localvol_surface import LocalVolSurface
from src.transforms import A_func, s_star, x_star


@pytest.fixture(scope="module")
def built_surface():
    """Build a surface once and reuse across tests (expensive to build).

    Uses euler_M=15, euler_A=23 so that far-tail q values (up to ~8× the
    mean of Q_t) are resolved accurately.  M=12/A=20 leaves g negative at
    those points; M=15/A=23 keeps g positive throughout the tested range.
    """
    params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    num_cfg = NumericalConfig(
        t_min=0.1, t_max=1.0, n_time=8,
        q_min=1e-5, q_max=3.0, n_q=60,
        q_small=1e-3, euler_M=15, euler_A=23.0
    )
    surface = LocalVolSurface(params, num_cfg)
    surface.build(verbose=False)
    return surface


@pytest.fixture
def default_params():
    return HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)


class TestSurfaceBuild:
    def test_surface_built(self, built_surface):
        """Surface is built successfully."""
        assert built_surface._built

    def test_phi_grid_shape(self, built_surface):
        """phi_grid has correct shape."""
        cfg = built_surface.num_cfg
        assert built_surface.phi_grid.shape == (cfg.n_time, cfg.n_q)

    def test_phi_grid_nonneg(self, built_surface):
        """All phi_t(q) values are nonneg."""
        assert np.all(built_surface.phi_grid >= 0), "phi grid should be nonneg"

    def test_t_grid_range(self, built_surface):
        """t_grid spans [t_min, t_max]."""
        cfg = built_surface.num_cfg
        assert abs(built_surface.t_grid[0] - cfg.t_min) < 1e-10
        assert abs(built_surface.t_grid[-1] - cfg.t_max) < 1e-10


class TestLambdaBarrier:
    def test_lambda_zero_at_barrier(self, built_surface, default_params):
        """lambda(t, s*(t)) = 0 exactly."""
        for t in [0.2, 0.5, 1.0]:
            if t > built_surface.num_cfg.t_max:
                continue
            sstar = s_star(t, default_params)
            lv = built_surface.lambda_t_s(t, np.array([sstar]))
            assert abs(lv[0]) < 1e-10, f"lambda at barrier = {lv[0]} for t={t}"

    def test_lambda_zero_above_barrier(self, built_surface, default_params):
        """lambda(t, s) = 0 for s > s*(t)."""
        t = 0.5
        sstar = s_star(t, default_params)
        s_above = np.array([1.1 * sstar, 1.5 * sstar, 2.0 * sstar])
        lv = built_surface.lambda_t_s(t, s_above.copy())
        np.testing.assert_allclose(lv, 0.0, atol=1e-10)

    def test_lambda_positive_below_barrier(self, built_surface, default_params):
        """lambda(t, s) > 0 for s well below s*(t)."""
        t = 0.5
        sstar = s_star(t, default_params)
        s_below = np.linspace(0.1 * sstar, 0.5 * sstar, 5)
        lv = built_surface.lambda_t_s(t, s_below.copy())
        assert np.all(lv > 0), f"lambda should be positive below barrier, got {lv}"

    def test_lambda_at_t_zero(self, built_surface, default_params):
        """lambda(0, s) = v0 for all s."""
        s = np.array([50.0, 100.0, 150.0])
        lv = built_surface.lambda_t_s(0.0, s.copy())
        np.testing.assert_allclose(lv, default_params.v0, rtol=1e-10)


class TestLambdaUpperBound:
    def test_lambda_leq_A_plus(self, built_surface, default_params):
        """lambda(t,s) <= A(t,s)^+ everywhere."""
        for t in built_surface.t_grid[:5]:
            sstar = s_star(t, default_params)
            s_grid = np.linspace(0.05 * sstar, 1.5 * sstar, 50)
            lv = built_surface.lambda_t_s(t, s_grid.copy())
            A = A_func(t, s_grid, default_params)
            A_plus = np.maximum(A, 0.0)
            excess = lv - A_plus
            assert np.all(excess <= 1e-8), (
                f"lambda exceeds A^+ at t={t:.3f}: max excess = {excess.max():.4e}")


class TestMuLogSpace:
    def test_mu_equals_lambda_at_exp_x(self, built_surface, default_params):
        """mu(t, x) = lambda(t, exp(x))."""
        t = 0.5
        x = np.array([4.0, 4.5, 5.0])  # log-spots
        mu = built_surface.mu_t_x(t, x.copy())
        lv = built_surface.lambda_t_s(t, np.exp(x))
        np.testing.assert_allclose(mu, lv, rtol=1e-10)

    def test_mu_zero_above_x_star(self, built_surface, default_params):
        """mu(t, x) = 0 for x >= x*(t)."""
        t = 0.5
        xstar = x_star(t, default_params)
        x_above = np.array([xstar, xstar + 0.1, xstar + 1.0])
        mu = built_surface.mu_t_x(t, x_above.copy())
        np.testing.assert_allclose(mu, 0.0, atol=1e-10)


class TestLambdaBatch:
    def test_lambda_batch_matches_pointwise(self, built_surface, default_params):
        """lambda_batch(t_arr, s_arr) matches pointwise lambda_t_s."""
        t_arr = np.array([0.2, 0.5, 0.5])
        s_arr = np.array([80.0, 100.0, 120.0])

        # Clip t values to t_grid range
        t_min, t_max = built_surface.t_grid[0], built_surface.t_grid[-1]
        t_arr = np.clip(t_arr, t_min, t_max)

        lv_batch = built_surface.lambda_batch(t_arr, s_arr)
        lv_point = np.array([
            built_surface.lambda_t_s(t, np.array([s]))[0]
            for t, s in zip(t_arr, s_arr)
        ])
        np.testing.assert_allclose(lv_batch, lv_point, rtol=1e-8)
