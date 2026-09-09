"""
Canonical local-volatility surface construction for Heston-Nandi (rho=-1).

Implements lambda(t,s) = phi_t(A(t,s)) below the barrier s*(t),
and zero above the barrier.

Custom — Context7 found no library equivalent for paper's exact
gap-density Laplace transform characterization.
"""
import numpy as np
from typing import Optional, Tuple
from scipy.interpolate import RegularGridInterpolator
from .config import HestonNandiParams, NumericalConfig
from .transforms import A_func, s_star, x_star
from .inversion import compute_phi


class LocalVolSurface:
    """
    Canonical local-volatility surface from paper's gap-density construction.
    
    Stores phi_t(q) on a (t, q) grid and provides callable lambda(t, s)
    and mu(t, x) = lambda(t, exp(x)) with exact barrier enforcement.
    """
    
    def __init__(self, params: HestonNandiParams, num_cfg: NumericalConfig):
        self.params = params
        self.num_cfg = num_cfg
        self.t_grid = None
        self.q_grid = None
        self.phi_grid = None  # shape (n_t, n_q)
        self._interpolator = None
        self._built = False
    
    def build(self, verbose: bool = True):
        """
        Build the surface by computing phi_t(q) on the (t, q) grid.
        
        This is the main construction step, implementing the paper's
        canonical lambda via numerical inverse Laplace transforms.
        """
        cfg = self.num_cfg
        params = self.params
        
        self.t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.n_time)
        self.q_grid = np.linspace(cfg.q_min, cfg.q_max, cfg.n_q)
        
        phi_grid = np.zeros((cfg.n_time, cfg.n_q))
        
        for i, t in enumerate(self.t_grid):
            if verbose and i % 10 == 0:
                print(f"  Building surface: t={t:.3f} ({i+1}/{cfg.n_time})")
            phi_grid[i, :] = compute_phi(
                self.q_grid, t, params,
                q_small=cfg.q_small, M=cfg.euler_M, A=cfg.euler_A
            )
        
        # Enforce nonnegativity
        phi_grid = np.maximum(phi_grid, 0.0)
        self.phi_grid = phi_grid
        
        # Build 2D interpolator in (t, q) space
        self._interpolator = RegularGridInterpolator(
            (self.t_grid, self.q_grid), phi_grid,
            method='linear', bounds_error=False, fill_value=0.0
        )
        self._built = True
        return self
    
    def phi_t(self, t: float, q: np.ndarray) -> np.ndarray:
        """
        Evaluate phi_t(q) = E[v_t | Q_t = q] via interpolation.
        
        Returns 0 for q <= 0 (above/at barrier).
        """
        assert self._built, "Call build() first"
        q = np.asarray(q, dtype=float)
        t_arr = np.full_like(q, t)
        pts = np.stack([t_arr.ravel(), q.ravel()], axis=1)
        vals = self._interpolator(pts).reshape(q.shape)
        return np.maximum(vals, 0.0)
    
    def lambda_t_s(self, t: float, s: np.ndarray) -> np.ndarray:
        """
        Evaluate lambda(t, s) = phi_t(A(t, s)) with exact barrier enforcement.
        
        lambda(t, s) = phi_t(A(t,s))  for s < s*(t)
        lambda(t, s) = 0               for s >= s*(t)
        lambda(0, s) = v0              (initial condition)
        """
        assert self._built, "Call build() first"
        s = np.asarray(s, dtype=float)
        params = self.params
        
        if t == 0.0:
            return np.full_like(s, params.v0)
        
        q = A_func(t, s, params)  # q = A(t, s)
        sstar = s_star(t, params)
        
        # Evaluate phi
        lv = self.phi_t(t, q)
        
        # Enforce barrier: lambda = 0 for s >= s*(t), i.e., q <= 0
        barrier_mask = s >= sstar
        lv[barrier_mask] = 0.0
        lv = np.maximum(lv, 0.0)
        return lv
    
    def mu_t_x(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Evaluate mu(t, x) = lambda(t, exp(x)) in log-spot coordinates.
        
        mu(t, x) = 0 for x >= x*(t)
        """
        x = np.asarray(x, dtype=float)
        s = np.exp(x)
        return self.lambda_t_s(t, s)
    
    def lambda_batch(self, t_arr: np.ndarray, s_arr: np.ndarray) -> np.ndarray:
        """
        Batch evaluate lambda(t_i, s_i) for paired arrays.
        Useful for path-wise evaluation during simulation.
        """
        assert self._built, "Call build() first"
        assert len(t_arr) == len(s_arr)
        params = self.params
        
        q = A_func(t_arr, s_arr, params)  # vectorized over (t, s) pairs
        # Need to interpolate at (t_i, q_i) pairs
        pts = np.stack([t_arr.ravel(), q.ravel()], axis=1)
        vals = self._interpolator(pts).reshape(t_arr.shape)
        vals = np.maximum(vals, 0.0)
        
        # Barrier: A(t,s) <= 0 means s >= s*(t)
        barrier_mask = q <= 0
        vals[barrier_mask] = 0.0
        return vals
    
    def A_func(self, t: float, s: np.ndarray) -> np.ndarray:
        """Convenience: compute A(t,s) = v0 + kappa*theta*t - xi*log(s/S0)."""
        return A_func(t, np.asarray(s), self.params)
    
    def s_star(self, t: float) -> float:
        """Convenience: compute barrier s*(t)."""
        return s_star(t, self.params)
    
    def x_star(self, t: float) -> float:
        """Convenience: compute log-barrier x*(t)."""
        return x_star(t, self.params)
