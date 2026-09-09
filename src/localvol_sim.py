"""
Local-volatility model simulation in log-space.

dX̄_t = -0.5 * mu(t, X̄_t) dt + sqrt(mu(t, X̄_t)) dB_t
X̄_0 = log(S0), S̄_t = exp(X̄_t)

Barrier enforcement: mu(t, x) = 0 for x >= x*(t).
Integrated variance: Ī_T^LV = integral_0^T mu(t, X̄_t) dt.

Custom — Context7 found no library equivalent for barrier-aware
local-vol simulation with paper's canonical coefficient.
"""
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .localvol_surface import LocalVolSurface
from .config import HestonNandiParams


def simulate_localvol(
    surface,  # LocalVolSurface
    params: HestonNandiParams,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int = 42
) -> dict:
    """
    Simulate the canonical local-vol model in log-space via Euler-Maruyama.
    
    dX̄_t = -0.5*mu(t,X̄_t)*dt + sqrt(mu(t,X̄_t))*dB_t
    
    Parameters:
        surface: built LocalVolSurface object
        params: Heston-Nandi parameters (for S0, barrier, etc.)
        T: maturity
        n_steps: number of Euler time steps
        n_paths: number of paths
        seed: random seed
    
    Returns:
        dict with keys:
        - 't_grid': time grid
        - 'X': log-spot paths shape (n_paths, n_steps)
        - 'S': spot paths shape (n_paths, n_steps)
        - 'I': integrated variance shape (n_paths, n_steps)
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t_grid = np.linspace(dt, T, n_steps)
    
    X = np.full(n_paths, np.log(params.S0))
    I = np.zeros(n_paths)

    X_paths = np.zeros((n_paths, n_steps))
    I_paths = np.zeros((n_paths, n_steps))

    # Helper: barrier-clamped mu evaluation
    t_min = float(surface.t_grid[0]) if surface.t_grid is not None else dt

    def _mu_clamped(t_val: float, X_val: np.ndarray) -> np.ndarray:
        t_eval = max(t_val, t_min)
        x_star = surface.x_star(t_val if t_val > 0 else t_min)
        mu_val = surface.mu_t_x(t_eval, X_val)
        mu_val = np.maximum(mu_val, 0.0)
        mu_val[X_val >= x_star] = 0.0
        return mu_val

    for j, t in enumerate(t_grid):
        t_curr = t - dt   # start of this Euler step

        # --- coefficient at step start ---
        mu_start = _mu_clamped(t_curr, X)

        # --- Euler-Maruyama step (driven by mu at step start) ---
        dB = rng.standard_normal(n_paths) * np.sqrt(dt)
        X_new = X - 0.5 * mu_start * dt + np.sqrt(mu_start) * dB

        # Hard barrier: clip X below x*(t_end) after stepping
        x_star_end = surface.x_star(t)
        X_new = np.minimum(X_new, x_star_end - 1e-10)

        # --- coefficient at step end (for trapezoidal variance integration) ---
        mu_end = _mu_clamped(t, X_new)

        # Trapezoidal rule: I += 0.5*(mu_start + mu_end)*dt  [O(dt²) bias]
        I = I + 0.5 * (mu_start + mu_end) * dt

        X = X_new
        X_paths[:, j] = X
        I_paths[:, j] = I
    
    # X_paths stores log(S_t), so exp(X_paths) is already S_t directly.
    S_paths = np.exp(X_paths)
    
    return {
        't_grid': t_grid,
        'X': X_paths,
        'S': S_paths,
        'I': I_paths
    }


def terminal_spot_lv(sim_result: dict) -> np.ndarray:
    """Extract terminal spot S̄_T."""
    return sim_result['S'][:, -1]


def terminal_integrated_variance_lv(sim_result: dict) -> np.ndarray:
    """Extract terminal integrated variance Ī_T^LV."""
    return sim_result['I'][:, -1]
