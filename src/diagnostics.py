"""
Structural validation diagnostics for the local-volatility surface.

Verifies:
1. Nonnegativity: lambda(t,s) >= 0
2. Upper bound: lambda(t,s) <= A(t,s)^+
3. Barrier: lambda(t,s) = 0 for s >= s*(t)
4. Positivity below barrier: lambda(t,s) > 0 for s < s*(t)
5. Near-barrier asymptotics: lambda ~ const * A(t,s)^2 as A -> 0+
"""
import numpy as np
from typing import Tuple, Optional
from .config import HestonNandiParams
from .transforms import A_func, s_star


def check_nonnegativity(lambda_vals: np.ndarray, tol: float = -1e-8) -> dict:
    """
    Check that all lambda values are >= 0 (within tolerance).
    
    Returns dict with keys: passed, n_violations, max_violation
    """
    violations = lambda_vals[lambda_vals < tol]
    return {
        'passed': len(violations) == 0,
        'n_violations': len(violations),
        'max_violation': float(np.min(violations)) if len(violations) > 0 else 0.0
    }


def check_upper_bound(lambda_vals: np.ndarray, A_vals: np.ndarray,
                      tol: float = 1e-8) -> dict:
    """
    Check lambda(t,s) <= A(t,s)^+ = max(A(t,s), 0).
    
    This is implied by lambda = E[v_t | Q_t = q] and v_t <= v_t + beta*I_t^H = Q_t = q = A(t,s).
    Returns dict with keys: passed, n_violations, max_excess
    """
    A_plus = np.maximum(A_vals, 0.0)
    excess = lambda_vals - A_plus - tol
    violations = excess[excess > 0]
    return {
        'passed': len(violations) == 0,
        'n_violations': len(violations),
        'max_excess': float(np.max(violations)) if len(violations) > 0 else 0.0
    }


def check_barrier(
    t: float, s_test: np.ndarray, lambda_vals: np.ndarray,
    params: HestonNandiParams, tol: float = 1e-10
) -> dict:
    """
    Check that lambda(t, s) = 0 for s >= s*(t).
    
    Returns dict: passed, n_violations, max_above_barrier
    """
    sstar = s_star(t, params)
    above = s_test >= sstar
    violations_lv = lambda_vals[above]
    nonzero = violations_lv[np.abs(violations_lv) > tol]
    return {
        'passed': len(nonzero) == 0,
        'n_violations': len(nonzero),
        'max_above_barrier': float(np.max(np.abs(nonzero))) if len(nonzero) > 0 else 0.0,
        's_star': float(sstar),
        'n_above': int(np.sum(above))
    }


def check_positivity_below_barrier(
    t: float, s_test: np.ndarray, lambda_vals: np.ndarray,
    params: HestonNandiParams, tol: float = 1e-8
) -> dict:
    """
    Check that lambda(t,s) > 0 for s sufficiently below barrier.
    
    Only checks points where A(t,s) > 0.01 (away from near-barrier region).
    """
    A = A_func(t, s_test, params)
    below = A > 0.01  # away from barrier
    lv_below = lambda_vals[below]
    nonpos = lv_below[lv_below <= tol]
    return {
        'passed': len(nonpos) == 0,
        'n_violations': len(nonpos),
        'min_lambda_below': float(np.min(lv_below)) if len(lv_below) > 0 else np.nan,
        'n_tested': int(np.sum(below))
    }


def near_barrier_asymptotics(
    t: float, q_near: np.ndarray, lambda_near: np.ndarray,
    params: HestonNandiParams
) -> dict:
    """
    Verify lambda ~ const * q^2 near barrier (Proposition 11 asymptotics).
    
    Expected coefficient: 4*a / (beta * (v0 + a*t)^2).
    
    Returns fit ratio lambda / (coeff * q^2) for q near 0.
    """
    a = params.a
    beta = params.beta
    v0_at = params.v0 + a * t
    expected_coeff = 4.0 * a / (beta * v0_at**2)
    
    # Compute ratio lambda / (expected_coeff * q^2)
    q2 = q_near**2
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(q2 > 1e-20, lambda_near / (expected_coeff * q2), np.nan)
    
    valid = ~np.isnan(ratio)
    return {
        'expected_coeff': float(expected_coeff),
        'mean_ratio': float(np.nanmean(ratio)),
        'std_ratio': float(np.nanstd(ratio)),
        'ratio': ratio,
        'n_valid': int(np.sum(valid))
    }


def validate_surface(
    surface,  # LocalVolSurface
    n_s: int = 100,
    verbose: bool = True
) -> dict:
    """
    Run all structural diagnostics on the local-vol surface.
    
    Tests on dense (t, s) grid.
    Returns dict of diagnostic results per time.
    """
    params = surface.params
    results = {}
    
    for t in surface.t_grid:
        sstar = s_star(t, params)
        # Test grid: from 1% to 200% of s*(t)
        s_grid = np.linspace(0.01 * sstar, 2.0 * sstar, n_s)
        lv = surface.lambda_t_s(t, s_grid.copy())
        A = A_func(t, s_grid, params)
        
        results[t] = {
            'nonnegativity': check_nonnegativity(lv),
            'upper_bound': check_upper_bound(lv, A),
            'barrier': check_barrier(t, s_grid, lv, params),
            'positivity': check_positivity_below_barrier(t, s_grid, lv, params),
        }
        
        # Near-barrier asymptotics (small q region)
        q_near_mask = (A > 0) & (A < 0.05)
        if q_near_mask.sum() > 2:
            lv_near = lv[q_near_mask]
            A_near = A[q_near_mask]
            results[t]['asymptotics'] = near_barrier_asymptotics(t, A_near, lv_near, params)
    
    # Summary
    all_passed = all(
        all(v['passed'] for v in r.values() if isinstance(v, dict) and 'passed' in v)
        for r in results.values()
    )
    if verbose:
        status = 'PASSED' if all_passed else 'FAILED'
        print(f"Surface validation: {status}")
    
    return {'by_time': results, 'all_passed': all_passed}
