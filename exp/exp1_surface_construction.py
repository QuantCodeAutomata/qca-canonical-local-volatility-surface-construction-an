"""
Experiment 1: Canonical Local-Volatility Surface Construction and Structural Validation.

Constructs the Heston-Nandi canonical local-variance surface lambda(t,s) using
the paper's exact gap-density Laplace transform characterization, then validates
structural properties: nonnegativity, upper bound, barrier, positivity, asymptotics.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import HestonNandiParams, NumericalConfig
from src.transforms import A_func, s_star, x_star, Lt_and_Mt
from src.inversion import compute_phi, phi_asymptotic
from src.localvol_surface import LocalVolSurface
from src.diagnostics import validate_surface, near_barrier_asymptotics

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp1(
    params: HestonNandiParams = None,
    num_cfg: NumericalConfig = None,
    verbose: bool = True
) -> dict:
    """
    Run Experiment 1: Surface construction and structural validation.

    Parameters:
        params: Heston-Nandi model parameters (uses defaults if None)
        num_cfg: numerical configuration (uses defaults if None)
        verbose: print progress

    Returns:
        dict with surface object and diagnostic results
    """
    if params is None:
        params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    if num_cfg is None:
        num_cfg = NumericalConfig(
            t_min=0.05, t_max=2.0, n_time=30,
            q_min=1e-5, q_max=4.0, n_q=150,
            q_small=1e-3, euler_M=15, euler_A=23.0
        )

    if verbose:
        print("=" * 60)
        print("EXP 1: Canonical Local-Vol Surface Construction")
        print("=" * 60)
        print(f"Parameters: S0={params.S0}, v0={params.v0}, kappa={params.kappa}, "
              f"theta={params.theta}, xi={params.xi}, rho={params.rho}")
        print(f"Derived: beta={params.beta:.4f}, a={params.a:.4f}")
        print(f"Grid: {num_cfg.n_time} time pts x {num_cfg.n_q} q pts")

    # Step 1: Build the canonical local-vol surface
    if verbose:
        print("\n[1] Building canonical local-vol surface via inverse Laplace transforms...")
    surface = LocalVolSurface(params, num_cfg)
    surface.build(verbose=verbose)

    # Step 2: Spot diagnostics on a few time points
    if verbose:
        print("\n[2] Surface spot checks...")
    test_times = [0.25, 0.5, 1.0, 2.0]
    test_times = [t for t in test_times if num_cfg.t_min <= t <= num_cfg.t_max]
    if not test_times:
        test_times = [surface.t_grid[len(surface.t_grid)//2]]

    spot_check_data = []
    for t in test_times:
        sstar_t = s_star(t, params)
        s_below = np.linspace(0.01 * sstar_t, 0.99 * sstar_t, 20)
        lv = surface.lambda_t_s(t, s_below.copy())
        A = A_func(t, s_below, params)
        spot_check_data.append({
            't': t, 's_star': sstar_t,
            'lambda_mean': float(np.mean(lv)),
            'lambda_max': float(np.max(lv)),
            'A_plus_max': float(np.max(np.maximum(A, 0.0))),
            'n_below_barrier': int(np.sum(s_below < sstar_t)),
            'all_nonneg': bool(np.all(lv >= -1e-8)),
            'all_below_Aplus': bool(np.all(lv <= np.maximum(A, 0.0) + 1e-8))
        })
        if verbose:
            print(f"  t={t:.2f}: s*(t)={sstar_t:.2f}, mean_lambda={np.mean(lv):.6f}, "
                  f"max_lambda={np.max(lv):.6f}, A+_max={np.max(np.maximum(A,0)):.4f}")

    # Step 3: Structural validation
    if verbose:
        print("\n[3] Running structural validation...")
    # Validate on a subset of time points for speed
    val_times_idx = np.linspace(0, len(surface.t_grid) - 1, min(10, len(surface.t_grid)), dtype=int)
    val_times_idx = list(set(val_times_idx))

    # Create reduced surface for validation
    validation_results = {}
    all_nonneg = True
    all_upper = True
    all_barrier = True
    all_pos = True

    for idx in val_times_idx:
        t = surface.t_grid[idx]
        sstar_t = s_star(t, params)
        s_grid = np.linspace(0.005 * sstar_t, 2.0 * sstar_t, 80)
        lv = surface.lambda_t_s(t, s_grid.copy())
        A = A_func(t, s_grid, params)

        # Nonnegativity
        neg_mask = lv < -1e-8
        nonneg_ok = not neg_mask.any()
        # Upper bound
        excess = lv - np.maximum(A, 0.0)
        upper_ok = bool(np.all(excess <= 1e-8))
        # Barrier
        above = s_grid >= sstar_t
        barrier_ok = bool(np.all(np.abs(lv[above]) <= 1e-10))
        # Positivity below barrier
        far_below = A > 0.05
        pos_ok = bool(np.all(lv[far_below] > -1e-8))

        validation_results[t] = {
            'nonneg': nonneg_ok, 'upper_bound': upper_ok,
            'barrier': barrier_ok, 'positivity': pos_ok
        }
        all_nonneg = all_nonneg and nonneg_ok
        all_upper = all_upper and upper_ok
        all_barrier = all_barrier and barrier_ok
        all_pos = all_pos and pos_ok

    if verbose:
        print(f"  Nonnegativity: {'PASS' if all_nonneg else 'FAIL'}")
        print(f"  Upper bound:   {'PASS' if all_upper else 'FAIL'}")
        print(f"  Barrier zero:  {'PASS' if all_barrier else 'FAIL'}")
        print(f"  Positivity:    {'PASS' if all_pos else 'FAIL'}")

    # Step 4: Near-barrier asymptotics check
    if verbose:
        print("\n[4] Near-barrier asymptotics (lambda ~ const * q^2)...")
    t_test = test_times[0]
    q_small_range = np.linspace(1e-4, 0.02, 30)
    phi_computed = compute_phi(q_small_range, t_test, params, q_small=1e-5,
                               M=num_cfg.euler_M, A=num_cfg.euler_A)
    phi_asym = phi_asymptotic(q_small_range, t_test, params)
    ratio = phi_computed / (phi_asym + 1e-15)
    if verbose:
        print(f"  t={t_test:.2f}: asymptotics ratio (computed/Prop11): "
              f"mean={np.nanmean(ratio):.3f}, std={np.nanstd(ratio):.3f}")

    # Step 5: Plots
    _plot_surface(surface, params, test_times, RESULTS_DIR)
    _plot_phi(surface, params, test_times, num_cfg, RESULTS_DIR)
    _plot_asymptotics(q_small_range, phi_computed, phi_asym, t_test, RESULTS_DIR)
    _plot_lambda_vs_strike(surface, params, test_times, RESULTS_DIR)

    # Summary DataFrame
    summary_df = pd.DataFrame(spot_check_data)
    summary_df.to_csv(RESULTS_DIR / "exp1_spot_checks.csv", index=False)

    results = {
        'surface': surface,
        'spot_checks': spot_check_data,
        'validation': validation_results,
        'all_nonneg': all_nonneg,
        'all_upper': all_upper,
        'all_barrier': all_barrier,
        'all_pos': all_pos,
        'asymptotics_ratio_mean': float(np.nanmean(ratio)),
        'asymptotics_ratio_std': float(np.nanstd(ratio)),
    }
    return results


def _plot_surface(surface, params, test_times, results_dir):
    """Plot lambda(t, s) surface slices."""
    fig, axes = plt.subplots(1, len(test_times), figsize=(4 * len(test_times), 4))
    if len(test_times) == 1:
        axes = [axes]

    for ax, t in zip(axes, test_times):
        sstar_t = s_star(t, params)
        s_grid = np.linspace(0.01 * sstar_t, 1.5 * sstar_t, 200)
        lv = surface.lambda_t_s(t, s_grid.copy())
        A = A_func(t, s_grid, params)
        ax.plot(s_grid, lv, 'b-', lw=2, label=r'$\lambda(t,s)$')
        ax.plot(s_grid, np.maximum(A, 0.0), 'r--', lw=1.5, label=r'$A(t,s)^+$ (upper bound)')
        ax.axvline(sstar_t, color='k', ls=':', lw=1.5, label=r'$s^*(t)$')
        ax.set_xlabel('Spot $s$')
        ax.set_ylabel('Local variance')
        ax.set_title(f'$\\lambda(t={t:.2f}, s)$')
        ax.legend(fontsize=8)
        ax.set_xlim([0, 1.5 * sstar_t])
        ax.set_ylim([-0.002, None])

    plt.suptitle('Canonical Local-Variance Surface Slices (Heston-Nandi, ρ=-1)', fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_surface_slices.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_phi(surface, params, test_times, num_cfg, results_dir):
    """Plot phi_t(q) as function of q."""
    fig, ax = plt.subplots(figsize=(7, 5))
    q_grid = np.linspace(1e-4, 3.0, 300)
    for t in test_times:
        phi = compute_phi(q_grid, t, params, q_small=num_cfg.q_small,
                          M=num_cfg.euler_M, A=num_cfg.euler_A)
        ax.plot(q_grid, phi, lw=2, label=f't={t:.2f}')

    ax.set_xlabel('Gap variable $q = A(t,s)$')
    ax.set_ylabel(r'$\phi_t(q) = E[v_t \mid Q_t = q]$')
    ax.set_title(r'Conditional expectation $\phi_t(q)$ (Heston-Nandi, $\rho=-1$)')
    ax.legend()
    ax.set_ylim([-0.001, None])
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_phi_function.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_asymptotics(q_small, phi_computed, phi_asym, t_test, results_dir):
    """Plot near-barrier asymptotics comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(q_small, phi_computed, 'b-', lw=2, label='Numerical $\\phi_t(q)$')
    ax1.plot(q_small, phi_asym, 'r--', lw=2, label='Prop. 11 asymptotic $\\sim C q^2$')
    ax1.set_xlabel('$q$')
    ax1.set_ylabel(r'$\phi_t(q)$')
    ax1.set_title(f'Near-barrier asymptotics (t={t_test:.2f})')
    ax1.legend()

    ratio = phi_computed / (phi_asym + 1e-15)
    ax2.plot(q_small, ratio, 'g-', lw=2)
    ax2.axhline(1.0, color='k', ls='--', lw=1)
    ax2.set_xlabel('$q$')
    ax2.set_ylabel(r'$\phi_t(q) / (C q^2)$')
    ax2.set_title('Asymptotics ratio (should → 1 as q → 0)')
    ax2.set_ylim([0, 3])

    plt.suptitle('Near-Barrier Asymptotic Validation (Proposition 11)', fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_asymptotics.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_lambda_vs_strike(surface, params, test_times, results_dir):
    """Plot lambda as function of log-moneyness."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for t in test_times:
        sstar_t = s_star(t, params)
        s_grid = np.linspace(0.1 * params.S0, sstar_t * 1.1, 300)
        lv = surface.lambda_t_s(t, s_grid.copy())
        lm = np.log(s_grid / params.S0)
        ax.plot(lm, np.sqrt(lv), lw=2, label=f't={t:.2f}')

    ax.axhline(0, color='k', ls='-', lw=0.5)
    ax.set_xlabel('Log-moneyness $x = \\log(s/S_0)$')
    ax.set_ylabel(r'Local volatility $\sqrt{\lambda(t,s)}$')
    ax.set_title(r'Local Volatility Smile (Heston-Nandi, $\rho=-1$)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "exp1_localvol_smile.png", dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_exp1(verbose=True)
    print("\n=== EXP 1 SUMMARY ===")
    print(f"Nonnegativity: {'PASS' if results['all_nonneg'] else 'FAIL'}")
    print(f"Upper bound:   {'PASS' if results['all_upper'] else 'FAIL'}")
    print(f"Barrier zero:  {'PASS' if results['all_barrier'] else 'FAIL'}")
    print(f"Positivity:    {'PASS' if results['all_pos'] else 'FAIL'}")
    print(f"Asymptotics ratio: mean={results['asymptotics_ratio_mean']:.3f}, "
          f"std={results['asymptotics_ratio_std']:.3f}")
