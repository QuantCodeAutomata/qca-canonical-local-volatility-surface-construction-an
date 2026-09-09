"""
Experiment 2: Mimicking Validation — Equality of One-Time Spot Marginals.

Tests that the canonical local-vol model reproduces the one-time spot marginals
of the rho=-1 Heston-Nandi model via European call price comparison.
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
from scipy.stats import wasserstein_distance

from src.config import HestonNandiParams, NumericalConfig
from src.transforms import s_star
from src.localvol_surface import LocalVolSurface
from src.heston_nandi import simulate_heston_nandi, terminal_spot, terminal_integrated_variance
from src.localvol_sim import simulate_localvol, terminal_spot_lv
from src.payoffs import spot_call_price

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp2(
    params: HestonNandiParams = None,
    surface: LocalVolSurface = None,
    maturities: list = None,
    n_paths: int = 30000,
    n_steps_per_year: int = 100,
    seed_heston: int = 1001,
    seed_lv: int = 2001,
    verbose: bool = True
) -> dict:
    """
    Run Experiment 2: Mimicking validation via spot call price comparison.

    Parameters:
        params: model parameters
        surface: pre-built canonical local-vol surface (built if None)
        maturities: list of maturities to test
        n_paths: number of MC paths
        n_steps_per_year: time discretization
        seed_heston: random seed for Heston simulation
        seed_lv: random seed for local-vol simulation
        verbose: print progress

    Returns:
        dict with per-maturity call price comparison results
    """
    if params is None:
        params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    if maturities is None:
        maturities = [0.5, 1.0, 2.0]

    if verbose:
        print("=" * 60)
        print("EXP 2: Mimicking Validation (Spot Law Equality)")
        print("=" * 60)
        print(f"Paths: {n_paths}, Steps/year: {n_steps_per_year}")

    # Build surface if not provided
    if surface is None:
        if verbose:
            print("Building canonical local-vol surface...")
        num_cfg = NumericalConfig(
            t_min=0.01, t_max=max(maturities) + 0.1, n_time=25,
            q_min=1e-5, q_max=5.0, n_q=120,
            q_small=1e-3, euler_M=15, euler_A=23.0
        )
        surface = LocalVolSurface(params, num_cfg)
        surface.build(verbose=verbose)

    all_results = []

    for T in maturities:
        if verbose:
            print(f"\n--- Maturity T={T:.2f} ---")

        n_steps = max(int(n_steps_per_year * T), 100)

        # Simulate Heston-Nandi
        if verbose:
            print(f"  Simulating Heston-Nandi ({n_paths} paths, {n_steps} steps)...")
        heston_result = simulate_heston_nandi(params, T, n_steps, n_paths, seed=seed_heston)
        S_H = terminal_spot(heston_result)

        # Verify pathwise identity residual
        residuals = heston_result['pathwise_residual'][:, -1]
        residual_rmse = float(np.sqrt(np.mean(residuals**2)))
        if verbose:
            print(f"  Pathwise identity residual RMSE: {residual_rmse:.2e}")

        # Verify barrier: S_t < s*(t) almost surely
        t_grid = heston_result['t_grid']
        S_all = heston_result['S']
        barrier_violations = 0
        for j, t in enumerate(t_grid):
            sstar = s_star(t, params)
            barrier_violations += int(np.sum(S_all[:, j] >= sstar))
        if verbose:
            print(f"  Barrier violations (S >= s*(t)): {barrier_violations} / {n_paths * n_steps}")

        # Simulate local-vol model
        if verbose:
            print(f"  Simulating local-vol ({n_paths} paths, {n_steps} steps)...")
        lv_result = simulate_localvol(surface, params, T, n_steps, n_paths, seed=seed_lv)
        S_LV = terminal_spot_lv(lv_result)

        # Strike grid: deep ITM to deep OTM
        sstar_T = s_star(T, params)
        K_grid = np.concatenate([
            np.linspace(0.5 * params.S0, params.S0, 10),
            np.linspace(params.S0, min(2.0 * params.S0, 0.95 * sstar_T), 15)
        ])
        K_grid = np.unique(K_grid)

        # Compute call prices
        C_H = np.array([spot_call_price(S_H, K) for K in K_grid])
        C_LV = np.array([spot_call_price(S_LV, K) for K in K_grid])
        abs_err = np.abs(C_H - C_LV)
        rel_err = abs_err / (C_H + 1e-6)

        # Distribution comparison
        w_dist = wasserstein_distance(S_H, S_LV)

        if verbose:
            print(f"  Call price max abs error: {abs_err.max():.4f}")
            print(f"  Call price RMS error:     {np.sqrt(np.mean(abs_err**2)):.4f}")
            print(f"  Wasserstein distance:     {w_dist:.4f}")
            print(f"  Heston E[S_T]:  {np.mean(S_H):.4f}  (S0={params.S0})")
            print(f"  LocalVol E[S_T]:{np.mean(S_LV):.4f}")

        row = {
            'T': T, 'n_steps': n_steps, 'n_paths': n_paths,
            'max_abs_call_err': float(abs_err.max()),
            'rms_call_err': float(np.sqrt(np.mean(abs_err**2))),
            'max_rel_call_err': float(rel_err.max()),
            'wasserstein_dist': float(w_dist),
            'E_S_H': float(np.mean(S_H)),
            'E_S_LV': float(np.mean(S_LV)),
            'std_S_H': float(np.std(S_H)),
            'std_S_LV': float(np.std(S_LV)),
            'pathwise_residual_rmse': residual_rmse,
            'barrier_violations': barrier_violations,
            'K_grid': K_grid.tolist(),
            'C_H': C_H.tolist(),
            'C_LV': C_LV.tolist(),
        }
        all_results.append(row)

        # Plot call price comparison
        _plot_call_comparison(T, K_grid, C_H, C_LV, RESULTS_DIR)
        _plot_terminal_distributions(T, S_H, S_LV, params, RESULTS_DIR)

    # Save summary table
    summary_cols = ['T', 'n_steps', 'n_paths', 'max_abs_call_err', 'rms_call_err',
                    'wasserstein_dist', 'E_S_H', 'E_S_LV', 'pathwise_residual_rmse',
                    'barrier_violations']
    summary_df = pd.DataFrame([{k: r[k] for k in summary_cols} for r in all_results])
    summary_df.to_csv(RESULTS_DIR / "exp2_mimicking_summary.csv", index=False)

    if verbose:
        print("\n=== EXP 2 SUMMARY ===")
        print(summary_df.to_string(index=False))

    return {'results': all_results, 'summary': summary_df}


def _plot_call_comparison(T, K_grid, C_H, C_LV, results_dir):
    """Plot Heston vs local-vol call prices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(K_grid, C_H, 'b-o', ms=4, lw=2, label='Heston-Nandi')
    ax1.plot(K_grid, C_LV, 'r--s', ms=4, lw=2, label='Local-Vol')
    ax1.set_xlabel('Strike $K$')
    ax1.set_ylabel('Call price $E[(S_T - K)^+]$')
    ax1.set_title(f'Spot Call Prices (T={T:.2f})')
    ax1.legend()

    ax2.plot(K_grid, np.abs(C_H - C_LV), 'g-o', ms=4, lw=2)
    ax2.axhline(0, color='k', ls='--')
    ax2.set_xlabel('Strike $K$')
    ax2.set_ylabel('|C_H - C_LV|')
    ax2.set_title(f'Absolute Call Price Error (T={T:.2f})')

    plt.suptitle(f'Mimicking Validation: Spot Law Equality (T={T:.2f})', fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / f"exp2_call_comparison_T{T:.2f}.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_terminal_distributions(T, S_H, S_LV, params, results_dir):
    """Plot terminal distribution comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(S_H.min(), S_LV.min()), max(S_H.max(), S_LV.max()), 50)
    ax.hist(S_H, bins=bins, density=True, alpha=0.5, color='blue', label='Heston-Nandi')
    ax.hist(S_LV, bins=bins, density=True, alpha=0.5, color='red', label='Local-Vol')
    ax.set_xlabel(f'Terminal spot $S_T$ (T={T:.2f})')
    ax.set_ylabel('Density')
    ax.set_title(f'Terminal Spot Distributions (T={T:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / f"exp2_terminal_dist_T{T:.2f}.png", dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_exp2(verbose=True)
