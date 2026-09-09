"""
Experiment 4: Secondary Consequence Validation.

Tests paper's corollary implications of the convex-order relation:
1. Equal means: E[I_T^H] = E[I_T^LV]
2. Variance ordering: Var(I_T^H) < Var(I_T^LV)
3. Volatility-swap ordering: E[sqrt(I_T^H/T)] > E[sqrt(I_T^LV/T)]
4. Squared IV ordering: E[(I_T^H)^2] < E[(I_T^LV)^2]
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
from scipy import stats as sp_stats

from src.config import HestonNandiParams, NumericalConfig
from src.localvol_surface import LocalVolSurface
from src.heston_nandi import simulate_heston_nandi, terminal_integrated_variance
from src.localvol_sim import simulate_localvol, terminal_integrated_variance_lv
from src.payoffs import volatility_swap_rate

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def bootstrap_ci(x: np.ndarray, stat_func, n_boot: int = 1000,
                  level: float = 0.95, seed: int = 9999) -> tuple:
    """Bootstrap confidence interval for a statistic."""
    rng = np.random.default_rng(seed)
    boot_stats = np.zeros(n_boot)
    for i in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boot_stats[i] = stat_func(sample)
    alpha = 1 - level
    return (float(np.percentile(boot_stats, 100 * alpha / 2)),
            float(np.percentile(boot_stats, 100 * (1 - alpha / 2))))


def run_exp4(
    params: HestonNandiParams = None,
    surface: LocalVolSurface = None,
    maturities: list = None,
    n_paths: int = 30000,
    n_steps_per_year: int = 100,
    n_boot: int = 500,
    seed_heston: int = 5001,
    seed_lv: int = 6001,
    verbose: bool = True
) -> dict:
    """
    Run Experiment 4: Secondary corollary validation.

    Parameters:
        params: model parameters
        surface: pre-built local-vol surface
        maturities: test maturities
        n_paths: MC paths
        n_steps_per_year: time discretization
        n_boot: bootstrap samples
        seed_heston, seed_lv: random seeds
        verbose: print progress

    Returns:
        dict with per-maturity statistics
    """
    if params is None:
        params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    if maturities is None:
        maturities = [0.5, 1.0, 2.0]

    if verbose:
        print("=" * 60)
        print("EXP 4: Secondary Consequences — Means, Variance, Vol-Swap")
        print("=" * 60)
        print(f"Paths: {n_paths}, Steps/year: {n_steps_per_year}")

    # Build surface if needed
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

    all_rows = []

    for T in maturities:
        if verbose:
            print(f"\n--- Maturity T={T:.2f} ---")

        n_steps = max(int(n_steps_per_year * T), 100)

        # Simulate both models
        if verbose:
            print(f"  Simulating Heston-Nandi ({n_paths} paths, {n_steps} steps)...")
        heston_result = simulate_heston_nandi(params, T, n_steps, n_paths, seed=seed_heston)
        I_H = terminal_integrated_variance(heston_result)

        if verbose:
            print(f"  Simulating local-vol ({n_paths} paths, {n_steps} steps)...")
        lv_result = simulate_localvol(surface, params, T, n_steps, n_paths, seed=seed_lv)
        I_LV = terminal_integrated_variance_lv(lv_result)

        # ---- Mean comparison ----
        mean_H = float(np.mean(I_H))
        mean_LV = float(np.mean(I_LV))
        se_mean_H = float(np.std(I_H, ddof=1) / np.sqrt(n_paths))
        se_mean_LV = float(np.std(I_LV, ddof=1) / np.sqrt(n_paths))
        delta_mean = mean_LV - mean_H
        se_delta_mean = np.sqrt(se_mean_H**2 + se_mean_LV**2)
        # Two-sided CI for delta_mean (should contain 0)
        z = sp_stats.norm.ppf(0.975)
        mean_ci = (delta_mean - z * se_delta_mean, delta_mean + z * se_delta_mean)

        # ---- Variance comparison ----
        var_H = float(np.var(I_H, ddof=1))
        var_LV = float(np.var(I_LV, ddof=1))
        delta_var = var_LV - var_H
        # Bootstrap CI for delta_var
        var_ci = bootstrap_ci(
            np.stack([I_H, I_LV], axis=1),
            lambda xy: np.var(xy[:, 1], ddof=1) - np.var(xy[:, 0], ddof=1),
            n_boot=n_boot, seed=42 + int(T * 100)
        ) if False else None  # skip for speed; use t-test approximation
        # Use asymptotic variance approximation
        # Var(var_H) ~ 2*var_H^2/(n-1), so SE(delta_var) ~ sqrt(2*(var_H^2 + var_LV^2)/(n-1))
        se_delta_var = np.sqrt(2 * (var_H**2 + var_LV**2) / (n_paths - 1))
        delta_var_ci = (delta_var - z * se_delta_var, delta_var + z * se_delta_var)

        # ---- Vol-swap comparison ----
        vs_H = volatility_swap_rate(I_H, T)
        vs_LV = volatility_swap_rate(I_LV, T)
        delta_vs = vs_H - vs_LV  # paper: E[sqrt(I_H/T)] > E[sqrt(I_LV/T)] => delta_vs > 0
        # SE via delta method: sqrt(x) approximation
        se_vs_H = float(np.std(np.sqrt(np.maximum(I_H, 0.0) / T), ddof=1) / np.sqrt(n_paths))
        se_vs_LV = float(np.std(np.sqrt(np.maximum(I_LV, 0.0) / T), ddof=1) / np.sqrt(n_paths))
        se_delta_vs = np.sqrt(se_vs_H**2 + se_vs_LV**2)
        vs_ci = (delta_vs - z * se_delta_vs, delta_vs + z * se_delta_vs)

        # ---- Squared IV comparison ----
        sq_H = float(np.mean(I_H**2))
        sq_LV = float(np.mean(I_LV**2))
        delta_sq = sq_LV - sq_H
        se_sq_H = float(np.std(I_H**2, ddof=1) / np.sqrt(n_paths))
        se_sq_LV = float(np.std(I_LV**2, ddof=1) / np.sqrt(n_paths))
        se_delta_sq = np.sqrt(se_sq_H**2 + se_sq_LV**2)
        sq_ci = (delta_sq - z * se_delta_sq, delta_sq + z * se_delta_sq)

        if verbose:
            print(f"  Mean: H={mean_H:.6f}, LV={mean_LV:.6f}, "
                  f"delta={delta_mean:.6f} [{mean_ci[0]:.6f}, {mean_ci[1]:.6f}] "
                  f"(should contain 0: {'YES' if mean_ci[0] < 0 < mean_ci[1] else 'NO'})")
            print(f"  Var:  H={var_H:.8f}, LV={var_LV:.8f}, "
                  f"delta={delta_var:.8f} > 0: {'YES' if delta_var > 0 else 'NO'}")
            print(f"  VolSwap: H={vs_H:.6f}, LV={vs_LV:.6f}, "
                  f"delta={delta_vs:.6f} > 0: {'YES' if delta_vs > 0 else 'NO'}")
            print(f"  E[I^2]: H={sq_H:.8f}, LV={sq_LV:.8f}, "
                  f"delta={delta_sq:.8f} > 0: {'YES' if delta_sq > 0 else 'NO'}")

        row = {
            'T': T, 'n_steps': n_steps, 'n_paths': n_paths,
            # Means
            'mean_H': mean_H, 'mean_LV': mean_LV, 'delta_mean': delta_mean,
            'mean_ci_low': mean_ci[0], 'mean_ci_high': mean_ci[1],
            'mean_equal': bool(mean_ci[0] < 0 < mean_ci[1]),
            # Variances
            'var_H': var_H, 'var_LV': var_LV, 'delta_var': delta_var,
            'var_ci_low': delta_var_ci[0], 'var_ci_high': delta_var_ci[1],
            'var_order_holds': bool(delta_var > 0),
            # Vol-swap
            'vs_H': vs_H, 'vs_LV': vs_LV, 'delta_vs': delta_vs,
            'vs_ci_low': vs_ci[0], 'vs_ci_high': vs_ci[1],
            'vs_order_holds': bool(delta_vs > 0),
            # Squared IV
            'sq_H': sq_H, 'sq_LV': sq_LV, 'delta_sq': delta_sq,
            'sq_ci_low': sq_ci[0], 'sq_ci_high': sq_ci[1],
            'sq_order_holds': bool(delta_sq > 0),
        }
        all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_DIR / "exp4_secondary_results.csv", index=False)

    # Plots
    _plot_secondary_comparison(df, maturities, RESULTS_DIR)
    _plot_iv_distributions(df, params, surface, maturities,
                           n_paths, n_steps_per_year, seed_heston, seed_lv, RESULTS_DIR)

    if verbose:
        print("\n=== EXP 4 SUMMARY ===")
        print(df[['T', 'mean_equal', 'var_order_holds', 'vs_order_holds', 'sq_order_holds']].to_string(index=False))

    return {
        'df': df,
        'mean_equal_all': bool(df['mean_equal'].all()),
        'var_order_all': bool(df['var_order_holds'].all()),
        'vs_order_all': bool(df['vs_order_holds'].all()),
        'sq_order_all': bool(df['sq_order_holds'].all()),
    }


def _plot_secondary_comparison(df, maturities, results_dir):
    """Bar chart of key secondary statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    T_vals = df['T'].values
    x = np.arange(len(T_vals))
    width = 0.35

    # Means
    ax = axes[0]
    ax.bar(x - width/2, df['mean_H'], width, label=r'$E[I_T^H]$', color='blue', alpha=0.7)
    ax.bar(x + width/2, df['mean_LV'], width, label=r'$E[I_T^{LV}]$', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={T:.1f}' for T in T_vals])
    ax.set_ylabel('Integrated Variance Mean')
    ax.set_title(r'Means: $E[I_T^H] \approx E[I_T^{LV}]$')
    ax.legend()

    # Variances
    ax = axes[1]
    ax.bar(x - width/2, df['var_H'], width, label=r'$\mathrm{Var}(I_T^H)$', color='blue', alpha=0.7)
    ax.bar(x + width/2, df['var_LV'], width, label=r'$\mathrm{Var}(I_T^{LV})$', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={T:.1f}' for T in T_vals])
    ax.set_ylabel('Integrated Variance Variance')
    ax.set_title(r'Variances: $\mathrm{Var}(I_T^H) < \mathrm{Var}(I_T^{LV})$')
    ax.legend()

    # Vol-swap
    ax = axes[2]
    ax.bar(x - width/2, df['vs_H'], width, label=r'$E[\sqrt{I_T^H/T}]$', color='blue', alpha=0.7)
    ax.bar(x + width/2, df['vs_LV'], width, label=r'$E[\sqrt{I_T^{LV}/T}]$', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'T={T:.1f}' for T in T_vals])
    ax.set_ylabel('Vol-Swap Rate')
    ax.set_title(r'Vol-Swap: $E[\sqrt{I_T^H/T}] > E[\sqrt{I_T^{LV}/T}]$')
    ax.legend()

    plt.suptitle('Secondary Corollary Validation (Heston-Nandi vs Local-Vol)', fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "exp4_secondary_comparison.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_iv_distributions(df, params, surface, maturities,
                            n_paths, n_steps_per_year, seed_h, seed_lv, results_dir):
    """Plot integrated variance distributions."""
    fig, axes = plt.subplots(1, len(maturities), figsize=(5 * len(maturities), 4))
    if len(maturities) == 1:
        axes = [axes]

    for ax, T in zip(axes, maturities):
        n_steps = max(int(n_steps_per_year * T), 100)
        heston_result = simulate_heston_nandi(params, T, n_steps, n_paths, seed=seed_h)
        I_H = terminal_integrated_variance(heston_result)
        lv_result = simulate_localvol(surface, params, T, n_steps, n_paths, seed=seed_lv)
        I_LV = terminal_integrated_variance_lv(lv_result)

        all_I = np.concatenate([I_H, I_LV])
        bins = np.linspace(max(0, all_I.min() - 0.001), all_I.max() * 1.1, 40)
        ax.hist(I_H, bins=bins, density=True, alpha=0.5, color='blue',
                label=r'$I_T^H$ (Heston)')
        ax.hist(I_LV, bins=bins, density=True, alpha=0.5, color='red',
                label=r'$I_T^{LV}$ (Local-Vol)')
        ax.set_xlabel('Integrated Variance')
        ax.set_ylabel('Density')
        ax.set_title(f'IV Distributions (T={T:.2f})')
        ax.legend(fontsize=8)

    plt.suptitle('Integrated Variance Distributions: Heston vs Local-Vol', fontsize=11)
    plt.tight_layout()
    plt.savefig(results_dir / "exp4_iv_distributions.png", dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_exp4(verbose=True)
    print(f"\nMean equality:       {'ALL PASS' if results['mean_equal_all'] else 'SOME FAIL'}")
    print(f"Variance ordering:   {'ALL PASS' if results['var_order_all'] else 'SOME FAIL'}")
    print(f"Vol-swap ordering:   {'ALL PASS' if results['vs_order_all'] else 'SOME FAIL'}")
    print(f"Squared IV ordering: {'ALL PASS' if results['sq_order_all'] else 'SOME FAIL'}")
