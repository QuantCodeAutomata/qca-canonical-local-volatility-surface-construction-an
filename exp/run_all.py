"""
Main runner: executes all 5 experiments and writes results/RESULTS.md.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import HestonNandiParams, NumericalConfig
from src.localvol_surface import LocalVolSurface

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_all(
    params: HestonNandiParams = None,
    n_paths: int = 50000,
    n_steps_per_year: int = 150,
    maturities: list = None,
    verbose: bool = True,
) -> dict:
    """
    Run all five experiments in sequence, sharing the surface object.
    """
    if params is None:
        params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    if maturities is None:
        maturities = [0.5, 1.0, 2.0]

    T_max = max(maturities)
    all_results = {}

    print("\n" + "=" * 70)
    print("HESTON-NANDI CANONICAL LOCAL-VOL — FULL EXPERIMENT SUITE")
    print("=" * 70)
    print(f"Params: S0={params.S0}, v0={params.v0}, kappa={params.kappa}, "
          f"theta={params.theta}, xi={params.xi}, rho={params.rho}")
    print(f"Paths: {n_paths}, Steps/year: {n_steps_per_year}, Maturities: {maturities}")

    # -------------------------------------------------------------------------
    # EXP 1: Surface construction and structural validation
    # -------------------------------------------------------------------------
    from exp.exp1_surface_construction import run_exp1
    t0 = time.time()
    num_cfg_shared = NumericalConfig(
        t_min=0.01, t_max=T_max + 0.05, n_time=50,
        q_min=1e-5, q_max=5.0, n_q=200,
        q_small=1e-3, euler_M=15, euler_A=23.0
    )
    num_cfg_exp1 = NumericalConfig(
        t_min=0.05, t_max=T_max + 0.05, n_time=30,
        q_min=1e-5, q_max=4.0, n_q=150,
        q_small=1e-3, euler_M=15, euler_A=23.0
    )
    r1 = run_exp1(params=params, num_cfg=num_cfg_exp1, verbose=verbose)
    all_results['exp1'] = {
        'nonneg': r1['all_nonneg'], 'upper_bound': r1['all_upper'],
        'barrier': r1['all_barrier'], 'positivity': r1['all_pos'],
        'asym_ratio_mean': r1['asymptotics_ratio_mean'],
        'asym_ratio_std': r1['asymptotics_ratio_std'],
        'elapsed_s': time.time() - t0
    }
    print(f"\nEXP 1 done in {all_results['exp1']['elapsed_s']:.1f}s")

    # Build shared surface for subsequent experiments
    print("\nBuilding shared canonical local-vol surface for Exp 2-5...")
    surface = LocalVolSurface(params, num_cfg_shared)
    surface.build(verbose=verbose)

    # -------------------------------------------------------------------------
    # EXP 2: Mimicking validation
    # -------------------------------------------------------------------------
    from exp.exp2_mimicking_validation import run_exp2
    t0 = time.time()
    r2 = run_exp2(
        params=params, surface=surface, maturities=maturities,
        n_paths=n_paths, n_steps_per_year=n_steps_per_year,
        seed_heston=1001, seed_lv=2001, verbose=verbose
    )
    all_results['exp2'] = {
        'max_call_err': float(r2['summary']['max_abs_call_err'].max()),
        'wasserstein_max': float(r2['summary']['wasserstein_dist'].max()),
        'barrier_violations_total': int(r2['summary']['barrier_violations'].sum()),
        'elapsed_s': time.time() - t0
    }
    print(f"\nEXP 2 done in {all_results['exp2']['elapsed_s']:.1f}s")

    # -------------------------------------------------------------------------
    # EXP 3: Convex-order inequality
    # -------------------------------------------------------------------------
    from exp.exp3_convex_order import run_exp3
    t0 = time.time()
    r3 = run_exp3(
        params=params, surface=surface, maturities=maturities,
        n_paths=n_paths, n_steps_per_year=n_steps_per_year,
        seed_heston=3001, seed_lv=4001, verbose=verbose
    )
    all_results['exp3'] = {
        'n_pass': r3['n_pass'], 'n_total': r3['n_total'],
        'n_sig_fail': r3.get('n_sig_fail', 0),
        'min_delta_C': r3['min_delta_C'],
        'min_delta_C_ci_low': r3['min_delta_C_ci_low'],
        'theorem_supported': r3['theorem_supported'],
        'elapsed_s': time.time() - t0
    }
    print(f"\nEXP 3 done in {all_results['exp3']['elapsed_s']:.1f}s")

    # -------------------------------------------------------------------------
    # EXP 4: Secondary consequences
    # -------------------------------------------------------------------------
    from exp.exp4_secondary_consequences import run_exp4
    t0 = time.time()
    r4 = run_exp4(
        params=params, surface=surface, maturities=maturities,
        n_paths=n_paths, n_steps_per_year=n_steps_per_year,
        seed_heston=5001, seed_lv=6001, verbose=verbose
    )
    all_results['exp4'] = {
        'mean_equal_all': r4['mean_equal_all'],
        'var_order_all': r4['var_order_all'],
        'vs_order_all': r4['vs_order_all'],
        'sq_order_all': r4['sq_order_all'],
        'elapsed_s': time.time() - t0
    }
    print(f"\nEXP 4 done in {all_results['exp4']['elapsed_s']:.1f}s")

    # -------------------------------------------------------------------------
    # EXP 5: Proof mechanisms
    # -------------------------------------------------------------------------
    from exp.exp5_proof_mechanisms import run_exp5
    t0 = time.time()
    r5 = run_exp5(
        params=params, surface=surface, T=maturities[-1],
        n_paths=n_paths, n_steps_per_year=n_steps_per_year,
        seed_heston=7001, seed_lv=8001, verbose=verbose
    )
    all_results['exp5'] = {
        'pathwise_rmse': r5['pathwise']['residual_rmse'],
        'Q_A_rmse': r5['pathwise']['Q_A_rmse'],
        'identity_holds': r5['pathwise']['identity_holds'],
        'duhamel_same_sign': r5['duhamel']['same_sign'],
        'duhamel_ratio': r5['duhamel']['ratio'],
        'cov_positive': r5['cond_covariance']['cov_positive'],
        'elapsed_s': time.time() - t0
    }
    print(f"\nEXP 5 done in {all_results['exp5']['elapsed_s']:.1f}s")

    # -------------------------------------------------------------------------
    # Write RESULTS.md
    # -------------------------------------------------------------------------
    _write_results_md(all_results, params, maturities, n_paths, n_steps_per_year,
                      r2['summary'], r3['df'], r4['df'])

    # Save JSON summary
    json_path = RESULTS_DIR / "experiment_summary.json"
    with open(json_path, 'w') as f:
        # Convert non-serializable values
        safe_results = {}
        for k, v in all_results.items():
            safe_results[k] = {kk: (bool(vv) if isinstance(vv, (bool, np.bool_)) else
                                    float(vv) if isinstance(vv, (np.floating, float)) else
                                    int(vv) if isinstance(vv, (np.integer, int)) else
                                    vv)
                               for kk, vv in v.items()}
        json.dump(safe_results, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    return all_results


def _write_results_md(all_results, params, maturities, n_paths, n_steps_per_year,
                      exp2_df, exp3_df, exp4_df):
    """Write the RESULTS.md summary file."""
    r1 = all_results['exp1']
    r2 = all_results['exp2']
    r3 = all_results['exp3']
    r4 = all_results['exp4']
    r5 = all_results['exp5']

    md_lines = [
        "# Experiment Results: Canonical Local-Volatility Surface Construction",
        "",
        "## Model Parameters",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| S0 | {params.S0} |",
        f"| v0 | {params.v0} |",
        f"| kappa | {params.kappa} |",
        f"| theta | {params.theta} |",
        f"| xi | {params.xi} |",
        f"| rho | {params.rho} (fixed) |",
        f"| beta = kappa + xi/2 | {params.beta:.4f} |",
        f"| a = kappa*theta | {params.a:.4f} |",
        "",
        f"**MC Settings**: {n_paths} paths, {n_steps_per_year} steps/year",
        f"**Maturities tested**: {maturities}",
        "",
        "---",
        "",
        "## Experiment 1: Canonical Local-Vol Surface Construction",
        "",
        "Constructed lambda(t,s) via inverse Laplace transforms of L_t(z) and M_t(z).",
        "",
        "| Structural Property | Result |",
        "|---------------------|--------|",
        f"| Nonnegativity (lambda >= 0) | {'✅ PASS' if r1['nonneg'] else '❌ FAIL'} |",
        f"| Upper bound (lambda <= A^+) | {'✅ PASS' if r1['upper_bound'] else '❌ FAIL'} |",
        f"| Barrier zero (s >= s*(t)) | {'✅ PASS' if r1['barrier'] else '❌ FAIL'} |",
        f"| Positivity below barrier | {'✅ PASS' if r1['positivity'] else '❌ FAIL'} |",
        f"| Near-barrier asymptotic ratio (mean) | {r1['asym_ratio_mean']:.3f} (±{r1['asym_ratio_std']:.3f}) |",
        "",
        "**Elapsed**: {:.1f}s".format(r1['elapsed_s']),
        "",
        "---",
        "",
        "## Experiment 2: Mimicking Validation (Spot Law Equality)",
        "",
        "Verified Law(S_T^H) = Law(S_T^LV) via European call price comparison.",
        "",
    ]

    # Exp2 table
    md_lines.append("| T | Max |C_H - C_LV| | Wasserstein dist | Barrier violations |")
    md_lines.append("|---|---------------|-----------------|-------------------|")
    for _, row in exp2_df.iterrows():
        md_lines.append(
            f"| {row['T']:.2f} | {row['max_abs_call_err']:.5f} | "
            f"{row['wasserstein_dist']:.4f} | {int(row['barrier_violations'])} |"
        )
    md_lines += [
        "",
        "**Elapsed**: {:.1f}s".format(r2['elapsed_s']),
        "",
        "---",
        "",
        "## Experiment 3: Primary Theorem — Convex-Order Inequality",
        "",
        "Tests E[(I_T^LV - K)^+] > E[(I_T^H - K)^+] for all K > 0.",
        "",
        f"**Strikes with Delta_C > 0**: {r3['n_pass']}/{r3['n_total']}",
        f"**Statistically significant failures** (Delta_C < 0 and p < 0.05): {r3['n_sig_fail']}",
        f"**Min Delta_C**: {r3['min_delta_C']:.6f}",
        f"**Min Delta_C CI lower bound**: {r3['min_delta_C_ci_low']:.6f}",
        "*(Near K=0 the call difference equals the mean difference ~0; noise-level negatives with p > 0.5 are not counter-evidence.)*",
        f"**Theorem numerically supported** (no stat. sig. failures): {'✅ YES' if r3['theorem_supported'] else '❌ NO'}",
        "",
        "**Elapsed**: {:.1f}s".format(r3['elapsed_s']),
        "",
        "---",
        "",
        "## Experiment 4: Secondary Consequences",
        "",
        "| T | E[I^H]≈E[I^LV] | Var^LV>Var^H | E[√I^H/T]>E[√I^LV/T] | E[(I^LV)^2]>E[(I^H)^2] |",
        "|---|----------------|--------------|----------------------|------------------------|",
    ]

    for _, row in exp4_df.iterrows():
        md_lines.append(
            f"| {row['T']:.2f} | "
            f"{'✅' if row['mean_equal'] else '❌'} | "
            f"{'✅' if row['var_order_holds'] else '❌'} | "
            f"{'✅' if row['vs_order_holds'] else '❌'} | "
            f"{'✅' if row['sq_order_holds'] else '❌'} |"
        )
    md_lines += [
        "",
        "**Elapsed**: {:.1f}s".format(r4['elapsed_s']),
        "",
        "---",
        "",
        "## Experiment 5: Proof Mechanism Replication",
        "",
        "| Mechanism | Result |",
        "|-----------|--------|",
        f"| Pathwise identity residual RMSE | {r5['pathwise_rmse']:.2e} |",
        f"| Q_t = A(t,S_t) RMSE | {r5['Q_A_rmse']:.2e} |",
        f"| Identity holds (RMSE < 0.01) | {'✅ YES' if r5['identity_holds'] else '❌ NO'} |",
        f"| Duhamel LHS/RHS same sign | {'✅ YES' if r5['duhamel_same_sign'] else '❌ NO'} |",
        f"| Duhamel ratio (LHS/RHS) | {r5['duhamel_ratio']:.3f} |",
        f"| Conditional covariance > 0 | {'✅ YES' if r5['cov_positive'] else '❌ NO'} |",
        "",
        "**Elapsed**: {:.1f}s".format(r5['elapsed_s']),
        "",
        "---",
        "",
        "## Figures Generated",
        "",
        "- `exp1_surface_slices.png`: lambda(t,s) slices with barrier and upper bound",
        "- `exp1_phi_function.png`: phi_t(q) conditional expectation function",
        "- `exp1_asymptotics.png`: Near-barrier asymptotics comparison (Proposition 11)",
        "- `exp1_localvol_smile.png`: Local volatility smile sqrt(lambda(t,s))",
        "- `exp2_call_comparison_T*.png`: Spot call price comparison (Heston vs LV)",
        "- `exp2_terminal_dist_T*.png`: Terminal spot distribution comparison",
        "- `exp3_delta_C.png`: Convex-order gap Delta_C = C^LV - C^H",
        "- `exp3_call_prices.png`: Integrated variance call prices",
        "- `exp4_secondary_comparison.png`: Bar chart of secondary statistics",
        "- `exp4_iv_distributions.png`: Integrated variance distributions",
        "- `exp5_pathwise_residual.png`: Pathwise identity residual distribution",
        "- `exp5_soft_call_convergence.png`: Soft-call convergence to hard call",
        "- `exp5_duhamel_integrand.png`: Duhamel comparison identity integrand",
        "- `exp5_cond_cov.png`: Conditional covariance by spot bins",
        "",
        "---",
        "Generated by `exp/run_all.py`",
    ]

    md_path = RESULTS_DIR / "RESULTS.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"RESULTS.md written to {md_path}")


if __name__ == "__main__":
    results = run_all(
        n_paths=20000,
        n_steps_per_year=80,
        maturities=[0.5, 1.0, 2.0],
        verbose=True
    )
