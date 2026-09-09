"""
Experiment 5: Proof-Mechanism Replication.

Verifies selected proof-level mechanisms:
1. Pathwise identity: v_t + beta*I_t^H + xi*log(S_t/S0) - v0 - kappa*theta*t ≈ 0
2. Soft-call convergence: Phi_{m,K}(i) = (1/m)*log(1+exp(m*(i-K))) -> (i-K)^+
3. Duhamel comparison identity (Monte Carlo estimate)
4. Conditional-covariance structure
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
from src.transforms import A_func, s_star
from src.localvol_surface import LocalVolSurface
from src.heston_nandi import simulate_heston_nandi, terminal_integrated_variance
from src.localvol_sim import simulate_localvol, terminal_integrated_variance_lv
from src.payoffs import soft_call, variance_call_price

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_exp5(
    params: HestonNandiParams = None,
    surface: LocalVolSurface = None,
    T: float = 1.0,
    K_soft: float = None,
    m_values: list = None,
    n_paths: int = 20000,
    n_steps_per_year: int = 100,
    seed_heston: int = 7001,
    seed_lv: int = 8001,
    verbose: bool = True
) -> dict:
    """
    Run Experiment 5: Proof mechanism replication.

    Parameters:
        params: model parameters
        surface: pre-built local-vol surface
        T: maturity for proof tests
        K_soft: strike for soft-call test (uses E[I_T] if None)
        m_values: list of m for soft-call convergence test
        n_paths: MC paths
        n_steps_per_year: time discretization
        seed_heston, seed_lv: random seeds
        verbose: print progress

    Returns:
        dict with mechanism verification results
    """
    if params is None:
        params = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
    if m_values is None:
        m_values = [5, 10, 20, 50, 100]

    if verbose:
        print("=" * 60)
        print("EXP 5: Proof Mechanism Replication")
        print("=" * 60)
        print(f"T={T}, paths={n_paths}, steps/year={n_steps_per_year}")

    # Build surface if needed
    if surface is None:
        if verbose:
            print("Building canonical local-vol surface...")
        num_cfg = NumericalConfig(
            t_min=0.01, t_max=T + 0.1, n_time=20,
            q_min=1e-5, q_max=5.0, n_q=100,
            q_small=1e-3, euler_M=15, euler_A=23.0
        )
        surface = LocalVolSurface(params, num_cfg)
        surface.build(verbose=verbose)

    n_steps = max(int(n_steps_per_year * T), 100)

    # =========================================================================
    # MECHANISM 1: Pathwise Identity Verification
    # =========================================================================
    if verbose:
        print("\n[Mechanism 1] Pathwise Identity Verification...")

    heston_result = simulate_heston_nandi(params, T, n_steps, n_paths, seed=seed_heston)
    v = heston_result['v']
    I_H_all = heston_result['I']
    S_all = heston_result['S']
    t_grid = heston_result['t_grid']
    residuals = heston_result['pathwise_residual']

    # Residual: v_t + beta*I_t^H + xi*log(S_t/S0) - v0 - kappa*theta*t
    residual_rmse = float(np.sqrt(np.mean(residuals**2)))
    residual_max = float(np.max(np.abs(residuals)))
    residual_mean = float(np.mean(residuals))

    if verbose:
        print(f"  Residual RMSE: {residual_rmse:.2e}")
        print(f"  Residual max:  {residual_max:.2e}")
        print(f"  Residual mean: {residual_mean:.2e}")

    # Verify Q_t = A(t, S_t)
    Q_t_from_def = v + params.beta * I_H_all  # shape (n_paths, n_steps)
    A_t_S_t = np.zeros_like(Q_t_from_def)
    for j, t in enumerate(t_grid):
        A_t_S_t[:, j] = A_func(t, S_all[:, j], params)
    Q_A_discrepancy = Q_t_from_def - A_t_S_t
    Q_A_rmse = float(np.sqrt(np.mean(Q_A_discrepancy**2)))

    if verbose:
        print(f"  Q_t vs A(t,S_t) RMSE: {Q_A_rmse:.2e} (should be ~0)")

    pathwise_results = {
        'residual_rmse': residual_rmse,
        'residual_max': residual_max,
        'residual_mean': residual_mean,
        'Q_A_rmse': Q_A_rmse,
        'identity_holds': residual_rmse < 0.01  # tolerance
    }

    # =========================================================================
    # MECHANISM 2: Soft-Call Convergence
    # =========================================================================
    if verbose:
        print("\n[Mechanism 2] Soft-Call Convergence...")

    I_H_T = terminal_integrated_variance(heston_result)
    lv_result = simulate_localvol(surface, params, T, n_steps, n_paths, seed=seed_lv)
    I_LV_T = terminal_integrated_variance_lv(lv_result)

    if K_soft is None:
        K_soft = float(np.mean(I_H_T))

    # Hard call difference (baseline)
    C_H_hard = variance_call_price(I_H_T, K_soft)
    C_LV_hard = variance_call_price(I_LV_T, K_soft)
    delta_C_hard = C_LV_hard - C_H_hard

    soft_call_results = []
    for m in m_values:
        phi_H = soft_call(I_H_T, K_soft, m)
        phi_LV = soft_call(I_LV_T, K_soft, m)
        E_phi_H = float(np.mean(phi_H))
        E_phi_LV = float(np.mean(phi_LV))
        delta_soft = E_phi_LV - E_phi_H

        soft_call_results.append({
            'm': m, 'E_phi_H': E_phi_H, 'E_phi_LV': E_phi_LV,
            'delta_soft': delta_soft,
            'sign_matches_hard': bool(delta_soft > 0) == bool(delta_C_hard > 0),
            'convergence_to_hard': abs(delta_soft - delta_C_hard)
        })

        if verbose:
            print(f"  m={m:4d}: E[Phi_H]={E_phi_H:.6f}, E[Phi_LV]={E_phi_LV:.6f}, "
                  f"delta={delta_soft:.6f}, |gap-hard|={abs(delta_soft - delta_C_hard):.6f}")

    if verbose:
        print(f"  Hard call delta: {delta_C_hard:.6f}")

    soft_df = pd.DataFrame(soft_call_results)

    # =========================================================================
    # MECHANISM 3: Duhamel Comparison Identity (Monte Carlo)
    # =========================================================================
    if verbose:
        print("\n[Mechanism 3] Duhamel Identity (Monte Carlo approximation)...")

    # E[Phi(I_T^H)] - E[Phi(I_T^LV)] = integral_0^T E[(v_t - lambda(t,S_t)) * H(t,S_t,I_t^H)] dt
    # We use the soft-call with large m as approximation of Phi
    m_duhamel = 50
    phi_H_full = soft_call(I_H_T, K_soft, m_duhamel)
    phi_LV_full = soft_call(I_LV_T, K_soft, m_duhamel)
    # Paper Duhamel identity: E[Phi^H] - E[Phi^LV] = integral E[(v_t - lambda) H] dt
    # LHS is negative when E[Phi^LV] > E[Phi^H] (convex ordering)
    lhs_estimate = float(np.mean(phi_H_full) - np.mean(phi_LV_full))

    # Approximate the Duhamel integrand at each time step
    # v_t - lambda(t, S_t): under Heston, lambda(t, S_t) = E[v_t | S_t]
    # For a proxy, use lambda(t, S_t) evaluated on Heston paths
    v_minus_lambda = np.zeros((n_paths, n_steps))
    for j, t in enumerate(t_grid):
        lv_at_heston_S = surface.lambda_t_s(t, S_all[:, j].copy())
        v_minus_lambda[:, j] = v[:, j] - lv_at_heston_S

    # H(t,s,i) ≈ derivative of soft-call w.r.t. i, evaluated at I_t^H
    # d/di Phi_{m,K}(i) = exp(m*(i-K)) / (1 + exp(m*(i-K))) = sigmoid(m*(i-K))
    def H_proxy(I_path: np.ndarray, K: float, m: float) -> np.ndarray:
        """H(t,s,i) proxy: first-order derivative of soft-call (sigma function)."""
        x = m * (I_path - K)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    I_H_paths = heston_result['I']  # (n_paths, n_steps)

    # Duhamel RHS: integral_0^T E[(v_t - lambda(t,S_t)) * H_proxy(t)] dt
    duhamel_integrand = np.zeros(n_steps)
    for j, t in enumerate(t_grid):
        H_t = H_proxy(I_H_paths[:, j], K_soft, m_duhamel)
        duhamel_integrand[j] = float(np.mean(v_minus_lambda[:, j] * H_t))

    # Trapezoidal integration
    dt = t_grid[1] - t_grid[0] if len(t_grid) > 1 else T / n_steps
    rhs_estimate = float(np.trapz(duhamel_integrand, t_grid))

    if verbose:
        print(f"  LHS (E[Phi^H] - E[Phi^LV]): {lhs_estimate:.6f}")
        print(f"  RHS (Duhamel integral):      {rhs_estimate:.6f}")
        print(f"  Ratio LHS/RHS: {lhs_estimate / (rhs_estimate + 1e-10):.4f} (should be ~1)")

    duhamel_results = {
        'lhs': lhs_estimate, 'rhs': rhs_estimate,
        'ratio': lhs_estimate / (rhs_estimate + 1e-10),
        'same_sign': bool(np.sign(lhs_estimate) == np.sign(rhs_estimate))
    }

    # =========================================================================
    # MECHANISM 4: Conditional Covariance Structure
    # =========================================================================
    if verbose:
        print("\n[Mechanism 4] Conditional Covariance Structure...")

    # E[Cov(I_t^H, H(t,S_t,I_t^H) | S_t)] via local binning
    # v_t - lambda(t,S_t) = -beta*(I_t^H - iota(t,S_t)) where iota(t,s) = (A(t,s) - lambda(t,s))/beta
    # Choose time point at T/2
    j_mid = n_steps // 2
    t_mid = t_grid[j_mid]
    S_mid = S_all[:, j_mid]
    I_mid = I_H_paths[:, j_mid]
    v_mid = v[:, j_mid]
    lv_mid = surface.lambda_t_s(t_mid, S_mid.copy())
    H_mid = H_proxy(I_mid, K_soft, m_duhamel)

    # Estimate conditional covariance Cov(I_t, H_t | S_t) via local binning in S_t
    n_bins = 10
    S_quantiles = np.quantile(S_mid, np.linspace(0, 1, n_bins + 1))
    cond_cov_bins = []
    for b in range(n_bins):
        mask = (S_mid >= S_quantiles[b]) & (S_mid < S_quantiles[b + 1])
        if mask.sum() < 10:
            continue
        I_bin = I_mid[mask]
        H_bin = H_mid[mask]
        cov_val = float(np.cov(I_bin, H_bin)[0, 1])
        cond_cov_bins.append({
            'bin': b, 'S_center': float(0.5 * (S_quantiles[b] + S_quantiles[b+1])),
            'n_paths': int(mask.sum()), 'cond_cov': cov_val
        })

    cov_df = pd.DataFrame(cond_cov_bins)
    mean_cond_cov = float(cov_df['cond_cov'].mean()) if len(cov_df) > 0 else np.nan
    # Reformulation: Duhamel = beta * integral_0^T E[Cov(I_t, H_t | S_t)] dt
    # Check sign: Cov should be positive (I_t and H_t are increasing functions of v_t path)

    if verbose:
        print(f"  t={t_mid:.2f}: mean Cov(I_t, H_t | S_t) = {mean_cond_cov:.8f}")
        print(f"  Sign > 0: {'YES' if mean_cond_cov > 0 else 'NO'}")
        print(f"  Beta = {params.beta:.4f}")

    cond_cov_results = {
        'cov_df': cov_df,
        'mean_cond_cov': mean_cond_cov,
        'cov_positive': bool(mean_cond_cov > 0) if not np.isnan(mean_cond_cov) else None
    }

    # =========================================================================
    # Plots
    # =========================================================================
    _plot_pathwise_residual(heston_result, RESULTS_DIR)
    _plot_soft_call_convergence(soft_df, delta_C_hard, K_soft, T, RESULTS_DIR)
    _plot_duhamel_integrand(t_grid, duhamel_integrand, T, RESULTS_DIR)
    _plot_conditional_covariance(cov_df, t_mid, RESULTS_DIR)

    # Save results
    soft_df.to_csv(RESULTS_DIR / "exp5_soft_call.csv", index=False)
    cov_df.to_csv(RESULTS_DIR / "exp5_cond_cov.csv", index=False)

    if verbose:
        print("\n=== EXP 5 SUMMARY ===")
        print(f"Pathwise identity RMSE: {residual_rmse:.2e} "
              f"({'PASS' if pathwise_results['identity_holds'] else 'FAIL'})")
        print(f"Q_t = A(t,S_t) RMSE: {Q_A_rmse:.2e}")
        print(f"Soft-call sign consistent with hard: "
              f"{'ALL' if soft_df['sign_matches_hard'].all() else 'PARTIAL'}")
        print(f"Duhamel LHS/RHS same sign: {'YES' if duhamel_results['same_sign'] else 'NO'}")
        print(f"Cond. covariance positive: {cond_cov_results['cov_positive']}")

    return {
        'pathwise': pathwise_results,
        'soft_call': soft_call_results,
        'duhamel': duhamel_results,
        'cond_covariance': cond_cov_results,
    }


def _plot_pathwise_residual(heston_result, results_dir):
    """Plot distribution of pathwise identity residuals."""
    residuals = heston_result['pathwise_residual'][:, -1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=50, density=True, color='steelblue', alpha=0.7)
    ax.axvline(0, color='r', ls='--', lw=2, label='Zero (exact identity)')
    ax.set_xlabel(r'Residual $v_T + \beta I_T^H + \xi\log(S_T/S_0) - v_0 - \kappa\theta T$')
    ax.set_ylabel('Density')
    ax.set_title('Pathwise Identity Residual Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "exp5_pathwise_residual.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_soft_call_convergence(soft_df, delta_hard, K, T, results_dir):
    """Plot soft-call delta convergence to hard-call delta."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(soft_df['m'], soft_df['delta_soft'], 'b-o', ms=6, lw=2,
                label=r'$E[\Phi_{m,K}(I^{LV})] - E[\Phi_{m,K}(I^H)]$')
    ax.axhline(delta_hard, color='r', ls='--', lw=2,
               label=r'Hard call $\Delta_C$ (limit $m\to\infty$)')
    ax.set_xlabel('Smoothing parameter $m$')
    ax.set_ylabel(r'$\Delta$ Soft-Call Price')
    ax.set_title(f'Soft-Call Convergence to Hard Call (K={K:.4f}, T={T:.2f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "exp5_soft_call_convergence.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_duhamel_integrand(t_grid, integrand, T, results_dir):
    """Plot Duhamel comparison identity integrand."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_grid, integrand, 'g-', lw=2)
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.fill_between(t_grid, 0, integrand, where=integrand > 0,
                    alpha=0.3, color='green', label='Positive (consistent with theorem)')
    ax.fill_between(t_grid, 0, integrand, where=integrand < 0,
                    alpha=0.3, color='red', label='Negative')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel(r'$E[(v_t - \lambda(t,S_t)) \cdot H(t,S_t,I_t^H)]$')
    ax.set_title(f'Duhamel Comparison Integrand (T={T:.2f})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(results_dir / "exp5_duhamel_integrand.png", dpi=120, bbox_inches='tight')
    plt.close()


def _plot_conditional_covariance(cov_df, t_mid, results_dir):
    """Plot conditional covariance by spot bins."""
    if len(cov_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(cov_df['S_center'], cov_df['cond_cov'], width=cov_df['S_center'].diff().mean() * 0.8,
           color='steelblue', alpha=0.7)
    ax.axhline(0, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Spot $S_t$ bin center')
    ax.set_ylabel(r'$\mathrm{Cov}(I_t^H, H_t \mid S_t)$')
    ax.set_title(f'Conditional Covariance by Spot Bin (t={t_mid:.2f})')
    plt.tight_layout()
    plt.savefig(results_dir / "exp5_cond_cov.png", dpi=120, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    results = run_exp5(verbose=True)
