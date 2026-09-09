"""
Payoff functions for integrated variance and spot options.
"""
import numpy as np
from scipy import stats
from typing import Union, Tuple


def variance_call(I: np.ndarray, K: float) -> np.ndarray:
    """
    Realized-variance call payoff: (I_T - K)^+.
    
    Parameters:
        I: array of integrated variances
        K: strike
    
    Returns:
        Payoff array
    """
    return np.maximum(I - K, 0.0)


def spot_call(S: np.ndarray, K: float) -> np.ndarray:
    """
    European call payoff on spot: (S_T - K)^+.
    """
    return np.maximum(S - K, 0.0)


def soft_call(I: np.ndarray, K: float, m: float) -> np.ndarray:
    """
    Soft-call payoff from paper's proof: Phi_{m,K}(i) = (1/m)*log(1 + exp(m*(i-K))).
    
    Converges to (i - K)^+ as m -> infinity.
    Uses numerically stable log1p formulation.
    
    Custom — Context7 found no library equivalent (paper proof mechanism).
    """
    x = m * (I - K)
    # Numerically stable: log(1 + exp(x)) = log1p(exp(x)) for small x,
    # = x + log(1 + exp(-x)) for large x
    return np.log1p(np.exp(np.minimum(x, 500.0))) / m + np.maximum(x - 500.0, 0.0) / m


def variance_call_price(I: np.ndarray, K: float) -> float:
    """Monte Carlo estimate of E[(I_T - K)^+]."""
    return float(np.mean(variance_call(I, K)))


def variance_call_price_se(I: np.ndarray, K: float) -> tuple:
    """Monte Carlo estimate and standard error of E[(I_T - K)^+]."""
    payoffs = variance_call(I, K)
    mean = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(len(payoffs)))
    return mean, se


def spot_call_price(S: np.ndarray, K: float) -> float:
    """Monte Carlo estimate of E[(S_T - K)^+]."""
    return float(np.mean(spot_call(S, K)))


def volatility_swap_rate(I: np.ndarray, T: float) -> float:
    """E[sqrt(I_T / T)] — continuously monitored vol-swap rate."""
    return float(np.mean(np.sqrt(np.maximum(I, 0.0) / T)))


def call_price_grid(I: np.ndarray, K_grid: np.ndarray) -> np.ndarray:
    """Compute MC call prices for all strikes at once."""
    return np.array([variance_call_price(I, K) for K in K_grid])


def call_price_se_grid(I: np.ndarray, K_grid: np.ndarray) -> tuple:
    """Compute MC call prices and standard errors for strike grid."""
    means = np.zeros(len(K_grid))
    ses = np.zeros(len(K_grid))
    for i, K in enumerate(K_grid):
        means[i], ses[i] = variance_call_price_se(I, K)
    return means, ses


# ---------------------------------------------------------------------------
# Public aliases used by experiment scripts and tests
# ---------------------------------------------------------------------------

def call_payoff(I: np.ndarray, K: float) -> np.ndarray:
    """Alias for variance_call: (I_T - K)^+ payoff array."""
    return variance_call(I, K)


def integrated_var_call_price(I: np.ndarray, K: float) -> float:
    """Alias for variance_call_price: MC estimate of E[(I_T - K)^+]."""
    return variance_call_price(I, K)


def vol_swap_expectation(I: np.ndarray, T: float) -> float:
    """Alias for volatility_swap_rate: E[sqrt(I_T / T)]."""
    return volatility_swap_rate(I, T)


def soft_call_payoff(I: np.ndarray, K: float, m: float) -> np.ndarray:
    """Alias for soft_call: Phi_{m,K}(I) = (1/m)*log(1+exp(m*(I-K)))."""
    return soft_call(I, K, m)


def mc_confidence_interval(
    samples: np.ndarray, alpha: float = 0.05
) -> Tuple[float, float, float, float]:
    """
    Compute Monte Carlo confidence interval for the mean.

    Parameters:
        samples: 1-D array of MC draws
        alpha: significance level (e.g. 0.05 → 95 % CI)

    Returns:
        (mean, se, lo, hi) — mean estimate, standard error, lower and upper
        confidence bounds based on the Student-t critical value.
    """
    n = len(samples)
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(n))
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
    lo = mean - t_crit * se
    hi = mean + t_crit * se
    return mean, se, lo, hi
