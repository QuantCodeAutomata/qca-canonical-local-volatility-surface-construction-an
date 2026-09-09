"""
Laplace transform ingredients for Heston-Nandi gap-density construction.

Implements exact transforms L_t(z) = E[e^{-z Q_t}] and M_t(z) = E[v_t e^{-z Q_t}]
where Q_t = v_t + beta * I_t^H is the gap variable.

Branch conventions:
- d(z) = sqrt(b^2 + 2*beta*c^2*z): principal square root (real part >= 0)
- log in Acal_t: principal logarithm, continued from positive real axis

Paper methodology: Section defining transforms via gap-density characterization.
"""
import numpy as np
from typing import Union
from .config import HestonNandiParams


def compute_d(z: np.ndarray, params: HestonNandiParams) -> np.ndarray:
    """
    Compute d(z) = sqrt(b^2 + 2*beta*c^2*z).
    
    Uses principal square root (Re(d) >= 0).
    For z on Bromwich contour with Re(z) > 0, discriminant has positive real part
    when beta > 0, ensuring stable principal branch.
    """
    disc = params.b**2 + 2.0 * params.beta * params.c**2 * z
    d = np.sqrt(disc.astype(complex))
    # Ensure principal branch: Re(d) >= 0
    d = np.where(d.real < 0, -d, d)
    return d


def compute_r_plus_minus(z: np.ndarray, d: np.ndarray, params: HestonNandiParams):
    """Compute r_+(z) = (b + d)/c^2 and r_-(z) = (b - d)/c^2."""
    c2 = params.c**2
    r_plus = (params.b + d) / c2
    r_minus = (params.b - d) / c2
    return r_plus, r_minus


def compute_h(z: np.ndarray, r_plus: np.ndarray, r_minus: np.ndarray) -> np.ndarray:
    """Compute h(z) = (z - r_+)/(z - r_-)."""
    return (z - r_plus) / (z - r_minus)


def laplace_ingredients(z: np.ndarray, t: float, params: HestonNandiParams) -> dict:
    """
    Compute all Laplace transform ingredients at complex argument z and time t.
    
    Returns dict with keys: d, r_plus, r_minus, h, B_t, Acal_t, Hcal_t, Jcal_t
    
    Formulas from paper:
    - B_t(z) = (r_+ - h*e^{-dt}*r_-) / (1 - h*e^{-dt})
    - Acal_t(z) = r_+*t + (2/c^2)*log((1 - h*e^{-dt})/(1 - h))
    - Hcal_t(z) = e^{-dt} * ((1-h)/(1-h*e^{-dt}))^2
    - Jcal_t(z) = 2*(1-e^{-dt}) / (c^2*(z-r_-)*(1-h*e^{-dt}))
    """
    z = np.asarray(z, dtype=complex)
    d = compute_d(z, params)
    r_plus, r_minus = compute_r_plus_minus(z, d, params)
    h = compute_h(z, r_plus, r_minus)

    exp_dt = np.exp(-d * t)
    h_exp = h * exp_dt
    denom = 1.0 - h_exp          # 1 - h*e^{-dt}
    denom_h = 1.0 - h            # 1 - h

    # B_t(z)
    B_t = (r_plus - h_exp * r_minus) / denom

    # Acal_t(z): log uses principal branch, continued from positive real axis
    log_arg = denom / denom_h
    Acal_t = r_plus * t + (2.0 / params.c**2) * np.log(log_arg)

    # Hcal_t(z)
    Hcal_t = exp_dt * (denom_h / denom)**2

    # Jcal_t(z)
    Jcal_t = 2.0 * (1.0 - exp_dt) / (params.c**2 * (z - r_minus) * denom)

    return {
        'd': d, 'r_plus': r_plus, 'r_minus': r_minus, 'h': h,
        'B_t': B_t, 'Acal_t': Acal_t, 'Hcal_t': Hcal_t, 'Jcal_t': Jcal_t
    }


def Lt(z: np.ndarray, t: float, params: HestonNandiParams) -> np.ndarray:
    """
    L_t(z) = E[e^{-z Q_t}] = exp(-a * Acal_t(z) - v0 * B_t(z)).
    
    Laplace transform of g_t(q), the density of Q_t = v_t + beta*I_t^H.
    """
    ingr = laplace_ingredients(z, t, params)
    return np.exp(-params.a * ingr['Acal_t'] - params.v0 * ingr['B_t'])


def Mt(z: np.ndarray, t: float, params: HestonNandiParams) -> np.ndarray:
    """
    M_t(z) = E[v_t * e^{-z Q_t}] = L_t(z) * (a*Jcal_t(z) + v0*Hcal_t(z)).
    
    Laplace transform of m_t(q), the v_t-weighted density of Q_t.
    """
    ingr = laplace_ingredients(z, t, params)
    L = np.exp(-params.a * ingr['Acal_t'] - params.v0 * ingr['B_t'])
    return L * (params.a * ingr['Jcal_t'] + params.v0 * ingr['Hcal_t'])


def Lt_and_Mt(z: np.ndarray, t: float, params: HestonNandiParams):
    """Compute L_t(z) and M_t(z) simultaneously (avoids duplicate ingredient computation)."""
    ingr = laplace_ingredients(z, t, params)
    L = np.exp(-params.a * ingr['Acal_t'] - params.v0 * ingr['B_t'])
    M = L * (params.a * ingr['Jcal_t'] + params.v0 * ingr['Hcal_t'])
    return L, M


def A_func(t: float, s: np.ndarray, params: HestonNandiParams) -> np.ndarray:
    """
    A(t,s) = v0 + kappa*theta*t - xi*log(s/S0).
    
    This is the gap variable evaluated at spot s and time t.
    Q_t = A(t, S_t) pathwise when rho = -1.
    """
    return params.v0 + params.kappa * params.theta * t - params.xi * np.log(np.asarray(s) / params.S0)


def s_star(t: float, params: HestonNandiParams) -> float:
    """
    Barrier level s*(t) = S0 * exp((v0 + kappa*theta*t)/xi).
    
    For s >= s*(t), A(t,s) <= 0 and lambda(t,s) = 0.
    """
    return params.S0 * np.exp((params.v0 + params.kappa * params.theta * t) / params.xi)


def x_star(t: float, params: HestonNandiParams) -> float:
    """Log-barrier x*(t) = log(s*(t)) = log(S0) + (v0 + kappa*theta*t)/xi."""
    return np.log(params.S0) + (params.v0 + params.kappa * params.theta * t) / params.xi
