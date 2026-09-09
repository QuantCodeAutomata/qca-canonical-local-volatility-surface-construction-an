"""
Numerical inverse Laplace transform via Abate-Whitt Euler summation.

Custom implementation — Context7 found no library equivalent matching
the exact branch conventions required by the paper's transforms.

Reference: Abate & Whitt (1995) "Numerical Inversion of Laplace Transforms
of Probability Distributions"
"""
import numpy as np
from typing import Callable
from scipy.special import comb as scipy_comb
from .config import HestonNandiParams
from .transforms import Lt_and_Mt, A_func, s_star


def euler_invert(F: Callable, t: float, M: int = 15, A: float = 23.0) -> float:
    """
    Abate-Whitt Euler summation for inverse Laplace transform.
    
    Approximates f(t) = (1/(2*pi*i)) * integral F(s)*e^{st} ds
    using the Bromwich contour discretization with Euler acceleration.
    
    Parameters:
        F: Laplace transform F(s) as complex-valued callable
        t: evaluation point t > 0
        M: number of terms (accuracy ~ 10^{-M})
        A: contour shift parameter, typically M * ln(10)
    
    Returns:
        Numerical approximation of f(t)
    """
    # Contour: c = A/(2t),  points s_k = (A + 2*pi*k*i)/(2t),  step h = pi/t.
    # Abate-Whitt (1995): f(t) = (e^{A/2}/t) * Euler_avg(S_M,...,S_{2M}).
    half_sum = 0.5 * F(A / (2.0 * t)).real

    # Raw partial sums S_j = Re[F(c)]/2 + sum_{k=1}^{j} (-1)^k Re[F(c+i*pi*k/t)]
    n_terms = 2 * M
    raw = np.zeros(n_terms + 1)
    raw[0] = half_sum
    for k in range(1, n_terms + 1):
        s_k = (A + 2.0j * np.pi * k) / (2.0 * t)
        val = ((-1)**k * F(s_k)).real
        raw[k] = raw[k-1] + val

    # Euler acceleration: weighted average of S_M,...,S_{2M}
    euler_weights = np.array([scipy_comb(M, k, exact=True) for k in range(M + 1)], dtype=float)
    euler_weights /= 2.0**M

    partial_sums = raw[M: 2*M + 1]
    euler_sum = np.dot(euler_weights, partial_sums)

    # Correct factor: e^{ct} / t = e^{(A/(2t))*t} / t = e^{A/2} / t
    return (np.exp(A / 2.0) / t) * euler_sum


def euler_invert_array(F: Callable, t: float, M: int = 15, A: float = 23.0) -> float:
    """
    Vectorized Euler inversion — computes all Bromwich points at once.

    Abate-Whitt formula (Abate & Whitt 1995):
      f(t) ≈ (e^A / (2t)) * Euler_avg(S_M, ..., S_{2M})
    where
      S_j = Re[F(A/(2t))]/2 + sum_{k=1}^{j} (-1)^k Re[F((A+2πki)/(2t))]

    The k=0 term is half-weighted (factor 1/2) because it arises from the
    trapezoidal discretization of the Bromwich contour at the real axis.
    """
    ks = np.arange(0, 2*M + 1, dtype=float)
    s_vals = (A + 2.0j * np.pi * ks) / (2.0 * t)

    F_vals = np.array([F(s) for s in s_vals], dtype=complex).real
    signs = (-1.0)**ks

    # Build terms: k=0 gets half weight (standard Bromwich half-endpoint rule)
    terms = signs * F_vals
    terms[0] *= 0.5  # half-weight for k=0

    # Cumulative partial sums S_j = terms[0] + ... + terms[j]
    partial_sums_raw = np.cumsum(terms)

    # Euler acceleration: weighted average of S_M, S_{M+1}, ..., S_{2M}
    euler_weights = np.array([scipy_comb(M, k, exact=True) for k in range(M + 1)], dtype=float)
    euler_weights /= 2.0**M

    partial_sums = partial_sums_raw[M: 2*M + 1]
    euler_sum = np.dot(euler_weights, partial_sums)

    # Correct factor: e^{ct} / t = e^{(A/(2t))*t} / t = e^{A/2} / t
    return (np.exp(A / 2.0) / t) * euler_sum


def compute_gt_mt(q: np.ndarray, t: float, params: HestonNandiParams,
                  M: int = 15, A: float = 23.0):
    """
    Numerically invert L_t and M_t to get g_t(q) and m_t(q).
    
    g_t(q): density of Q_t = v_t + beta*I_t^H
    m_t(q): v_t-weighted density (m_t(q) = g_t(q) * E[v_t | Q_t = q])
    
    Parameters:
        q: array of q values (q > 0)
        t: time > 0
        params: model parameters
        M, A: Euler inversion settings
    
    Returns:
        (g_t, m_t) arrays of same shape as q
    """
    q = np.asarray(q, dtype=float)
    g = np.zeros_like(q)
    m = np.zeros_like(q)
    
    for i, qi in enumerate(q.flat):
        if qi <= 0:
            continue
        
        def F_g(s):
            L, _ = Lt_and_Mt(np.array([s]), t, params)
            return L[0]
        
        def F_m(s):
            _, M_val = Lt_and_Mt(np.array([s]), t, params)
            return M_val[0]
        
        try:
            g.flat[i] = euler_invert_array(F_g, qi, M=M, A=A)
            m.flat[i] = euler_invert_array(F_m, qi, M=M, A=A)
        except Exception:
            g.flat[i] = np.nan
            m.flat[i] = np.nan
    
    return g, m


def phi_asymptotic(q: np.ndarray, t: float, params: HestonNandiParams) -> np.ndarray:
    """
    Near-barrier asymptotic for phi_t(q) from Proposition 11:
    phi_t(q) ≈ [4a / (beta * (v0 + a*t)^2)] * q^2 as q -> 0.
    
    Custom — Context7 found no library equivalent (paper Proposition 11).
    """
    a = params.a
    beta = params.beta
    v0_at = params.v0 + a * t
    coeff = 4.0 * a / (beta * v0_at**2)
    return coeff * np.asarray(q)**2


def compute_phi(q: np.ndarray, t: float, params: HestonNandiParams,
                q_small: float = 1e-3, M: int = 15, A: float = 23.0) -> np.ndarray:
    """
    Compute phi_t(q) = m_t(q)/g_t(q) = E[v_t | Q_t = q].
    
    For q < q_small, uses near-barrier asymptotic from Proposition 11.
    For q >= q_small, uses numerical inverse Laplace inversion.
    
    Parameters:
        q: array of positive q values
        t: time
        params: model parameters
        q_small: threshold below which asymptotic is used
        M, A: Euler inversion parameters
    
    Returns:
        phi_t(q) array
    """
    q = np.asarray(q, dtype=float)
    phi = np.zeros_like(q)
    
    small_mask = (q > 0) & (q < q_small)
    large_mask = q >= q_small
    
    if small_mask.any():
        phi[small_mask] = phi_asymptotic(q[small_mask], t, params)
    
    if large_mask.any():
        q_large = q[large_mask]
        g, m = compute_gt_mt(q_large, t, params, M=M, A=A)
        # When g is negligibly small the event Q_t = q has near-zero probability
        # (q is far in the tail); set phi = 0 there rather than falling back to
        # the small-q asymptotic which is only valid as q → 0.
        valid = g > 1e-10
        # Use np.divide with out= to avoid divide-by-zero warnings when
        # np.where evaluates both branches (g ≈ 0 in the distribution tail).
        ratio = np.divide(m, g, out=np.zeros_like(g), where=valid)
        phi_large = np.where(valid, ratio, 0.0)
        # Enforce 0 ≤ phi ≤ q  (E[v_t|Q_t=q] ∈ [0,q] since 0 ≤ v_t ≤ Q_t)
        phi_large = np.clip(phi_large, 0.0, q_large)
        phi[large_mask] = phi_large
    
    return phi
