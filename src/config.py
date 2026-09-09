"""Model parameter configurations for Heston-Nandi local-vol experiments."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HestonNandiParams:
    """Heston-Nandi model parameters with rho fixed to -1."""
    S0: float = 100.0
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.3
    rho: float = -1.0  # Fixed by paper assumption

    def __post_init__(self):
        assert self.S0 > 0, "S0 must be positive"
        assert self.v0 > 0, "v0 must be positive"
        assert self.kappa > 0, "kappa must be positive"
        assert self.theta > 0, "theta must be positive"
        assert self.xi > 0, "xi must be positive"
        assert self.rho == -1.0, "rho must be -1 for Heston-Nandi"

    @property
    def beta(self) -> float:
        """beta = kappa + xi/2."""
        return self.kappa + self.xi / 2.0

    @property
    def a(self) -> float:
        """a = kappa * theta."""
        return self.kappa * self.theta

    @property
    def b(self) -> float:
        """b = -kappa."""
        return -self.kappa

    @property
    def c(self) -> float:
        """c = xi."""
        return self.xi

    def as_dict(self) -> dict:
        return {
            'S0': self.S0, 'v0': self.v0, 'kappa': self.kappa,
            'theta': self.theta, 'xi': self.xi, 'rho': self.rho,
            'beta': self.beta, 'a': self.a, 'b': self.b, 'c': self.c
        }


@dataclass
class NumericalConfig:
    """Numerical settings for inverse Laplace and surface construction."""
    # Time grid
    t_min: float = 0.01
    t_max: float = 2.0
    n_time: int = 50

    # q-grid for gap variable Q_t
    q_min: float = 1e-6
    q_max: float = 5.0
    n_q: int = 200

    # Asymptotic threshold for near-barrier approximation
    q_small: float = 1e-3

    # Inverse Laplace (Euler/Abate-Whitt settings)
    euler_M: int = 15       # Euler summation order
    euler_A: float = 23.0  # log(10) * M

    # Tolerance for structural validation
    atol: float = 1e-8
    rtol: float = 1e-6

    # Monte Carlo
    n_paths: int = 50000
    n_steps_per_year: int = 252
    seed: int = 42


DEFAULT_PARAMS = HestonNandiParams(S0=100.0, v0=0.04, kappa=2.0, theta=0.04, xi=0.3)
DEFAULT_NUM = NumericalConfig()
