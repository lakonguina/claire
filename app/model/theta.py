"""Length likelihood and the multilevel theta / alpha grids (section 6.5)."""

import math
from dataclasses import dataclass, field

from app.model.reaction import OUTCOMES
from app.model.stats import GRID, grid_mean_var, grid_update, normal_grid, phi

SUFFICIENT = {"moved_on", "too_long"}
INSUFFICIENT = {"too_short", "missing_info"}


@dataclass
class Hyper:
    mu0: float = math.log(0.6)
    sigma_alpha: float = 0.5
    sigma_gamma: float = 0.3
    sigma_eta: float = 0.3
    tau: float = 0.35
    gamma: dict[str, float] = field(default_factory=dict)

    def gamma_of(self, request_type: str) -> float:
        return self.gamma.get(request_type, 0.0)


def length_likelihood(r_hat: float, offset: float, sigma: float, reaction: list[float]) -> list[float]:
    """sum_y P(y) l_y(theta) over GRID, with P(sufficient | theta) = Phi((r_hat - theta - offset) / sigma)."""
    p_suff = sum(p for y, p in zip(OUTCOMES, reaction) if y in SUFFICIENT)
    p_insuff = sum(p for y, p in zip(OUTCOMES, reaction) if y in INSUFFICIENT)
    p_none = 1.0 - p_suff - p_insuff
    out = []
    for theta in GRID:
        s = phi((r_hat - theta - offset) / sigma)
        out.append(p_suff * s + p_insuff * (1 - s) + p_none)
    return out


def alpha_prior(hyper: Hyper) -> list[float]:
    return normal_grid(hyper.mu0, hyper.sigma_alpha**2)


def new_theta_cell(alpha: list[float], request_type: str, hyper: Hyper) -> list[float]:
    """Prior N(E[alpha] + gamma(t), sigma_eta^2 + Var alpha) for a (user, type) cell with no data."""
    mean, var = grid_mean_var(alpha)
    return normal_grid(mean + hyper.gamma_of(request_type), hyper.sigma_eta**2 + var)


def update(theta: list[float], alpha: list[float], request_type: str, r_hat: float, length_var: float,
           b: float, b_var: float, reaction: list[float], hyper: Hyper) -> tuple[list[float], list[float]]:
    """One turn's censored update of the (user, type) theta grid and the user's alpha grid."""
    var = hyper.tau**2 + length_var + b_var
    theta = grid_update(theta, length_likelihood(r_hat, b, math.sqrt(var), reaction))
    alpha_var = var + hyper.sigma_eta**2 + hyper.sigma_gamma**2
    alpha = grid_update(alpha, length_likelihood(r_hat, b + hyper.gamma_of(request_type), math.sqrt(alpha_var), reaction))
    return theta, alpha
