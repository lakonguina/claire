"""Reading behaviour (section 6.6): read fully / skimmed / away mixture."""

import math
from dataclasses import dataclass

from app.model.stats import log_normal_pdf, mean_var, normalize

COMPONENTS = ["read", "skim", "away"]


@dataclass
class ReadingParams:
    s_r: float = 0.5
    mu_away: float = math.log(600.0)
    s_away: float = 1.5


def component_densities(log_t: float, full_read: float, params: ReadingParams) -> list[float]:
    """Densities of log reading time under each component; full_read = L - rho."""
    read = math.exp(log_normal_pdf(log_t, full_read, params.s_r**2))
    skim = 1.0 / full_read if full_read > 0 and 0.0 <= log_t <= full_read else 1e-6
    away = math.exp(log_normal_pdf(log_t, params.mu_away, params.s_away**2))
    return [read, skim, away]


def responsibilities(log_t: float, full_read: float, weights: list[float], params: ReadingParams) -> list[float]:
    dens = component_densities(log_t, full_read, params)
    return normalize([w * d for w, d in zip(weights, dens)])


def update_rho(rho_mean: float, rho_var: float, log_t: float, length: float, length_var: float,
               p_read: float, params: ReadingParams) -> tuple[float, float]:
    """Conjugate update of the user's log reading speed from one turn, weighted by P(read fully)."""
    if p_read < 1e-3:
        return rho_mean, rho_var
    obs_var = (params.s_r**2 + length_var) / p_read
    precision = 1 / rho_var + 1 / obs_var
    mean = (rho_mean / rho_var + (length - log_t) / obs_var) / precision
    return mean, 1 / precision


def fit_mixture(data: list[tuple[float, float]], params: ReadingParams, iterations: int = 50
                ) -> tuple[ReadingParams, list[float]]:
    """Pooled EM over (log_t, full_read) pairs: refits s_r, the away component and the weights."""
    weights = [1 / 3] * 3
    for _ in range(iterations):
        resp = [responsibilities(lt, fr, weights, params) for lt, fr in data]
        weights = normalize([sum(r[k] for r in resp) + 1e-3 for k in range(3)])
        read_w = [r[0] for r in resp]
        away_w = [r[2] for r in resp]
        if sum(read_w) > 1:
            s_r2 = sum(w * (lt - fr) ** 2 for w, (lt, fr) in zip(read_w, data)) / sum(read_w)
            params = ReadingParams(math.sqrt(max(s_r2, 1e-3)), params.mu_away, params.s_away)
        if sum(away_w) > 1:
            mu, var = mean_var([lt for lt, _ in data], away_w)
            params = ReadingParams(params.s_r, mu, math.sqrt(max(var, 1e-2)))
    return params, weights
