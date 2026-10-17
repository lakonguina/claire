"""Reading behaviour (SPEC 6.6): read fully / skimmed / away mixture."""

import dataclasses
import math

from app.model import stats

COMPONENTS = ["read", "skim", "away"]


@dataclasses.dataclass
class ReadingParams:
    """Population parameters of the reading mixture."""

    s_r: float = 0.5
    mu_away: float = math.log(600.0)
    s_away: float = 1.5


def component_densities(
    log_t: float, full_read: float, params: ReadingParams
) -> list[float]:
    """Returns the density of log reading time under each component.

    Args:
        log_t: The log reading time.
        full_read: The log time of a full read, L - rho.
        params: The mixture parameters.

    Returns:
        The read, skim and away densities.
    """
    read = math.exp(stats.log_normal_pdf(log_t, full_read, params.s_r**2))
    skimmed = full_read > 0 and 0.0 <= log_t <= full_read
    skim = 1.0 / full_read if skimmed else 1e-6
    away = math.exp(
        stats.log_normal_pdf(log_t, params.mu_away, params.s_away**2)
    )
    return [read, skim, away]


def responsibilities(
    log_t: float, full_read: float, weights: list[float], params: ReadingParams
) -> list[float]:
    """Returns P(component | reading time)."""
    dens = component_densities(log_t, full_read, params)
    return stats.normalize([w * d for w, d in zip(weights, dens)])


def update_rho(
    rho: tuple[float, float],
    log_t: float,
    length: tuple[float, float],
    p_read: float,
    params: ReadingParams,
) -> tuple[float, float]:
    """Updates the user's log reading speed, weighted by P(read fully).

    Args:
        rho: The mean and variance of the log reading speed.
        log_t: The log reading time.
        length: The mean and variance of the log reply length.
        p_read: P(read fully) for the turn.
        params: The mixture parameters.

    Returns:
        The updated mean and variance.
    """
    if p_read < 1e-3:
        return rho
    obs_var = (params.s_r**2 + length[1]) / p_read
    precision = 1 / rho[1] + 1 / obs_var
    mean = (rho[0] / rho[1] + (length[0] - log_t) / obs_var) / precision
    return mean, 1 / precision


def fit_mixture(
    data: list[tuple[float, float]],
    params: ReadingParams,
    iterations: int = 50,
) -> tuple[ReadingParams, list[float]]:
    """Refits s_r, the away component and the weights by pooled EM.

    Args:
        data: (log reading time, log full-read time) pairs.
        params: The starting parameters.
        iterations: The number of EM iterations.

    Returns:
        The fitted parameters and component weights.
    """
    weights = [1 / 3] * 3
    for _ in range(iterations):
        resp = [responsibilities(lt, fr, weights, params) for lt, fr in data]
        weights = stats.normalize(
            [sum(r[k] for r in resp) + 1e-3 for k in range(3)]
        )
        read_w = [r[0] for r in resp]
        away_w = [r[2] for r in resp]
        if sum(read_w) > 1:
            s_r2 = stats.mean_var([lt - fr for lt, fr in data], read_w)
            second = s_r2[1] + s_r2[0] ** 2  # E[(lt - fr)^2] around 0.
            params = dataclasses.replace(
                params, s_r=math.sqrt(max(second, 1e-3))
            )
        if sum(away_w) > 1:
            mu, var = stats.mean_var([lt for lt, _ in data], away_w)
            params = dataclasses.replace(
                params, mu_away=mu, s_away=math.sqrt(max(var, 1e-2))
            )
    return params, weights
