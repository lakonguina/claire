"""Numeric helpers shared by the models: normal law, bootstrap."""

from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

NORMAL = scipy_stats.norm


def phi(x):
    """Returns the standard normal CDF at x (scalar or array)."""
    return NORMAL.cdf(x)


def z_quantile(q: float) -> float:
    """Returns the standard normal quantile of q."""
    return float(NORMAL.ppf(q))


def log_normal_pdf(x, mean, var):
    """Returns the log density of N(mean, var) at x (scalar or array)."""
    return NORMAL.logpdf(x, mean, np.sqrt(var))


def normalize(weights) -> np.ndarray:
    """Scales weights to sum to 1; uniform if they cannot."""
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0 or not np.isfinite(total):
        return np.full(len(weights), 1 / len(weights))
    return weights / total


def normalize_log(log_weights) -> np.ndarray:
    """Normalizes weights given as logarithms."""
    log_weights = np.asarray(log_weights, dtype=float)
    return normalize(np.exp(log_weights - log_weights.max()))


def mean_var(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> tuple[float, float]:
    """Returns the weighted mean and variance of values."""
    mean = float(np.average(values, weights=weights))
    var = float(np.average((np.asarray(values) - mean) ** 2, weights=weights))
    return mean, var


def cluster_bootstrap(
    numerators: np.ndarray,
    denominators: np.ndarray,
    resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Returns the 95 % interval of sum(num) / sum(den) over cluster resamples.

    Args:
        numerators: The numerator total of each cluster.
        denominators: The denominator total of each cluster.
        resamples: The number of bootstrap resamples.
        seed: The random seed, for reproducible intervals.

    Returns:
        The interval, or None with fewer than two clusters.
    """
    if len(numerators) < 2:
        return None
    picks = np.random.default_rng(seed).integers(
        len(numerators), size=(resamples, len(numerators))
    )
    den = denominators[picks].sum(axis=1)
    ratios = numerators[picks].sum(axis=1)[den > 0] / den[den > 0]
    if ratios.size == 0:
        return None
    low, high = np.percentile(ratios, [2.5, 97.5])
    return float(low), float(high)
