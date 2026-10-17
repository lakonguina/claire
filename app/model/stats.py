"""Numeric helpers shared by the models: normal law, grids, bootstrap."""

from collections.abc import Callable, Sequence
import math
import random
import statistics

STD_NORMAL = statistics.NormalDist()
GRID_N = 81
GRID_LO, GRID_HI = math.log(0.1), math.log(2.0)
GRID = [GRID_LO + i * (GRID_HI - GRID_LO) / (GRID_N - 1) for i in range(GRID_N)]


def phi(x: float) -> float:
    """Returns the standard normal CDF at x."""
    return STD_NORMAL.cdf(x)


def z_quantile(q: float) -> float:
    """Returns the standard normal quantile of q."""
    return STD_NORMAL.inv_cdf(q)


def log_normal_pdf(x: float, mean: float, var: float) -> float:
    """Returns the log density of N(mean, var) at x."""
    return -0.5 * (math.log(2 * math.pi * var) + (x - mean) ** 2 / var)


def normalize(weights: list[float]) -> list[float]:
    """Scales weights to sum to 1; uniform if they cannot."""
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def normalize_log(log_weights: list[float]) -> list[float]:
    """Normalizes weights given as logarithms."""
    top = max(log_weights)
    return normalize([math.exp(lw - top) for lw in log_weights])


def normal_grid(mean: float, var: float) -> list[float]:
    """Returns N(mean, var) discretised on GRID."""
    return normalize_log([log_normal_pdf(x, mean, var) for x in GRID])


def grid_update(probs: list[float], likelihood: list[float]) -> list[float]:
    """Returns the grid posterior after one likelihood."""
    return normalize([p * lik for p, lik in zip(probs, likelihood)])


def grid_mean_var(probs: list[float]) -> tuple[float, float]:
    """Returns the mean and variance of a grid distribution."""
    mean = sum(p * x for p, x in zip(probs, GRID))
    var = sum(p * (x - mean) ** 2 for p, x in zip(probs, GRID))
    return mean, max(var, 1e-9)


def grid_sample(probs: list[float], rng: random.Random) -> float:
    """Samples the grid, jittered uniformly within the sampled cell."""
    i = rng.choices(range(GRID_N), weights=probs)[0]
    half = (GRID_HI - GRID_LO) / (GRID_N - 1) / 2
    return GRID[i] + rng.uniform(-half, half)


def mean_var(
    values: Sequence[float], weights: Sequence[float] | None = None
) -> tuple[float, float]:
    """Returns the weighted mean and variance of values."""
    weights = weights or [1.0] * len(values)
    total = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values)) / total
    var = sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / total
    return mean, var


def bootstrap_ci(
    clusters: list,
    statistic: Callable[[list], float | None],
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Returns the 95 % percentile interval of statistic over resamples.

    Args:
        clusters: The units resampled with replacement.
        statistic: Maps a resample to a value, or None when undefined.
        n: The number of resamples.
        seed: The random seed, for reproducible intervals.

    Returns:
        The interval, or None with fewer than two clusters.
    """
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    resamples = (rng.choices(clusters, k=len(clusters)) for _ in range(n))
    values = sorted(v for v in map(statistic, resamples) if v is not None)
    if not values:
        return None
    last = len(values) - 1
    return values[int(0.025 * last)], values[int(0.975 * last)]
