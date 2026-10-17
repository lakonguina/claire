"""Small numeric helpers shared by the models: normal distribution, grids, bootstrap."""

import math
import random
from statistics import NormalDist

STD_NORMAL = NormalDist()
GRID_N = 81
GRID_LO, GRID_HI = math.log(0.1), math.log(2.0)
GRID = [GRID_LO + i * (GRID_HI - GRID_LO) / (GRID_N - 1) for i in range(GRID_N)]


def phi(x: float) -> float:
    return STD_NORMAL.cdf(x)


def z_quantile(q: float) -> float:
    return STD_NORMAL.inv_cdf(q)


def log_normal_pdf(x: float, mean: float, var: float) -> float:
    return -0.5 * (math.log(2 * math.pi * var) + (x - mean) ** 2 / var)


def normalize(weights: list[float]) -> list[float]:
    total = sum(weights)
    if total <= 0 or not math.isfinite(total):
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


def normalize_log(log_weights: list[float]) -> list[float]:
    top = max(log_weights)
    return normalize([math.exp(lw - top) for lw in log_weights])


def normal_grid(mean: float, var: float) -> list[float]:
    """Discretised N(mean, var) on GRID."""
    return normalize_log([log_normal_pdf(x, mean, var) for x in GRID])


def grid_update(probs: list[float], likelihood: list[float]) -> list[float]:
    return normalize([p * lik for p, lik in zip(probs, likelihood)])


def grid_mean_var(probs: list[float]) -> tuple[float, float]:
    mean = sum(p * x for p, x in zip(probs, GRID))
    return mean, max(sum(p * (x - mean) ** 2 for p, x in zip(probs, GRID)), 1e-9)


def grid_sample(probs: list[float], rng: random.Random) -> float:
    """Sample from the grid, jittered uniformly within the sampled cell."""
    i = rng.choices(range(GRID_N), weights=probs)[0]
    half = (GRID_HI - GRID_LO) / (GRID_N - 1) / 2
    return GRID[i] + rng.uniform(-half, half)


def mean_var(values: list[float], weights: list[float] | None = None) -> tuple[float, float]:
    weights = weights or [1.0] * len(values)
    total = sum(weights)
    mean = sum(w * v for w, v in zip(weights, values)) / total
    return mean, sum(w * (v - mean) ** 2 for w, v in zip(weights, values)) / total


def bootstrap_ci(clusters: list, statistic, n: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95 % percentile interval of statistic(resampled clusters)."""
    if len(clusters) < 2:
        return None
    rng = random.Random(seed)
    values = sorted(
        v for v in (statistic(rng.choices(clusters, k=len(clusters))) for _ in range(n)) if v is not None
    )
    if not values:
        return None
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]
