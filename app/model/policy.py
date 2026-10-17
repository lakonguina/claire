"""Length policies (sections 5.4 and 6.7): decide(context) -> (decision, propensity)."""

import math
import random
from dataclasses import dataclass

from app.model.stats import grid_mean_var, grid_sample, z_quantile

PROPENSITY_SAMPLES = 200


@dataclass
class Context:
    needed_words: int
    theta: list[float]  # theta(u, t) grid posterior
    b_mean: float
    b_var: float
    kappa_mean: float
    tau: float
    q: float


@dataclass
class Decision:
    target_words: int
    requested_words: int
    theta_sample: float | None = None
    b_sample: float | None = None
    post_mean: float | None = None
    post_sd: float | None = None


def round_words(words: float) -> int:
    step = 10 if words > 50 else 5
    return max(step, int(round(words / step)) * step)


def clamp_target(needed: int, ratio: float) -> float:
    return max(min(needed * ratio, 1.5 * needed), 0.5 * needed, 10)


def requested_from_target(target: float, kappa_mean: float) -> int:
    return round_words(target / math.exp(kappa_mean))


class BayesPolicy:
    """Thompson sampling on the theta grid, at the q-quantile of the minimal sufficient length."""

    version = "bayes-v1"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def _draw(self, ctx: Context) -> tuple[float, float, float]:
        theta = grid_sample(ctx.theta, self.rng)
        b = self.rng.gauss(ctx.b_mean, math.sqrt(ctx.b_var)) if ctx.b_var > 0 else ctx.b_mean
        ratio = math.exp(theta + b + z_quantile(ctx.q) * ctx.tau)
        return theta, b, clamp_target(ctx.needed_words, ratio)

    def decide(self, ctx: Context) -> tuple[Decision, float]:
        theta, b, target = self._draw(ctx)
        rounded = round_words(target)
        same = sum(round_words(self._draw(ctx)[2]) == rounded for _ in range(PROPENSITY_SAMPLES))
        mean, var = grid_mean_var(ctx.theta)
        decision = Decision(rounded, requested_from_target(target, ctx.kappa_mean), theta, b, mean, math.sqrt(var))
        return decision, same / PROPENSITY_SAMPLES


class FixedRatioPolicy:
    """Baseline: always a fixed share of the assistant's estimate."""

    version = "fixed-v1"
    ratio = 0.8

    def __init__(self, rng: random.Random | None = None):
        pass

    def decide(self, ctx: Context) -> tuple[Decision, float]:
        target = clamp_target(ctx.needed_words, self.ratio)
        return Decision(round_words(target), requested_from_target(target, ctx.kappa_mean)), 1.0


POLICIES = {p.version: p for p in (BayesPolicy, FixedRatioPolicy)}
