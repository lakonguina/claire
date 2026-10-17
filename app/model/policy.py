"""Length policies (SPEC 5.4 and 6.7): decide(context) -> decision."""

import dataclasses
import math
import random

from app.model import stats

PROPENSITY_SAMPLES = 200


@dataclasses.dataclass
class Context:
    """What a policy decides from.

    Attributes:
        needed_words: The assistant's estimate of a complete answer.
        theta: The cell's grid posterior.
        b_mean: The assistant bias b(c).
        b_var: The variance of b(c).
        kappa_mean: The conversation's log compliance.
        tau: The spread of the sufficiency threshold.
        q: The quantile of the minimal sufficient length to aim at.
    """

    needed_words: int
    theta: list[float]
    b_mean: float
    b_var: float
    kappa_mean: float
    tau: float
    q: float


@dataclasses.dataclass
class Decision:
    """A length decision and what it was drawn from."""

    target_words: int
    requested_words: int
    theta_sample: float | None = None
    b_sample: float | None = None
    post_mean: float | None = None
    post_sd: float | None = None


def round_words(words: float) -> int:
    """Rounds to the nearest 5, or 10 above 50."""
    step = 10 if words > 50 else 5
    return max(step, int(round(words / step)) * step)


def clamp_target(needed: int, ratio: float) -> float:
    """Returns needed x ratio within [0.5, 1.5] x needed, at least 10."""
    return max(min(needed * ratio, 1.5 * needed), 0.5 * needed, 10)


def requested_from_target(target: float, kappa_mean: float) -> int:
    """Returns the words to ask for, corrected for compliance."""
    return round_words(target / math.exp(kappa_mean))


class BayesPolicy:
    """Thompson sampling at the q-quantile of the minimal sufficient length."""

    version = "bayes-v1"

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def _draw(self, ctx: Context) -> tuple[float, float, float]:
        theta = stats.grid_sample(ctx.theta, self.rng)
        b = (
            self.rng.gauss(ctx.b_mean, math.sqrt(ctx.b_var))
            if ctx.b_var > 0
            else ctx.b_mean
        )
        ratio = math.exp(theta + b + stats.z_quantile(ctx.q) * ctx.tau)
        return theta, b, clamp_target(ctx.needed_words, ratio)

    def decide(self, ctx: Context) -> tuple[Decision, float]:
        """Returns the decision and its propensity."""
        theta, b, target = self._draw(ctx)
        rounded = round_words(target)
        same = sum(
            round_words(self._draw(ctx)[2]) == rounded
            for _ in range(PROPENSITY_SAMPLES)
        )
        mean, var = stats.grid_mean_var(ctx.theta)
        decision = Decision(
            rounded,
            requested_from_target(target, ctx.kappa_mean),
            theta,
            b,
            mean,
            math.sqrt(var),
        )
        return decision, same / PROPENSITY_SAMPLES


class FixedRatioPolicy:
    """Baseline: always a fixed share of the assistant's estimate."""

    version = "fixed-v1"
    ratio = 0.8

    def __init__(self, rng: random.Random | None = None):
        del rng  # Deterministic.

    def decide(self, ctx: Context) -> tuple[Decision, float]:
        """Returns the decision and its propensity (always 1)."""
        target = clamp_target(ctx.needed_words, self.ratio)
        decision = Decision(
            round_words(target), requested_from_target(target, ctx.kappa_mean)
        )
        return decision, 1.0


POLICIES = {p.version: p for p in (BayesPolicy, FixedRatioPolicy)}
