"""The preferred length as a Bayesian regression on the request (SPEC 6.5).

No request categories: a request is described by features learned from the
data, its soft membership in the learned topic domains (section 7.1), plus
whether it comes from a phone. A user's preferred log ratio of length to
needed words is theta(x) = w . x, and w has a Gaussian posterior, starting
from the population prior and updated after every turn by assumed-density
filtering: the censored likelihood is a mixture of probit factors, whose
Gaussian moment matching is exact in closed form. Nearby requests share what
is learned through their shared domains; nothing has a hard frontier.
"""

import dataclasses
import math

import numpy as np

from app import vocab
from app.model import reaction
from app.model import stats

SUFFICIENT = frozenset({vocab.Outcome.MOVED_ON, vocab.Outcome.TOO_LONG})
INSUFFICIENT = frozenset({vocab.Outcome.TOO_SHORT, vocab.Outcome.MISSING_INFO})
BIAS, MOBILE = "bias", "mobile"
# Profile-page length choices: the ratio of needed words they mean.
EDIT_RATIOS = {
    vocab.LengthChoice.SHORT: 0.4,
    vocab.LengthChoice.MEDIUM: 0.7,
    vocab.LengthChoice.DETAILED: 1.0,
}
EDIT_SD = 0.15
MIN_VAR = 1e-6
INFORMED = 0.9  # Posterior / prior variance under which data has spoken.


@dataclasses.dataclass
class Prior:
    """The population prior of the weights, learned nightly, and tau.

    Attributes:
        mean: Feature -> prior mean; absent features have mean 0.
        var: Feature -> prior variance; absent ones get default_var.
        default_var: The prior variance of a domain not yet learned.
        tau: The spread of the sufficiency threshold from turn to turn.
    """

    mean: dict[str, float] = dataclasses.field(
        default_factory=lambda: {BIAS: math.log(0.6)}
    )
    var: dict[str, float] = dataclasses.field(
        default_factory=lambda: {BIAS: 0.5**2, MOBILE: 0.3**2}
    )
    default_var: float = 0.3**2
    tau: float = 0.35


@dataclasses.dataclass
class Belief:
    """A Gaussian posterior over the weights of the named features."""

    names: list[str]
    mean: np.ndarray
    cov: np.ndarray

    @classmethod
    def empty(cls) -> "Belief":
        return cls([], np.zeros(0), np.zeros((0, 0)))

    def vector(self, x: dict[str, float], prior: Prior) -> np.ndarray:
        """Returns x over the belief's features, adding the missing ones.

        A feature seen for the first time (a new domain) joins at its prior,
        independent of the others.
        """
        new = [n for n in x if n not in self.names]
        if new:
            self.names = self.names + new
            self.mean = np.concatenate(
                [self.mean, [prior.mean.get(n, 0.0) for n in new]]
            )
            size = len(self.names)
            cov = np.zeros((size, size))
            cov[: size - len(new), : size - len(new)] = self.cov
            for i, n in enumerate(new, size - len(new)):
                cov[i, i] = prior.var.get(n, prior.default_var)
            self.cov = cov
        return np.array([x.get(n, 0.0) for n in self.names])

    def predict(self, x: dict[str, float], prior: Prior) -> tuple[float, float]:
        """Returns the mean and variance of theta(x)."""
        z = self.vector(x, prior)
        return float(z @ self.mean), max(float(z @ self.cov @ z), MIN_VAR)

    def absorb(self, z: np.ndarray, g: float, h: float) -> None:
        """Moment-matching step for a factor of theta = w . z.

        Args:
            z: The feature vector.
            g: d log Z / d mean of theta.
            h: d2 log Z / d mean2 of theta (negative).
        """
        cz = self.cov @ z
        self.mean = self.mean + cz * g
        self.cov = self.cov + np.outer(cz, cz) * h

    def observe(
        self, x: dict[str, float], value: float, var: float, prior: Prior
    ) -> None:
        """Conditions on theta(x) ~ N(value, var): a profile-page edit."""
        z = self.vector(x, prior)
        mean, v = self.predict(x, prior)
        self.absorb(z, (value - mean) / (v + var), -1 / (v + var))


def informed(belief: Belief, prior: Prior) -> list[str]:
    """Returns the features beyond the bias that data has informed."""
    return [
        n
        for i, n in enumerate(belief.names)
        if n != BIAS
        and belief.cov[i, i] < INFORMED * prior.var.get(n, prior.default_var)
    ]


def features(domains: dict[str, float], mobile: bool) -> dict[str, float]:
    """Returns a request's features: bias, phone, soft domain membership."""
    return {BIAS: 1.0, **({MOBILE: 1.0} if mobile else {}), **domains}


def update(
    belief: Belief,
    x: dict[str, float],
    r_hat: float,
    length_var: float,
    bias: tuple[float, float],
    outcome_probs: list[float],
    prior: Prior,
) -> None:
    """Applies one turn's censored observation to the belief, in place.

    The reply of log ratio r_hat was long enough when it reached the user's
    threshold theta(x) + b, up to a spread of sd s: P(sufficient) =
    Phi((r_hat - theta - b) / s). The likelihood mixes that factor, its
    complement and a constant by the outcome probabilities, and its Gaussian
    projection has closed-form slopes.

    Args:
        belief: The user's belief, updated in place.
        x: The turn's features.
        r_hat: The observed log ratio of length to needed words.
        length_var: The variance of the fused log length.
        bias: The assistant bias b(c) and its variance.
        outcome_probs: P(y) for every outcome of reaction.OUTCOMES.
        prior: The population prior.
    """
    probs = dict(zip(reaction.OUTCOMES, outcome_probs))
    p_suff = sum(probs[y] for y in SUFFICIENT)
    p_insuff = sum(probs[y] for y in INSUFFICIENT)
    b, b_var = bias
    z = belief.vector(x, prior)
    mean, var = belief.predict(x, prior)
    s = math.sqrt(prior.tau**2 + length_var + b_var + var)
    t = (r_hat - b - mean) / s
    cdf, pdf = float(stats.phi(t)), math.exp(-t * t / 2) / math.sqrt(
        2 * math.pi
    )
    total = p_suff * cdf + p_insuff * (1 - cdf) + (1 - p_suff - p_insuff)
    slope = (p_insuff - p_suff) * pdf / s  # dZ / d mean.
    curve = (p_insuff - p_suff) * t * pdf / s**2  # d2Z / d mean2.
    g = slope / total
    belief.absorb(z, g, curve / total - g * g)


def edit_features(scope: str, domain_prefix: str) -> dict[str, float]:
    """Returns the features a profile-page length row stands for.

    Args:
        scope: "*" (in general), "mobile", or a domain scope. Rows of the
            retired request types read as general, or mobile for "t@mobile".
        domain_prefix: The prefix of domain scopes.

    Returns:
        The features.
    """
    if scope.startswith(domain_prefix):
        return {BIAS: 1.0, scope: 1.0}
    return features({}, scope == MOBILE or scope.endswith("@" + MOBILE))


def edit(belief: Belief, x: dict[str, float], choice: str, prior: Prior):
    """Applies a profile-page length correction, a sharp observation."""
    belief.observe(x, math.log(EDIT_RATIOS[choice]), EDIT_SD**2, prior)
