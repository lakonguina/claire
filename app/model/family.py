"""Assistant behaviour: latent behaviour classes.

What an assistant says about itself is never trusted, nor even asked. A
"family" is a class of assistants that behave alike, learned nightly from what
the server measures: the writing speed and the compliance with requested
lengths of each conversation. A Gaussian mixture finds the classes, its size
chosen by BIC; their ids stay stable across nights. Online, a conversation's
class is inferred from its host and the speed measured on every turn.
"""

import dataclasses
import math

import numpy as np
from sklearn import mixture

from app.model import matching
from app.model import stats

POPULATION = "population"  # The single class before any is learned.
CLASS_PREFIX = "class-"
FIELDS = ("b", "w", "kappa", "speed_mu", "speed_sd")
# Before any class is learned: no bias, ~50 words per second, wide spread.
PRIOR = dict(zip(FIELDS, (0.0, 0.05, 0.0, math.log(50.0), 0.6)))
MAX_CLASSES = 6
CONVERSATIONS_PER_CLASS = 10  # Evidence needed per mixture component.
MIN_VAR = 1e-2


@dataclasses.dataclass
class Families:
    """What is learned about the behaviour classes.

    Attributes:
        params: Class -> behaviour (FIELDS), learned nightly.
        host_prior: Host -> P(class), learned nightly.
    """

    params: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )
    host_prior: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )

    def known(self) -> list[str]:
        """Returns the classes learned so far, or the population alone."""
        return sorted(self.params) or [POPULATION]

    def of(self, name: str) -> dict[str, float]:
        """Returns a class's behaviour; a vanished class gets the prior."""
        return self.params.get(name, PRIOR)

    def initial_probs(self, host: str) -> dict[str, float]:
        """Returns pi(c, f) before any turn: P(f | host)."""
        classes = self.known()
        prior = self.host_prior.get(host, {})
        weights = [prior.get(f, 1 / len(classes)) for f in classes]
        return dict(zip(classes, stats.normalize(weights)))

    def update_with_speed(
        self, probs: dict[str, float], speed: float
    ) -> dict[str, float]:
        """Multiplies in N(v; mu_v(f), s_v(f)^2) for one turn's log speed."""
        logs = [
            math.log(max(p, 1e-12))
            + stats.log_normal_pdf(
                speed, self.of(f)["speed_mu"], self.of(f)["speed_sd"] ** 2
            )
            for f, p in probs.items()
        ]
        return dict(zip(probs, stats.normalize_log(logs)))

    def mix(self, probs: dict[str, float], key: str) -> float:
        """Returns the class mixture of one behaviour field."""
        return sum(p * self.of(f)[key] for f, p in probs.items())

    def bias(self, probs: dict[str, float]) -> tuple[float, float]:
        """Returns b(c) and Var b(c) for the class mixture."""
        b = self.mix(probs, "b")
        second = sum(
            p * (self.of(f)["b"] ** 2 + self.of(f)["w"])
            for f, p in probs.items()
        )
        return b, max(second - b**2, 0.0)

    def speed_prior(self, probs: dict[str, float]) -> tuple[float, float]:
        """Returns the mixture mean and variance of the log writing speed."""
        mean = self.mix(probs, "speed_mu")
        second = sum(
            p * (self.of(f)["speed_mu"] ** 2 + self.of(f)["speed_sd"] ** 2)
            for f, p in probs.items()
        )
        return mean, max(second - mean**2, 1e-4)


@dataclasses.dataclass
class Fit:
    """Behaviour classes fitted on conversations.

    Attributes:
        names: The stable name of each class.
        means: Per class, the mean (log speed, log compliance).
        variances: Per class, the variances of the same.
        memberships: Per conversation, P(class), one column per class.
    """

    names: list[str]
    means: np.ndarray
    variances: np.ndarray
    memberships: np.ndarray


def fit_classes(features: np.ndarray, previous: Families) -> Fit | None:
    """Fits behaviour classes on (log speed, log compliance) per conversation.

    Args:
        features: One row per conversation.
        previous: The classes learned before, to keep their names.

    Returns:
        The fit, or None without enough conversations for one class.
    """
    largest = min(MAX_CLASSES, len(features) // CONVERSATIONS_PER_CLASS)
    if largest < 1:
        return None
    best = min(
        (
            mixture.GaussianMixture(
                k, covariance_type="diag", random_state=0
            ).fit(features)
            for k in range(1, largest + 1)
        ),
        key=lambda model: model.bic(features),
    )
    old = [n for n in previous.params if n.startswith(CLASS_PREFIX)]
    old_means = np.array(
        [
            [previous.params[n]["speed_mu"], previous.params[n]["kappa"]]
            for n in old
        ]
    ).reshape(len(old), 2)
    distance = np.linalg.norm(
        best.means_[:, None, :] - old_means[None, :, :], axis=-1
    )
    ids = matching.stable_ids(
        -distance, [int(n.removeprefix(CLASS_PREFIX)) for n in old]
    )
    return Fit(
        names=[f"{CLASS_PREFIX}{i}" for i in ids],
        means=best.means_,
        variances=np.maximum(best.covariances_, MIN_VAR),
        memberships=best.predict_proba(features),
    )
