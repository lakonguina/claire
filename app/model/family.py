"""Assistant behaviour (SPEC 6.3): an open set of model families.

A family is whatever model the assistants say they are, normalised by
dropping version tokens: "mistral-large-2411" and "mistral-large-2407" are
one family, "gemini-2.5-pro" is "gemini-pro". No list of vendors is known in
advance: a family exists once some assistant reports it, starts from the
average of the families learned so far, and gets its own behaviour (writing
speed, estimation bias, compliance) as its conversations accumulate.
"""

import dataclasses
import math
import re

from app.model import stats

UNKNOWN = "unknown"  # The family of an assistant that does not say.
SEPARATORS = re.compile(r"[\s\-_./:@]+")
VERSION = re.compile(r"\d")  # Version and date tokens all hold a digit.
FIELDS = ("b", "w", "kappa", "speed_mu", "speed_sd")
# Before any family is learned: no bias, ~50 words per second, wide spread.
POPULATION = dict(zip(FIELDS, (0.0, 0.05, 0.0, math.log(50.0), 0.6)))
SELF_REPORT_RELIABILITY = 0.85  # Initial P(true family | reported family).


def family_of(assistant_model: str | None) -> str:
    """Returns the family of a self-reported model name: its version-free words.

    Args:
        assistant_model: The model name the assistant gave, if any.

    Returns:
        The family key, or UNKNOWN.
    """
    if not assistant_model:
        return UNKNOWN
    words = SEPARATORS.split(assistant_model.lower())
    return "-".join(w for w in words if w and not VERSION.search(w)) or UNKNOWN


@dataclasses.dataclass
class Families:
    """What is learned about every family seen so far.

    Attributes:
        params: Family -> behaviour (FIELDS), learned nightly.
        reliability: Reported family -> P(true family), learned nightly.
        host_prior: Host -> P(family), learned nightly.
    """

    params: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )
    reliability: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )
    host_prior: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )

    def known(self, *extra: str) -> list[str]:
        """Returns every family known, plus any given."""
        return sorted({UNKNOWN, *self.params, *extra})

    def of(self, name: str) -> dict[str, float]:
        """Returns a family's behaviour, or the average of the learned ones."""
        if name in self.params:
            return self.params[name]
        if not self.params:
            return POPULATION
        learned = list(self.params.values())
        return {k: sum(p[k] for p in learned) / len(learned) for k in FIELDS}

    def initial_probs(self, host: str, reported: str) -> dict[str, float]:
        """Returns pi(c, f) before any turn: P(f | host) x P(report | f).

        Args:
            host: The MCP client name.
            reported: The family the assistant reported, or UNKNOWN.

        Returns:
            The probability of each known family.
        """
        families = self.known(reported)
        uniform = 1 / len(families)
        host_prior = self.host_prior.get(host, {})
        learned = self.reliability.get(reported, {})
        off = (1 - SELF_REPORT_RELIABILITY) / max(len(families) - 1, 1)

        def report(f: str) -> float:
            if reported == UNKNOWN:
                return 1.0
            if f in learned:
                return learned[f]
            return SELF_REPORT_RELIABILITY if f == reported else off

        weights = [host_prior.get(f, uniform) * report(f) for f in families]
        return dict(zip(families, stats.normalize(weights)))

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
        """Returns the family mixture of one behaviour field."""
        return sum(p * self.of(f)[key] for f, p in probs.items())

    def bias(self, probs: dict[str, float]) -> tuple[float, float]:
        """Returns b(c) and Var b(c) for the family mixture."""
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
