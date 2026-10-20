"""Form preferences and two-value bandits (SPEC sections 4.1, 7 and 8).

Every preference is a categorical dimension with a hierarchical Dirichlet
posterior: population -> user -> user x scope. A yes/no choice (a button
clicked or not, a clarification arm that worked or not) is the two-value case
of the same model.
"""

import random

from app.core import vocab

POPULATION = 0
USER_SCOPE = "*"
STRENGTH = 4.0  # Pseudo-counts each level borrows from the level above.
EXPLICIT = 3.0  # Count added by a button or a format issue.
IMPLICIT = 0.3  # Scale of the counts added by the turn's reaction.
EDIT = 10.0  # Count added by a correction on the profile page.
BINARY = (vocab.PrefValue.YES, vocab.PrefValue.NO)
AUTO_PRIOR, OTHER_PRIOR = 4.0, 0.25

# (level, dimension, value) -> count; level is "scope", "user" or "population".
Counts = dict[tuple[str, str, str], float]


def values_of(dimension: str) -> tuple[str, ...]:
    """Returns the values of a form dimension, or yes/no for a bandit arm."""
    return vocab.DIMENSION_VALUES.get(dimension, BINARY)


def prior(dimension: str) -> dict[str, float]:
    """Returns the population pseudo-counts of a dimension.

    Form dimensions lean on AUTO, so a new user rarely gets an untried form;
    bandit arms start flat.
    """
    if dimension not in vocab.DIMENSION_VALUES:
        return dict.fromkeys(BINARY, 1.0)
    return {
        v: AUTO_PRIOR if v == vocab.PrefValue.AUTO else OTHER_PRIOR
        for v in values_of(dimension)
    }


def arm(kind: vocab.ArmKind, name: str) -> str:
    """Returns the two-value dimension holding a bandit arm's outcomes."""
    return f"{kind}:{name}"


def _mean(
    counts: Counts, level: str, dimension: str, base: dict[str, float]
) -> dict[str, float]:
    raw = {v: counts.get((level, dimension, v), 0.0) + base[v] for v in base}
    total = sum(raw.values())
    return {v: c / total for v, c in raw.items()}


def user_mean(counts: Counts, dimension: str) -> dict[str, float]:
    """Returns the user-level preference distribution of a dimension."""
    population = _mean(counts, "population", dimension, prior(dimension))
    shrunk = {v: STRENGTH * p for v, p in population.items()}
    return _mean(counts, "user", dimension, shrunk)


def posterior(counts: Counts, dimension: str) -> dict[str, float]:
    """Returns the Dirichlet parameters of the scope, shrunk toward the user."""
    return {
        v: counts.get(("scope", dimension, v), 0.0) + STRENGTH * p
        for v, p in user_mean(counts, dimension).items()
    }


def draw(alphas: dict[str, float], rng: random.Random) -> dict[str, float]:
    """Returns one Dirichlet draw (a Beta draw for two values)."""
    gammas = {v: rng.gammavariate(a, 1.0) for v, a in alphas.items()}
    total = sum(gammas.values()) or 1.0
    return {v: g / total for v, g in gammas.items()}


def thompson(counts: Counts, dimension: str, rng: random.Random) -> str:
    """Returns the value with the highest posterior draw."""
    sample = draw(posterior(counts, dimension), rng)
    return max(sample, key=sample.get)


def signals(
    used: dict[str, str],
    explicit: list[tuple[str, str | None]],
    p_moved_on: float,
    p_wrong_format: float,
) -> dict[tuple[str, str], float]:
    """Returns the counts one closed turn adds to each (dimension, value).

    Args:
        used: The value drawn for each form dimension on the turn.
        explicit: (dimension, value) pairs from a button or a format issue;
            a None value spreads over every value but the one used.
        p_moved_on: The turn's posterior probability of moved_on.
        p_wrong_format: The turn's posterior probability of wrong_format.

    Returns:
        The count to add for each (dimension, value).
    """
    out: dict[tuple[str, str], float] = {}

    def add(dimension: str, values: list[str], total: float) -> None:
        if not total:
            return
        for value in values:
            key = (dimension, value)
            out[key] = out.get(key, 0.0) + total / len(values)

    for dimension, value in used.items():
        others = [v for v in values_of(dimension) if v != value]
        add(dimension, [value], IMPLICIT * p_moved_on)
        add(dimension, others, IMPLICIT * p_wrong_format)
    for dimension, value in explicit:
        others = [v for v in values_of(dimension) if v != used.get(dimension)]
        add(dimension, [value] if value else others, EXPLICIT)
    return out
