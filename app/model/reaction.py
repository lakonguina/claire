"""User reaction (SPEC section 6.4): noisy annotators and Dawid-Skene."""

import dataclasses

from app.core import vocab
from app.model import bandits
from app.model import stats

OUTCOMES = [o for o in vocab.Outcome if o != vocab.Outcome.NO_SIGNAL]
POOLED = "all"  # The stratum shared by every observation of an annotator.
NO_INFO = "_none"  # An observation with P = 1 under every true outcome.
POSITIONS = bandits.SHOWN
# The button annotator observes one joint token per turn:
# "<position>:<outcome>" for a click, or "no_click".
BUTTON_OBS = ["no_click"] + [
    f"{p}:{o}"
    for p in range(POSITIONS)
    for o in sorted({e.outcome for e in bandits.EFFECTS.values()})
]
ALPHABETS = {
    vocab.Source.ASSISTANT: OUTCOMES,
    vocab.Source.BUTTON: BUTTON_OBS,
    vocab.Source.READING: ["read", "skim"],
}

# Stratum -> true outcome -> observed value -> probability.
Confusion = dict[str, dict[str, dict[str, float]]]


@dataclasses.dataclass
class Observation:
    """One annotator's (possibly soft) observation of a turn.

    Attributes:
        annotator: The annotator name, a key of ALPHABETS.
        dist: Observed value -> probability.
        strata: Stratum -> weight (soft model-family assignment).
    """

    annotator: str
    dist: dict[str, float]
    strata: dict[str, float] = dataclasses.field(
        default_factory=lambda: {POOLED: 1.0}
    )


@dataclasses.dataclass
class Item:
    """The observations of one turn, with its gold label if any."""

    observations: list[Observation]
    gold: str | None = None


def _rows(diagonal: dict[str, float]) -> dict[str, dict[str, float]]:
    rest = len(OUTCOMES) - 1
    return {
        y: {
            o: diagonal[y] if o == y else (1 - diagonal[y]) / rest
            for o in OUTCOMES
        }
        for y in OUTCOMES
    }


def _button_row(outcome: str) -> dict[str, float]:
    no_click = 0.9 if outcome == vocab.Outcome.MOVED_ON else 0.6
    clicks = BUTTON_OBS[1:]
    matching = [o for o in clicks if o.endswith(":" + outcome)]
    row = {"no_click": no_click}
    for o in clicks:
        if not matching:
            row[o] = (1 - no_click) / len(clicks)
        elif o in matching:
            row[o] = (1 - no_click) * 0.95 / len(matching)
        else:
            row[o] = (1 - no_click) * 0.05 / (len(clicks) - len(matching))
    return row


def default_confusions() -> dict[str, Confusion]:
    """Returns the initial confusion matrices of every annotator.

    Each annotator has a POOLED stratum; finer strata (one per model family
    for the assistant) are learned by Dawid-Skene as data arrives.
    """
    assistant = {
        y: 0.7 if y == vocab.Outcome.MOVED_ON else 0.8 for y in OUTCOMES
    }
    skim = {y: 0.5 if y == vocab.Outcome.TOO_LONG else 0.25 for y in OUTCOMES}
    return {
        vocab.Source.ASSISTANT: {POOLED: _rows(assistant)},
        vocab.Source.BUTTON: {POOLED: {y: _button_row(y) for y in OUTCOMES}},
        vocab.Source.READING: {
            POOLED: {y: {"skim": p, "read": 1 - p} for y, p in skim.items()}
        },
    }


def likelihood(obs: Observation, confusion: Confusion) -> list[float]:
    """Returns P(observation | true outcome) for every outcome."""
    out = []
    for y in OUTCOMES:
        total = 0.0
        for stratum, weight in obs.strata.items():
            rows = confusion[stratum if stratum in confusion else POOLED]
            row = rows[y]
            total += weight * sum(
                p * (1.0 if o == NO_INFO else row.get(o, 0.0))
                for o, p in obs.dist.items()
            )
        out.append(total)
    return out


def posterior(
    prior: list[float],
    observations: list[Observation],
    confusions: dict[str, Confusion],
) -> list[float]:
    """Returns P(true outcome | observations)."""
    post = list(prior)
    for obs in observations:
        lik = likelihood(obs, confusions[obs.annotator])
        post = [p * l for p, l in zip(post, lik)]
    return stats.normalize(post)


def dawid_skene(
    items: list[Item],
    confusions: dict[str, Confusion],
    iterations: int = 50,
    smoothing: float = 0.01,
) -> tuple[dict[str, Confusion], list[float], list[list[float]]]:
    """Runs EM over the true outcomes and every confusion matrix.

    Args:
        items: The turns to learn from; gold labels are clamped.
        confusions: The starting confusion matrices.
        iterations: The number of EM iterations.
        smoothing: The pseudo-count added to every confusion cell.

    Returns:
        The learned confusions, the outcome prior and each item's posterior.
    """
    prior = [1.0 / len(OUTCOMES)] * len(OUTCOMES)
    posts: list[list[float]] = []
    for _ in range(iterations):
        posts = [
            [float(y == item.gold) for y in OUTCOMES]
            if item.gold
            else posterior(prior, item.observations, confusions)
            for item in items
        ]
        counts = {
            a: {
                s: {y: dict.fromkeys(ALPHABETS[a], smoothing) for y in OUTCOMES}
                for s in strata
            }
            for a, strata in confusions.items()
        }
        for item, post in zip(items, posts):
            for obs in item.observations:
                for stratum, weight in obs.strata.items():
                    cells = counts[obs.annotator].setdefault(
                        stratum,
                        {
                            y: dict.fromkeys(
                                ALPHABETS[obs.annotator], smoothing
                            )
                            for y in OUTCOMES
                        },
                    )
                    for o, p in obs.dist.items():
                        if o != NO_INFO:
                            for y, py in zip(OUTCOMES, post):
                                cells[y][o] += py * weight * p
        confusions = {
            a: {
                s: {
                    y: dict(zip(row, stats.normalize(list(row.values()))))
                    for y, row in rows.items()
                }
                for s, rows in strata.items()
            }
            for a, strata in counts.items()
        }
        prior = stats.normalize(
            [sum(post[i] for post in posts) + 1.0 for i in range(len(OUTCOMES))]
        )
    return confusions, prior, posts
