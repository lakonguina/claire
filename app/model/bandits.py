"""Buttons (SPEC sections 4.4 and 8): what a click means, Thompson choice.

A button is an intent (shorter, an example, the next question…) whose label
the assistant writes for its reply. Intents are ranked by how often they
helped: clicked, and the reply that followed the click was fine.
"""

import dataclasses
import random

from app.core import vocab
from app.model import prefs

SHUFFLE_PROB = 0.2
SHOWN = 3


@dataclasses.dataclass(frozen=True)
class Effect:
    """What a click on a button tells the models.

    Attributes:
        outcome: The value it reports to the button annotator.
        preference: The (dimension, value) it reinforces, if any.
    """

    outcome: str
    preference: tuple[vocab.Dimension, vocab.PrefValue] | None = None


STYLE_CLICK = "style_click"  # A click that reports no outcome, only a style.
EFFECTS = {
    vocab.Button.SHORTER: Effect(vocab.Outcome.TOO_LONG),
    vocab.Button.MORE_DETAIL: Effect(vocab.Outcome.TOO_SHORT),
    vocab.Button.ONE_LINE: Effect(vocab.Outcome.TOO_LONG),
    vocab.Button.EXAMPLE: Effect(
        STYLE_CLICK, (vocab.Dimension.EXAMPLES, vocab.PrefValue.YES)
    ),
    vocab.Button.SIMPLER: Effect(
        vocab.Outcome.WRONG_FORMAT,
        (vocab.Dimension.LEVEL, vocab.PrefValue.SIMPLE),
    ),
    vocab.Button.AS_LIST: Effect(
        vocab.Outcome.WRONG_FORMAT,
        (vocab.Dimension.FORMAT, vocab.PrefValue.BULLETS),
    ),
    vocab.Button.AS_PROSE: Effect(
        vocab.Outcome.WRONG_FORMAT,
        (vocab.Dimension.FORMAT, vocab.PrefValue.PROSE),
    ),
    vocab.Button.NOT_THIS: Effect(vocab.Outcome.MISUNDERSTOOD),
    # Asking a follow-up question: the reply was enough to move on.
    vocab.Button.RELATED: Effect(vocab.Outcome.MOVED_ON),
}


# The buttons that adjust a reply, chosen per user by Thompson sampling.
@dataclasses.dataclass(frozen=True)
class Shown:
    """A button placed under a reply."""

    button: str
    position: int
    shuffled: bool
    thompson_draw: float | None


def choose(counts: prefs.Counts, rng: random.Random) -> list[Shown]:
    """Returns [most useful, explore, explore], sometimes shuffled.

    Args:
        counts: Preference counts holding the buttons' usefulness, as the
            two-value dimensions "button:<intent>".
        rng: The random source.

    Returns:
        The intents to show, with their positions and Thompson draws.
    """
    alphas = {
        b: prefs.posterior(counts, prefs.arm(vocab.ArmKind.BUTTON, b))
        for b in vocab.Button
    }
    useful = {
        b: a[vocab.PrefValue.YES] / sum(a.values()) for b, a in alphas.items()
    }
    best = max(useful, key=useful.get)
    draws = {
        b: prefs.draw(a, rng)[vocab.PrefValue.YES]
        for b, a in alphas.items()
        if b != best
    }
    chosen = [best, *sorted(draws, key=draws.get, reverse=True)[: SHOWN - 1]]
    shuffled = rng.random() < SHUFFLE_PROB
    if shuffled:
        rng.shuffle(chosen)
    return [Shown(b, i, shuffled, draws.get(b)) for i, b in enumerate(chosen)]
