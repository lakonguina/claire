"""Buttons (SPEC sections 4.4 and 8): what a click means, Thompson choice."""

import dataclasses
import random

from app import vocab
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
}


@dataclasses.dataclass(frozen=True)
class Shown:
    """A button placed under a reply."""

    button: str
    position: int
    shuffled: bool
    thompson_draw: float | None


def choose(counts: prefs.Counts, rng: random.Random) -> list[Shown]:
    """Returns [safe, explore, explore], or a random order sometimes.

    Args:
        counts: Preference counts holding the buttons' click dimensions.
        rng: The random source.

    Returns:
        The buttons to show, with their positions and Thompson draws.
    """
    alphas = {
        b: prefs.posterior(counts, prefs.arm(vocab.ArmKind.BUTTON, b))
        for b in vocab.Button
    }
    p_click = {
        b: a[vocab.PrefValue.YES] / sum(a.values()) for b, a in alphas.items()
    }
    safe = max(p_click, key=p_click.get)
    draws = {
        b: prefs.draw(a, rng)[vocab.PrefValue.YES]
        for b, a in alphas.items()
        if b != safe
    }
    chosen = [safe, *sorted(draws, key=draws.get, reverse=True)[: SHOWN - 1]]
    shuffled = rng.random() < SHUFFLE_PROB
    if shuffled:
        rng.shuffle(chosen)
    return [Shown(b, i, shuffled, draws.get(b)) for i, b in enumerate(chosen)]
