"""Beta-Bernoulli bandits for buttons (section 8) and style modifiers (section 7)."""

import random
from dataclasses import dataclass

BUTTONS = {
    "shorter": ("Shorter", "Shorter."),
    "more_detail": ("More detail", "More detail."),
    "example": ("Example", "Give an example."),
    "simpler": ("Simpler", "Simpler, please."),
    "one_line": ("One-line summary", "Summarize in one line."),
    "rephrase_request": (
        "Rephrase my request",
        "Rewrite my last request as a clearer prompt, and tell me in one line what was missing.",
    ),
}
STYLES = {"simple": "Use simple words.", "example": "Include one short example."}
STYLE_BUTTON = {"simple": "simpler", "example": "example"}
SHUFFLE_PROB = 0.2
SHOWN = 3


@dataclass
class Shown:
    button: str
    position: int
    shuffled: bool
    thompson_draw: float | None


def beta_mean(counts: tuple[float, float]) -> float:
    return counts[0] / (counts[0] + counts[1])


def choose_buttons(counts: dict[str, tuple[float, float]], exclude: set[str], rng: random.Random) -> list[Shown]:
    """[safe, explore, explore], or a random order with probability SHUFFLE_PROB."""
    candidates = [b for b in BUTTONS if b not in exclude]
    get = lambda b: counts.get(b, (1.0, 1.0))
    safe = max(candidates, key=lambda b: beta_mean(get(b)))
    draws = {b: rng.betavariate(*get(b)) for b in candidates if b != safe}
    explore = sorted(draws, key=draws.get, reverse=True)[: SHOWN - 1]
    chosen = [safe, *explore]
    shuffled = rng.random() < SHUFFLE_PROB
    if shuffled:
        rng.shuffle(chosen)
    return [Shown(b, i, shuffled, draws.get(b)) for i, b in enumerate(chosen)]


def active_styles(counts: dict[str, tuple[float, float]]) -> list[str]:
    """Modifiers whose posterior mean is above 0.5 with at least 2 signals."""
    out = []
    for style in STYLES:
        a, b = counts.get("style:" + style, (1.0, 1.0))
        if a + b - 2 >= 2 and beta_mean((a, b)) > 0.5:
            out.append(style)
    return out
