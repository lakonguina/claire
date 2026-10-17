"""User reaction (section 6.4): noisy annotators, online fusion and Dawid-Skene."""

from dataclasses import dataclass, field

from app.model.stats import normalize

OUTCOMES = ["moved_on", "too_long", "too_short", "missing_info", "misunderstood", "wrong_format"]
NO_INFO = "_none"  # observation carrying no information (P = 1 for every true outcome)

BUTTON_OUTCOME = {
    "shorter": "too_long",
    "more_detail": "too_short",
    "example": "style_click",
    "simpler": "wrong_format",
    "one_line": "too_long",
    "rephrase_request": "misunderstood",
}
POSITIONS = 3
# The button annotator observes one joint token per turn: "<position>:<outcome>" for a click, or "no_click".
BUTTON_OBS = ["no_click"] + [
    f"{p}:{o}" for p in range(POSITIONS) for o in sorted(set(BUTTON_OUTCOME.values()))
]
ALPHABETS = {"assistant": OUTCOMES, "button": BUTTON_OBS, "reading": ["read", "skim"]}

Confusion = dict[str, dict[str, dict[str, float]]]  # stratum -> true outcome -> observed -> prob


@dataclass
class Observation:
    annotator: str
    dist: dict[str, float]  # observed value -> probability (soft observations)
    strata: dict[str, float] = field(default_factory=lambda: {"all": 1.0})  # stratum -> weight


@dataclass
class Item:
    observations: list[Observation]
    gold: str | None = None


def _rows(diagonal: dict[str, float], alphabet: list[str]) -> dict[str, dict[str, float]]:
    rows = {}
    for y in OUTCOMES:
        d = diagonal.get(y, 0.0)
        rest = [o for o in alphabet if o != y]
        rows[y] = {o: d if o == y else (1 - d) / len(rest) for o in alphabet}
    return rows


def default_confusions(families: list[str]) -> dict[str, Confusion]:
    assistant = _rows({y: 0.7 if y == "moved_on" else 0.8 for y in OUTCOMES}, OUTCOMES)
    button = {}
    for y in OUTCOMES:
        no_click = 0.9 if y == "moved_on" else 0.6
        matching = [o for o in BUTTON_OBS[1:] if o.endswith(":" + y)]
        others = [o for o in BUTTON_OBS[1:] if o not in matching]
        row = {"no_click": no_click}
        for o in BUTTON_OBS[1:]:
            if matching:
                row[o] = (1 - no_click) * (0.95 / len(matching) if o in matching else 0.05 / len(others))
            else:
                row[o] = (1 - no_click) / len(BUTTON_OBS[1:])
        button[y] = row
    reading = {y: {"skim": 0.5 if y == "too_long" else 0.25} for y in OUTCOMES}
    for row in reading.values():
        row["read"] = 1 - row["skim"]
    return {
        "assistant": {f: {y: dict(row) for y, row in assistant.items()} for f in families},
        "button": {"all": button},
        "reading": {"all": reading},
    }


def likelihood(obs: Observation, confusion: Confusion) -> list[float]:
    out = []
    for y in OUTCOMES:
        total = 0.0
        for stratum, w in obs.strata.items():
            row = confusion[stratum][y]
            total += w * sum(p * (1.0 if o == NO_INFO else row.get(o, 0.0)) for o, p in obs.dist.items())
        out.append(total)
    return out


def posterior(prior: list[float], observations: list[Observation], confusions: dict[str, Confusion]) -> list[float]:
    post = list(prior)
    for obs in observations:
        post = [p * lik for p, lik in zip(post, likelihood(obs, confusions[obs.annotator]))]
    return normalize(post)


def dawid_skene(
    items: list[Item], confusions: dict[str, Confusion], iterations: int = 50, smoothing: float = 0.01
) -> tuple[dict[str, Confusion], list[float], list[list[float]]]:
    """EM over true outcomes and every annotator's confusion matrix; gold labels are clamped."""
    prior = [1.0 / len(OUTCOMES)] * len(OUTCOMES)
    posts: list[list[float]] = []
    for _ in range(iterations):
        posts = [
            [1.0 if y == item.gold else 0.0 for y in OUTCOMES] if item.gold else posterior(prior, item.observations, confusions)
            for item in items
        ]
        counts = {
            a: {s: {y: {o: smoothing for o in ALPHABETS[a]} for y in OUTCOMES} for s in strata}
            for a, strata in confusions.items()
        }
        for item, post in zip(items, posts):
            for obs in item.observations:
                for stratum, w in obs.strata.items():
                    for o, p in obs.dist.items():
                        if o == NO_INFO:
                            continue
                        for y, py in zip(OUTCOMES, post):
                            counts[obs.annotator][stratum][y][o] += py * w * p
        confusions = {
            a: {
                s: {y: dict(zip(row, normalize(list(row.values())))) for y, row in rows.items()}
                for s, rows in strata.items()
            }
            for a, strata in counts.items()
        }
        prior = normalize([sum(post[i] for post in posts) + 1.0 for i in range(len(OUTCOMES))])
    return confusions, prior, posts

