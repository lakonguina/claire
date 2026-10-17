"""Global learned parameters (refit nightly), with their initial values when nothing is learned yet."""

import math
import time
from dataclasses import dataclass, field

from app.db import AsyncConnection, fetch
from app.model.family import FAMILIES
from app.model.reaction import Confusion, default_confusions
from app.model.reading import ReadingParams
from app.model.theta import Hyper

DEFAULT_FAMILY = {
    # Log writing speed in assistant-counted words per second.
    "claude-opus": math.log(35), "claude-sonnet": math.log(50), "claude-haiku": math.log(90),
    "gpt": math.log(50), "other": math.log(40),
}
DEFAULT_NOISE = {"self_report": 0.3, "writing_time": 0.5, "requested": 0.25}
DEFAULT_READING_WEIGHTS = [6.0, 3.0, 1.0]  # Dirichlet counts for read / skim / away
DEFAULT_REACTION_PRIOR = [6.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Dirichlet counts over OUTCOMES
RHO_PRIOR = (math.log(4.0), 0.25)  # log words per second, variance


def default_host_prior(host: str) -> dict[str, float]:
    name = host.lower()
    if "claude" in name or "anthropic" in name:
        shares = [0.3, 0.5, 0.15, 0.01, 0.04]
    elif "openai" in name or "chatgpt" in name:
        shares = [0.01, 0.01, 0.01, 0.9, 0.07]
    else:
        shares = [0.2] * 5
    return dict(zip(FAMILIES, shares))


@dataclass
class Params:
    hyper: Hyper = field(default_factory=Hyper)
    reading: ReadingParams = field(default_factory=ReadingParams)
    overhead: float = 3.0
    family: dict[str, dict[str, float]] = field(default_factory=lambda: {
        f: {"b": 0.0, "w": 0.05, "kappa": 0.0, "speed_mu": DEFAULT_FAMILY[f], "speed_sd": 0.4} for f in FAMILIES
    })
    noise: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_NOISE))
    confusions: dict[str, Confusion] = field(default_factory=lambda: default_confusions(FAMILIES))
    reliability: dict[str, dict[str, float]] = field(default_factory=lambda: {
        r: {f: 0.85 if f == r else 0.15 / (len(FAMILIES) - 1) for f in FAMILIES} for r in FAMILIES
    })
    host_prior: dict[str, dict[str, float]] = field(default_factory=dict)

    def host_family_prior(self, host: str) -> dict[str, float]:
        return self.host_prior.get(host) or default_host_prior(host)


async def load_params(conn: AsyncConnection) -> Params:
    p = Params()
    hyper = {r["name"]: r["value"] for r in await fetch(conn, "SELECT name, value FROM hyperparams")}
    for name in ("mu0", "sigma_alpha", "sigma_gamma", "sigma_eta", "tau"):
        if name in hyper:
            setattr(p.hyper, name, hyper[name])
    p.hyper.gamma = {k.removeprefix("gamma:"): v for k, v in hyper.items() if k.startswith("gamma:")}
    p.overhead = hyper.get("overhead", p.overhead)
    p.reading = ReadingParams(hyper.get("reading_s_r", p.reading.s_r), hyper.get("reading_mu_away", p.reading.mu_away),
                              hyper.get("reading_s_away", p.reading.s_away))
    for r in await fetch(conn, "SELECT * FROM family_params"):
        p.family[r["model_family"]] = {k: r[k] for k in ("b", "w", "kappa", "speed_mu", "speed_sd")}
    for r in await fetch(conn, "SELECT sensor, sigma FROM sensor_noise WHERE model_family = 'all'"):
        p.noise[r["sensor"]] = r["sigma"]
    for r in await fetch(conn, "SELECT * FROM annotator_confusion"):
        p.confusions.setdefault(r["annotator"], {}).setdefault(r["stratum"], {}).setdefault(
            r["true_outcome"], {})[r["observed"]] = r["prob"]
    for r in await fetch(conn, "SELECT * FROM self_report_reliability"):
        p.reliability[r["reported_family"]][r["true_family"]] = r["prob"]
    for r in await fetch(conn, "SELECT * FROM host_family_prior"):
        p.host_prior.setdefault(r["host"], {})[r["model_family"]] = r["prob"]
    return p


_cache: tuple[float, Params] | None = None


async def cached_params(conn: AsyncConnection, ttl: float = 60.0) -> Params:
    global _cache
    if _cache is None or time.monotonic() - _cache[0] > ttl:
        _cache = (time.monotonic(), await load_params(conn))
    return _cache[1]


def invalidate() -> None:
    global _cache
    _cache = None
