"""Global learned parameters (refit nightly) and their initial values."""

import dataclasses
import math
import time

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import models
from app.model import family as family_lib
from app.model import reaction
from app.model import reading as reading_lib
from app.model import theta

INITIAL_NOISE = {"self_report": 0.3, "writing_time": 0.5, "requested": 0.25}
READING_PRIOR = (6.0, 3.0, 1.0)  # Dirichlet counts: read, skim, away.
REACTION_PRIOR = (6.0,) + (1.0,) * (len(reaction.OUTCOMES) - 1)
RHO_PRIOR = (math.log(4.0), 0.25)  # Log words per second, variance.
CACHE_SECONDS = 60.0
ALL_FAMILIES = "all"  # Sensor noise shared by every model family.


@dataclasses.dataclass
class Params:
    """Every learned global parameter, at its initial value until refit."""

    prior: theta.Prior = dataclasses.field(default_factory=theta.Prior)
    reading: reading_lib.ReadingParams = dataclasses.field(
        default_factory=reading_lib.ReadingParams
    )
    overhead: float = 3.0
    families: family_lib.Families = dataclasses.field(
        default_factory=family_lib.Families
    )
    noise: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(INITIAL_NOISE)
    )
    confusions: dict[str, reaction.Confusion] = dataclasses.field(
        default_factory=reaction.default_confusions
    )


READING_NAMES = {
    "reading_s_r": "s_r",
    "reading_mu_away": "mu_away",
    "reading_s_away": "s_away",
}
# Hyperparameter names of the length prior: "<prefix><feature>".
PRIOR_MEAN, PRIOR_VAR = "prior_mean:", "prior_var:"


def _prefixed(values: dict[str, float], prefix: str) -> dict[str, float]:
    return {
        k.removeprefix(prefix): v
        for k, v in values.items()
        if k.startswith(prefix)
    }


async def load_params(session: sa_asyncio.AsyncSession) -> Params:
    """Returns the parameters, learned values over the initial ones."""

    async def rows(model):
        return list(await session.scalars(sa.select(model)))

    p = Params()
    hyper = {r.name: r.value for r in await rows(models.Hyperparam)}
    p.prior.mean |= _prefixed(hyper, PRIOR_MEAN)
    p.prior.var |= _prefixed(hyper, PRIOR_VAR)
    p.prior.tau = hyper.get("tau", p.prior.tau)
    p.overhead = hyper.get("overhead", p.overhead)
    p.reading = dataclasses.replace(
        p.reading,
        **{f: hyper[n] for n, f in READING_NAMES.items() if n in hyper},
    )
    families = p.families
    for r in await rows(models.FamilyParam):
        families.params[r.model_family] = {
            k: getattr(r, k) for k in family_lib.FIELDS
        }
    for r in await rows(models.SensorNoise):
        if r.model_family == ALL_FAMILIES:
            p.noise[r.sensor] = r.sigma
    for r in await rows(models.AnnotatorConfusion):
        strata = p.confusions.setdefault(r.annotator, {})
        rows_of = strata.setdefault(r.stratum, {})
        rows_of.setdefault(r.true_outcome, {})[r.observed] = r.prob
    for r in await rows(models.HostFamilyPrior):
        families.host_prior.setdefault(r.host, {})[r.model_family] = r.prob
    return p


class _Cache:
    """The parameters, reloaded at most every CACHE_SECONDS."""

    def __init__(self):
        self.loaded_at = -math.inf
        self.params: Params | None = None

    async def get(self, session: sa_asyncio.AsyncSession) -> Params:
        if time.monotonic() - self.loaded_at > CACHE_SECONDS:
            self.params = await load_params(session)
            self.loaded_at = time.monotonic()
        return self.params

    def invalidate(self) -> None:
        self.loaded_at = -math.inf


cache = _Cache()
