"""Global learned parameters (refit nightly) and their initial values."""

import dataclasses
import math
import time

from app import db
from app.model import family as family_lib
from app.model import reaction
from app.model import reading as reading_lib
from app.model import theta

# Initial log writing speeds, in assistant-counted words per second.
INITIAL_SPEED = dict(
    zip(family_lib.FAMILIES, map(math.log, (35, 50, 90, 50, 40)))
)
INITIAL_NOISE = {"self_report": 0.3, "writing_time": 0.5, "requested": 0.25}
READING_PRIOR = (6.0, 3.0, 1.0)  # Dirichlet counts: read, skim, away.
REACTION_PRIOR = (6.0,) + (1.0,) * (len(reaction.OUTCOMES) - 1)
RHO_PRIOR = (math.log(4.0), 0.25)  # Log words per second, variance.
CACHE_SECONDS = 60.0
ALL_FAMILIES = "all"  # Sensor noise shared by every model family.


def _initial_family() -> dict[str, dict[str, float]]:
    return {
        f: {"b": 0.0, "w": 0.05, "kappa": 0.0, "speed_mu": s, "speed_sd": 0.4}
        for f, s in INITIAL_SPEED.items()
    }


def _initial_reliability() -> dict[str, dict[str, float]]:
    off = 0.15 / (len(family_lib.FAMILIES) - 1)
    return {
        r: {f: 0.85 if f == r else off for f in family_lib.FAMILIES}
        for r in family_lib.FAMILIES
    }


@dataclasses.dataclass
class Params:
    """Every learned global parameter, at its initial value until refit."""

    hyper: theta.Hyper = dataclasses.field(default_factory=theta.Hyper)
    reading: reading_lib.ReadingParams = dataclasses.field(
        default_factory=reading_lib.ReadingParams
    )
    overhead: float = 3.0
    family: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=_initial_family
    )
    noise: dict[str, float] = dataclasses.field(
        default_factory=lambda: dict(INITIAL_NOISE)
    )
    confusions: dict[str, reaction.Confusion] = dataclasses.field(
        default_factory=lambda: reaction.default_confusions(family_lib.FAMILIES)
    )
    reliability: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=_initial_reliability
    )
    host_prior: dict[str, dict[str, float]] = dataclasses.field(
        default_factory=dict
    )

    def host_family_prior(self, host: str) -> dict[str, float]:
        """Returns P(family | host): learned, else uniform."""
        uniform = dict.fromkeys(
            family_lib.FAMILIES, 1 / len(family_lib.FAMILIES)
        )
        return self.host_prior.get(host, uniform)


HYPER_NAMES = ("mu0", "sigma_alpha", "sigma_gamma", "sigma_eta", "tau")
READING_NAMES = {
    "reading_s_r": "s_r",
    "reading_mu_away": "mu_away",
    "reading_s_away": "s_away",
}
GAMMA_PREFIX = "gamma:"


async def load_params(conn: db.Connection) -> Params:
    """Returns the parameters, learned values over the initial ones."""
    p = Params()
    rows = await db.fetch(conn, "SELECT name, value FROM hyperparams")
    hyper = {r["name"]: r["value"] for r in rows}
    p.hyper = dataclasses.replace(
        p.hyper,
        **{n: hyper[n] for n in HYPER_NAMES if n in hyper},
        gamma={
            k.removeprefix(GAMMA_PREFIX): v
            for k, v in hyper.items()
            if k.startswith(GAMMA_PREFIX)
        },
    )
    p.overhead = hyper.get("overhead", p.overhead)
    p.reading = dataclasses.replace(
        p.reading,
        **{f: hyper[n] for n, f in READING_NAMES.items() if n in hyper},
    )
    for r in await db.fetch(conn, "SELECT * FROM family_params"):
        p.family[r["model_family"]] = {k: r[k] for k in p.family["other"]}
    for r in await db.fetch(
        conn,
        "SELECT sensor, sigma FROM sensor_noise WHERE model_family = %s",
        ALL_FAMILIES,
    ):
        p.noise[r["sensor"]] = r["sigma"]
    for r in await db.fetch(conn, "SELECT * FROM annotator_confusion"):
        strata = p.confusions.setdefault(r["annotator"], {})
        rows_of = strata.setdefault(r["stratum"], {})
        rows_of.setdefault(r["true_outcome"], {})[r["observed"]] = r["prob"]
    for r in await db.fetch(conn, "SELECT * FROM self_report_reliability"):
        p.reliability[r["reported_family"]][r["true_family"]] = r["prob"]
    for r in await db.fetch(conn, "SELECT * FROM host_family_prior"):
        p.host_prior.setdefault(r["host"], {})[r["model_family"]] = r["prob"]
    return p


class _Cache:
    """The parameters, reloaded at most every CACHE_SECONDS."""

    def __init__(self):
        self.loaded_at = -math.inf
        self.params: Params | None = None

    async def get(self, conn: db.Connection) -> Params:
        if time.monotonic() - self.loaded_at > CACHE_SECONDS:
            self.params = await load_params(conn)
            self.loaded_at = time.monotonic()
        return self.params

    def invalidate(self) -> None:
        self.loaded_at = -math.inf


cache = _Cache()
