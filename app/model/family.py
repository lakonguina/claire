"""Assistant behaviour (SPEC 6.3): soft model-family posterior."""

import math

from app import vocab
from app.model import stats

FAMILIES = list(vocab.ModelFamily)
FamilyParams = dict[str, dict[str, float]]


def reported_family(assistant_model: str | None) -> str | None:
    """Maps a self-reported model name to the family it names, if any."""
    if not assistant_model:
        return None
    words = set(assistant_model.lower().replace("_", "-").split("-"))
    named = [
        f
        for f in FAMILIES
        if f != vocab.ModelFamily.OTHER and f.marker in words
    ]
    return named[0] if named else vocab.ModelFamily.OTHER


def initial_probs(
    host_prior: dict[str, float],
    reported: str | None,
    reliability: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Returns pi(c, f) before any turn: P(f | host) x P(self-report | f)."""
    weights = [
        host_prior.get(f, 1e-3)
        * (reliability[reported].get(f, 1e-3) if reported else 1.0)
        for f in FAMILIES
    ]
    return dict(zip(FAMILIES, stats.normalize(weights)))


def update_with_speed(
    probs: dict[str, float], speed: float, params: FamilyParams
) -> dict[str, float]:
    """Multiplies in N(v; mu_v(f), s_v(f)^2) for one turn's log speed."""
    logs = [
        math.log(max(probs[f], 1e-12))
        + stats.log_normal_pdf(
            speed, params[f]["speed_mu"], params[f]["speed_sd"] ** 2
        )
        for f in FAMILIES
    ]
    return dict(zip(FAMILIES, stats.normalize_log(logs)))


def mix(probs: dict[str, float], params: FamilyParams, key: str) -> float:
    """Returns the family mixture of one parameter."""
    return sum(probs[f] * params[f][key] for f in FAMILIES)


def _mixture(
    probs: dict[str, float],
    params: FamilyParams,
    mean_key: str,
    second: dict[str, float],
) -> tuple[float, float]:
    mean = mix(probs, params, mean_key)
    return mean, sum(probs[f] * second[f] for f in FAMILIES) - mean**2


def bias(probs: dict[str, float], params: FamilyParams) -> tuple[float, float]:
    """Returns b(c) and Var b(c) for the family mixture."""
    second = {f: p["b"] ** 2 + p["w"] for f, p in params.items()}
    b, var = _mixture(probs, params, "b", second)
    return b, max(var, 0.0)


def speed_prior(
    probs: dict[str, float], params: FamilyParams
) -> tuple[float, float]:
    """Returns the mixture mean and variance of the log writing speed."""
    second = {
        f: p["speed_mu"] ** 2 + p["speed_sd"] ** 2 for f, p in params.items()
    }
    mean, var = _mixture(probs, params, "speed_mu", second)
    return mean, max(var, 1e-4)
