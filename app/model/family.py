"""Assistant behaviour (section 6.3): soft model-family posterior and the quantities mixed from it."""

import math

from app.model.stats import log_normal_pdf, normalize, normalize_log

FAMILIES = ["claude-opus", "claude-sonnet", "claude-haiku", "gpt", "other"]


def reported_family(assistant_model: str | None) -> str | None:
    if not assistant_model:
        return None
    name = assistant_model.lower()
    for key in ("opus", "sonnet", "haiku"):
        if key in name:
            return f"claude-{key}"
    return "gpt" if "gpt" in name else "other"


def initial_probs(
    host_prior: dict[str, float], reported: str | None, reliability: dict[str, dict[str, float]]
) -> dict[str, float]:
    """pi(c, f) before any turn: P(f | host) x P(self-report | f)."""
    weights = [
        host_prior.get(f, 1e-3) * (reliability[reported].get(f, 1e-3) if reported else 1.0) for f in FAMILIES
    ]
    return dict(zip(FAMILIES, normalize(weights)))


def update_with_speed(probs: dict[str, float], speed: float, params: dict[str, dict]) -> dict[str, float]:
    """Multiply in N(v_hat; mu_v(f), s_v(f)^2) for one turn's estimated log writing speed."""
    logs = [
        math.log(max(probs[f], 1e-12)) + log_normal_pdf(speed, params[f]["speed_mu"], params[f]["speed_sd"] ** 2)
        for f in FAMILIES
    ]
    return dict(zip(FAMILIES, normalize_log(logs)))


def mix(probs: dict[str, float], params: dict[str, dict], key: str) -> float:
    return sum(probs[f] * params[f][key] for f in FAMILIES)


def bias(probs: dict[str, float], params: dict[str, dict]) -> tuple[float, float]:
    """b(c) and Var b(c) for the family mixture."""
    b = mix(probs, params, "b")
    second = sum(probs[f] * (params[f]["b"] ** 2 + params[f]["w"]) for f in FAMILIES)
    return b, max(second - b * b, 0.0)


def speed_prior(probs: dict[str, float], params: dict[str, dict]) -> tuple[float, float]:
    """Mixture mean and variance of the log writing speed."""
    mean = mix(probs, params, "speed_mu")
    second = sum(probs[f] * (params[f]["speed_mu"] ** 2 + params[f]["speed_sd"] ** 2) for f in FAMILIES)
    return mean, max(second - mean * mean, 1e-4)
