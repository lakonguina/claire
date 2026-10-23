"""Thompson sampling for reply form and word length (per person).

Each person has Dirichlet counts over the closed form catalog and the word
buckets. We draw one sample, pick the max, and estimate the propensity by
re-drawing many times — so later policies can be scored on the same log.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from app.core import config, vocab
from app.mcp.schemas import PlannedPartIn, PlannedPartOut

PROPENSITY_SAMPLES = 200
FormArm = tuple[vocab.Form, ...]


def form_key(forms: Sequence[vocab.Form | str]) -> str:
    """Stable catalog key for a form arm (e.g. ``text+bullets``)."""
    return "+".join(
        f.value if isinstance(f, vocab.Form) else str(f) for f in forms
    )


FORM_ARMS: tuple[FormArm, ...] = vocab.FORM_OPTIONS
FORM_KEYS: tuple[str, ...] = tuple(form_key(arm) for arm in FORM_ARMS)
FORM_BY_KEY: dict[str, FormArm] = dict(zip(FORM_KEYS, FORM_ARMS, strict=True))
WORDS_KEYS: tuple[str, ...] = tuple(str(bucket) for bucket in vocab.WORD_BUCKETS)


def _prior() -> float:
    return float(config.settings.cell_prior_strength)


def _alphas(
    counts: Mapping[str, float],
    arms: Sequence[str],
    prior: float | None = None,
) -> dict[str, float]:
    strength = _prior() if prior is None else prior
    return {arm: float(counts.get(arm, 0.0)) + strength for arm in arms}


def _draw(alphas: Mapping[str, float], rng: random.Random) -> dict[str, float]:
    """One Dirichlet sample (Gamma / sum)."""
    gammas = {key: rng.gammavariate(max(alpha, 1e-6), 1.0) for key, alpha in alphas.items()}
    total = sum(gammas.values()) or 1.0
    return {key: value / total for key, value in gammas.items()}


def thompson(
    counts: Mapping[str, float],
    arms: Sequence[str],
    rng: random.Random,
) -> str:
    """Return the arm with the highest Dirichlet draw."""
    sample = _draw(_alphas(counts, arms), rng)
    return max(sample, key=sample.get)


def with_propensity(
    counts: Mapping[str, float],
    arms: Sequence[str],
    rng: random.Random,
    *,
    samples: int = PROPENSITY_SAMPLES,
) -> tuple[str, float]:
    """Return a Thompson choice and P(same choice) under re-draws.

    Args:
        counts: Observed pseudo-counts for this person (arm → weight).
        arms: Closed catalog of arm keys.
        rng: Random source (tests pass a fixed seed).
        samples: Monte-Carlo draws for the propensity estimate.

    Returns:
        ``(arm_key, propensity)`` with propensity in ``(0, 1]``.
    """
    choice = thompson(counts, arms, rng)
    same = sum(
        thompson(counts, arms, rng) == choice for _ in range(samples)
    )
    return choice, max(same / samples, 1.0 / samples)


def nearest_bucket(words: int) -> int:
    """Snap a word target onto ``WORD_BUCKETS``."""
    return min(vocab.WORD_BUCKETS, key=lambda bucket: abs(bucket - words))


def _apply_length_forced(words: int, forced: vocab.ForcedLength | None) -> int:
    if forced == vocab.ForcedLength.SHORTER:
        return max(vocab.WORD_BUCKETS[0], words // 2)
    if forced == vocab.ForcedLength.LONGER:
        return min(vocab.WORD_BUCKETS[-1], words * 2)
    return words


def decide_part(
    part: PlannedPartIn,
    *,
    form_counts: Mapping[str, float],
    words_counts: Mapping[str, float],
    rng: random.Random,
) -> tuple[PlannedPartOut, dict]:
    """Choose form and length for one part; return the part and log meta.

    Forced form / length skip the bandit (propensity 1). Otherwise Thompson
    samples the person-level catalogs.

    Args:
        part: Host proposal for this slice.
        form_counts: Person form arm weights.
        words_counts: Person word-bucket weights.
        rng: Random source.

    Returns:
        ``(planned_part, decision_meta)`` where meta holds arms + propensities.
    """
    meta: dict = {"name": part.name}

    if part.form_forced and part.form:
        forms = tuple(part.form)
        form_p = 1.0
        meta["form_forced"] = True
    else:
        key, form_p = with_propensity(form_counts, FORM_KEYS, rng)
        forms = FORM_BY_KEY[key]
        meta["form_arm"] = key
    meta["form_propensity"] = form_p

    if part.words_forced:
        # Person gave an absolute target → snap to catalog, skip bandit.
        words = nearest_bucket(part.words)
        words_p = 1.0
        meta["words_forced"] = True
        meta["words_arm"] = str(words)
    elif part.length_forced is not None:
        # Direction only → Claire picks how far to move from the estimate.
        words = nearest_bucket(
            _apply_length_forced(part.words, part.length_forced)
        )
        words_p = 1.0
        meta["length_forced"] = part.length_forced.value
        meta["words_arm"] = str(words)
    else:
        key, words_p = with_propensity(words_counts, WORDS_KEYS, rng)
        words = int(key)
        meta["words_arm"] = key
    meta["words_propensity"] = words_p

    return (
        PlannedPartOut(name=part.name, form=list(forms), words=words),
        meta,
    )


def decide_parts(
    parts: Sequence[PlannedPartIn],
    *,
    form_counts: Mapping[str, float] | None = None,
    words_counts: Mapping[str, float] | None = None,
    rng: random.Random | None = None,
) -> tuple[list[PlannedPartOut], list[dict]]:
    """Decide every part; return parts and per-part decision metas.

    Args:
        parts: Host plan parts.
        form_counts: Person form weights (empty → prior only).
        words_counts: Person word-bucket weights.
        rng: Optional RNG (default ``random.Random()``).

    Returns:
        ``(out_parts, metas)``.
    """
    rng = rng or random.Random()
    form_counts = form_counts or {}
    words_counts = words_counts or {}
    out: list[PlannedPartOut] = []
    metas: list[dict] = []
    for part in parts:
        planned, meta = decide_part(
            part,
            form_counts=form_counts,
            words_counts=words_counts,
            rng=rng,
        )
        out.append(planned)
        metas.append(meta)
    return out, metas
