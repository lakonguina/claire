"""Update person bandit counts from forced fields on the next plan.

The host LLM is the sensor: when it calls ``claire_plan_reply`` again with
``form_forced``, ``words_forced``, or ``length_forced``, that closes the
previous turn and tells us what went wrong. The forced args *are* the signal.

* No force → previous form and length look fine → credit both arms.
* ``form_forced`` → withhold form credit.
* ``words_forced`` or ``length_forced`` → withhold length credit.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from app import models
from app.core import vocab
from app.mcp.schemas import PlannedPartIn

SUCCESS_WEIGHT = 1.0


def is_trainable(meta: Mapping[str, Any] | None) -> bool:
    """Return whether this decision should update posteriors."""
    if not meta:
        return False
    if meta.get("holdout") or meta.get("mode") == vocab.Mode.NEUTRAL:
        return False
    return True


def forced_signal(parts: Sequence[PlannedPartIn]) -> dict[str, bool]:
    """Summarize learning signal from the new plan's forced flags.

    Args:
        parts: Parts of the *current* ``claire_plan_reply`` call.

    Returns:
        Flags: any form force; any length force (absolute or directional).
    """
    return {
        "form_forced": any(part.form_forced for part in parts),
        "length_forced": any(
            part.words_forced or part.length_forced is not None
            for part in parts
        ),
        "words_forced": any(part.words_forced for part in parts),
    }


def _bump(counts: dict[str, float], arm: str | None, weight: float) -> None:
    if not arm or weight <= 0:
        return
    counts[arm] = float(counts.get(arm, 0.0)) + weight


def _length_was_forced(part_meta: Mapping[str, Any]) -> bool:
    """True if this previous decision did not choose length freely."""
    return bool(part_meta.get("words_forced") or part_meta.get("length_forced"))


def apply_turn_outcome(
    person: models.Person,
    decision_meta: Mapping[str, Any],
    *,
    form_forced: bool,
    length_forced: bool,
) -> None:
    """Update bandits for the closed turn from the next plan's forced flags.

    Args:
        person: Person whose counts are updated in place.
        decision_meta: Previous turn's ``decision`` event meta (``parts``).
        form_forced: Host imposed form on the new plan → prior form failed.
        length_forced: Host imposed absolute or directional length → prior
            length failed.
    """
    if not is_trainable(decision_meta):
        return

    form_weight = 0.0 if form_forced else SUCCESS_WEIGHT
    words_weight = 0.0 if length_forced else SUCCESS_WEIGHT

    form_counts = dict(person.form_counts or {})
    words_counts = dict(person.words_counts or {})
    for part_meta in decision_meta.get("parts") or []:
        if not isinstance(part_meta, dict):
            continue
        # Arms forced on the *previous* decision were not Claire's choice.
        if not part_meta.get("form_forced"):
            _bump(form_counts, part_meta.get("form_arm"), form_weight)
        if not _length_was_forced(part_meta):
            _bump(words_counts, part_meta.get("words_arm"), words_weight)
    person.form_counts = form_counts
    person.words_counts = words_counts
