"""Rebuild bandit counts from the append-only event log.

Online writes update ``people.form_counts`` / ``words_counts`` as turns
close. Nightly replay walks ``decision`` then ``turn_closed`` in order and
rebuilds the same counters — that is the ``replay = online`` check.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from app.core import vocab
from app.model import rewards


@dataclass
class BanditState:
    """Person-level Dirichlet weights rebuilt from history."""

    form_counts: dict[str, float] = field(default_factory=dict)
    words_counts: dict[str, float] = field(default_factory=dict)


def _as_float_map(raw: Mapping[str, Any] | None) -> dict[str, float]:
    return {str(key): float(value) for key, value in (raw or {}).items()}


def rebuild_bandits(
    events: Iterable[Any],
) -> dict[int, BanditState]:
    """Replay decisions + closes into fresh bandit counts.

    Args:
        events: Rows with ``person_id``, ``type``, ``turn_id``, ``meta``,
            in ascending id / time order.

    Returns:
        ``person_id →`` rebuilt :class:`BanditState`.
    """
    decisions: dict[str, dict[str, Any]] = {}
    states: dict[int, BanditState] = defaultdict(BanditState)

    for event in events:
        event_type = str(getattr(event, "type", ""))
        turn_id = getattr(event, "turn_id", None)
        person_id = int(getattr(event, "person_id"))
        meta = dict(getattr(event, "meta", None) or {})

        if event_type == vocab.EventType.DECISION and turn_id:
            decisions[turn_id] = meta
            continue

        if event_type != vocab.EventType.TURN_CLOSED or not turn_id:
            continue

        decision_meta = decisions.get(turn_id)
        if decision_meta is None:
            continue

        state = states[person_id]
        # Duck-typed stand-in: rewards only touch form_counts / words_counts.
        holder = SimpleNamespace(
            form_counts=dict(state.form_counts),
            words_counts=dict(state.words_counts),
        )
        rewards.apply_turn_outcome(
            holder,  # type: ignore[arg-type]
            decision_meta,
            form_forced=bool(meta.get("form_forced")),
            length_forced=bool(meta.get("length_forced")),
        )
        state.form_counts = _as_float_map(holder.form_counts)
        state.words_counts = _as_float_map(holder.words_counts)

    return dict(states)


def counts_match(
    online: Mapping[str, Any] | None,
    replayed: Mapping[str, float],
) -> bool:
    """Return whether two count maps agree (float-tolerant)."""
    left = _as_float_map(online)
    right = _as_float_map(replayed)
    if set(left) != set(right):
        return False
    return all(abs(left[key] - right[key]) < 1e-9 for key in left)
