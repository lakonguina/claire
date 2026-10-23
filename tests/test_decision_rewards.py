"""Thompson decision and forced-flag rewards."""

from __future__ import annotations

import random

from app import models
from app.core import vocab
from app.mcp.schemas import PlannedPartIn
from app.model import decision, rewards


def test_words_forced_snaps_bucket() -> None:
    parts, metas = decision.decide_parts(
        [PlannedPartIn(name="a", words=95, words_forced=True)],
        rng=random.Random(0),
    )
    assert parts[0].words == 80
    assert metas[0]["words_forced"] is True


def test_length_forced_shorter() -> None:
    parts, metas = decision.decide_parts(
        [
            PlannedPartIn(
                name="a",
                words=100,
                length_forced=vocab.ForcedLength.SHORTER,
            )
        ],
        rng=random.Random(0),
    )
    assert parts[0].words <= 80
    assert metas[0]["length_forced"] == "shorter"


def test_form_forced() -> None:
    parts, metas = decision.decide_parts(
        [
            PlannedPartIn(
                name="a",
                words=40,
                form=[vocab.Form.BULLETS],
                form_forced=True,
            )
        ],
        rng=random.Random(0),
    )
    assert parts[0].form == [vocab.Form.BULLETS]
    assert metas[0]["form_propensity"] == 1.0


def test_reward_success_credits_arms() -> None:
    person = models.Person(form_counts={}, words_counts={})
    rewards.apply_turn_outcome(
        person,
        {
            "mode": "adapted",
            "parts": [{"form_arm": "text", "words_arm": "80"}],
        },
        form_forced=False,
        length_forced=False,
    )
    assert person.form_counts["text"] == 1.0
    assert person.words_counts["80"] == 1.0


def test_reward_form_forced_withholds_form() -> None:
    person = models.Person(form_counts={}, words_counts={})
    rewards.apply_turn_outcome(
        person,
        {
            "mode": "adapted",
            "parts": [{"form_arm": "text", "words_arm": "80"}],
        },
        form_forced=True,
        length_forced=False,
    )
    assert "text" not in person.form_counts
    assert person.words_counts["80"] == 1.0


def test_forced_signal() -> None:
    signal = rewards.forced_signal(
        [PlannedPartIn(name="a", words=40, words_forced=True)]
    )
    assert signal["words_forced"] is True
    assert signal["length_forced"] is True
