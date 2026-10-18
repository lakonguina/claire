"""Model behaviour and recovery on synthetic data (SPEC 12)."""

import json
import math
import random

import numpy as np
import pytest

from app import i18n
from app import service
from app import text
from app import topics
from app import vocab
from app.model import bandits
from app.model import family
from app.model import length
from app.model import matching
from app.model import policy
from app.model import prefs
from app.model import reaction
from app.model import reading
from app.model import theta

CONF = reaction.default_confusions()
FAMILY_PROBS = {"class-1": 0.5, "class-2": 0.5}


def outcome_probs(verdict: str) -> list[float]:
    prior = [1 / len(reaction.OUTCOMES)] * len(reaction.OUTCOMES)
    obs = reaction.Observation(
        vocab.Source.ASSISTANT, {verdict: 1.0}, FAMILY_PROBS
    )
    return reaction.posterior(prior, [obs], CONF)


CODE, COOKING = "topic:1", "topic:2"


def belief_after(verdict: str, x: dict) -> theta.Belief:
    belief = theta.Belief.empty()
    theta.update(
        belief,
        x,
        math.log(0.6),
        0.0,
        (0.0, 0.0),
        outcome_probs(verdict),
        theta.Prior(),
    )
    return belief


def test_outcomes_move_the_preference_the_right_way():
    x = theta.features({CODE: 1.0}, mobile=False)
    before = theta.Belief.empty().predict(x, theta.Prior())[0]
    too_long, too_short, moved_on = (
        belief_after(v, x).predict(x, theta.Prior())[0]
        for v in ("too_long", "too_short", "moved_on")
    )
    assert too_long < moved_on < before < too_short


def test_learning_is_shared_by_nearby_requests_only():
    prior = theta.Prior()
    belief = theta.Belief.empty()
    code = theta.features({CODE: 1.0}, mobile=False)
    for _ in range(5):
        theta.update(
            belief,
            code,
            math.log(0.6),
            0.0,
            (0.0, 0.0),
            outcome_probs("too_long"),
            prior,
        )
    shift = {
        name: belief.predict(theta.features(d, False), prior)[0]
        - theta.Belief.empty().predict(theta.features(d, False), prior)[0]
        for name, d in (
            ("code", {CODE: 1.0}),
            ("between", {CODE: 0.5, COOKING: 0.5}),
            ("cooking", {COOKING: 1.0}),
        )
    }
    assert shift["code"] < shift["between"] < shift["cooking"] < 0
    assert theta.informed(belief, prior) == [CODE]


def test_rounding_and_clamps():
    assert policy.round_words(47) == 45
    assert policy.round_words(73) == 70
    assert policy.round_words(2) == 5
    assert policy.clamp_target(100, 3.0) == 150
    assert policy.clamp_target(100, 0.1) == 50
    assert policy.clamp_target(8, 0.5) == 10


def test_policy_decision_and_propensity():
    ctx = policy.Context(
        200,
        math.log(0.6),
        0.01,
        0.0,
        0.0,
        math.log(0.8),
        0.35,
        0.8,
    )
    decision, propensity = policy.BayesPolicy(random.Random(1)).decide(ctx)
    assert 100 <= decision.target_words <= 300
    assert decision.requested_words > decision.target_words
    assert 0 < propensity <= 1
    fixed, p = policy.FixedRatioPolicy().decide(ctx)
    assert (fixed.target_words, p) == (160, 1.0)


def test_self_report_noise_from_two_sensors():
    rng = random.Random(7)
    truth = [rng.gauss(5, 1) for _ in range(2000)]
    self_reports = [x + rng.gauss(0, 0.3) for x in truth]
    requested = [x + rng.gauss(0, 0.2) for x in truth]
    estimate = length.self_report_variance(self_reports, requested, 0.2**2)
    assert estimate == pytest.approx(0.3**2, rel=0.2)


def test_sensor_fusion_exact_length_wins():
    prior = length.Gaussian3.independent(
        [math.log(100), 0.0, math.log(50)], [1.0, 0.09, 0.16]
    )
    post = length.fuse(
        prior,
        length.Sensors(
            exact_words=80, writing_seconds=2.0, requested_words=100
        ),
        {"self_report": 0.3, "writing_time": 0.5, "requested": 0.25},
    )
    assert post.mean[0] == pytest.approx(math.log(80), abs=1e-3)
    assert post.mean[1] < -0.1  # Wrote 80 for 100 requested.


def test_dawid_skene_recovers_confusions():
    rng = random.Random(3)

    def matrix(diagonal):
        rest = (1 - diagonal) / (len(reaction.OUTCOMES) - 1)
        return {
            y: {o: diagonal if o == y else rest for o in reaction.OUTCOMES}
            for y in reaction.OUTCOMES
        }

    true = {"a": matrix(0.85), "b": matrix(0.7), "c": matrix(0.6)}
    items = []
    for _ in range(2000):
        y = rng.choice(reaction.OUTCOMES)
        items.append(
            reaction.Item(
                [
                    reaction.Observation(
                        vocab.Source.ASSISTANT,
                        {
                            rng.choices(
                                reaction.OUTCOMES,
                                weights=[m[y][o] for o in reaction.OUTCOMES],
                            )[0]: 1.0
                        },
                        {name: 1.0},
                    )
                    for name, m in true.items()
                ]
            )
        )
    start = {vocab.Source.ASSISTANT: {name: matrix(0.75) for name in true}}
    learned, _, _ = reaction.dawid_skene(items, start)
    errors = [
        abs(learned[vocab.Source.ASSISTANT][n][y][o] - m[y][o])
        for n, m in true.items()
        for y in reaction.OUTCOMES
        for o in reaction.OUTCOMES
    ]
    # Mean error: at 2,000 turns even the true labels miss single entries.
    assert sum(errors) / len(errors) < 0.05
    assert max(errors) < 0.1


def test_behaviour_classes_are_found_and_keep_their_names():
    rng = np.random.default_rng(1)
    fast = rng.normal([math.log(90), -0.1], 0.05, size=(60, 2))
    slow = rng.normal([math.log(35), 0.1], 0.05, size=(60, 2))
    fit = family.fit_classes(np.vstack([fast, slow]), family.Families())
    assert len(fit.names) == 2
    assert (
        fit.memberships[:60].argmax(1).tolist()
        == [fit.memberships[0].argmax()] * 60
    )
    previous = family.Families(
        params={
            name: {**family.PRIOR, "speed_mu": m[0], "kappa": m[1]}
            for name, m in zip(fit.names, fit.means)
        }
    )
    refit = family.fit_classes(np.vstack([slow, fast]), previous)
    assert sorted(refit.names) == sorted(fit.names)
    assert family.fit_classes(fast[:5], family.Families()) is None


def test_family_posterior_follows_measured_speed():
    rng = random.Random(11)
    speeds = {"class-1": 35, "class-2": 50, "class-3": 90}
    families = family.Families(
        params={
            f: {**family.PRIOR, "speed_mu": math.log(v), "speed_sd": 0.3}
            for f, v in speeds.items()
        }
    )
    hits = 0
    for _ in range(100):
        probs = families.initial_probs("host")
        for _ in range(8):
            probs = families.update_with_speed(
                probs, rng.gauss(math.log(90), 0.3)
            )
        hits += max(probs, key=probs.get) == "class-3"
    assert hits >= 95


def test_reading_mixture_separates_components():
    rng = random.Random(5)
    data, labels = [], []
    for _ in range(1500):
        full = rng.uniform(2.5, 4.0)
        kind = rng.choices(reading.COMPONENTS, weights=[0.5, 0.3, 0.2])[0]
        lt = {
            "read": rng.gauss(full, 0.3),
            "skim": rng.uniform(0, full),
            "away": rng.gauss(math.log(900), 0.8),
        }[kind]
        data.append((lt, full))
        labels.append(kind)
    fitted, weights = reading.fit_mixture(data, reading.ReadingParams())
    assert fitted.mu_away == pytest.approx(math.log(900), abs=0.3)
    assert weights[2] == pytest.approx(0.2, abs=0.05)
    correct = sum(
        reading.COMPONENTS[
            np.argmax(reading.responsibilities(lt, fr, weights, fitted))
        ]
        == k
        for (lt, fr), k in zip(data, labels)
    )
    assert correct / len(data) > 0.75


def test_preferences_explore_a_little_and_follow_signals():
    rng = random.Random(0)
    draws = [
        prefs.thompson({}, vocab.Dimension.FORMAT, rng) for _ in range(4000)
    ]
    explored = sum(d != vocab.PrefValue.AUTO for d in draws) / len(draws)
    assert 0.01 < explored < 0.1
    signal = prefs.signals(
        {vocab.Dimension.FORMAT: vocab.PrefValue.AUTO},
        [vocab.FORMAT_ISSUES[vocab.FormatIssue.WANT_LIST]],
        0.0,
        1.0,
    )
    counts = {("scope", d, v): c for (d, v), c in signal.items()}
    counts |= {("user", d, v): c for (d, v), c in signal.items()}
    wins = sum(
        prefs.thompson(counts, vocab.Dimension.FORMAT, rng)
        == vocab.PrefValue.BULLETS
        for _ in range(1000)
    )
    assert wins > 800


def test_wrong_tone_spreads_over_the_other_tones():
    signal = prefs.signals(
        {vocab.Dimension.TONE: vocab.PrefValue.WARM},
        [(vocab.Dimension.TONE, None)],
        0.0,
        0.0,
    )
    assert (vocab.Dimension.TONE, vocab.PrefValue.WARM) not in signal
    assert signal[(vocab.Dimension.TONE, vocab.PrefValue.DIRECT)] > 0


def test_buttons_are_distinct_and_positioned():
    shown = bandits.choose({}, random.Random(2))
    assert len({s.button for s in shown}) == bandits.SHOWN
    assert [s.position for s in shown] == list(range(bandits.SHOWN))
    assert all(s.button in vocab.Button for s in shown)


def test_assistant_labels_are_checked():
    label = service._label  # pylint: disable=protected-access
    assert label("  Un   exemple de budget ") == "Un exemple de budget"
    assert label("word " * (service.MAX_LABEL_WORDS + 1)) is None
    assert label("x" * (service.MAX_LABEL_CHARS + 1)) is None
    assert label("") is None and label(3) is None


def test_stable_ids_refuse_distant_matches():
    similarity = np.array([[0.9, 0.1], [0.2, 0.3]])
    assert matching.stable_ids(similarity, [7, 8], minimum=0.5) == [7, 9]


def test_profile_edits_reach_their_scope():
    prior, belief = theta.Prior(), theta.Belief.empty()
    mobile = theta.edit_features(theta.MOBILE, topics.DOMAIN_PREFIX)
    assert theta.edit_features("code@mobile", topics.DOMAIN_PREFIX) == mobile
    theta.edit(belief, mobile, vocab.LengthChoice.SHORT, prior)
    assert math.exp(belief.predict(mobile, prior)[0]) < 0.45
    general = theta.edit_features("*", topics.DOMAIN_PREFIX)
    assert belief.predict(general, prior)[0] > belief.predict(mobile, prior)[0]


def test_word_count_parses_markdown():
    assert (
        text.count_words(
            "## Title\n\n- **bold** item\n- [a link](https://x.y)\n\n"
            "```py\nx = 1\n```"
        )
        == 7
    )
    assert text.count_words("snake_case and *emphasis*") == 3
    assert text.count_words("| a | b |\n|---|---|\n| one | two |") == 4


def test_catalogs_cover_every_identifier():
    english = i18n.catalog(i18n.FALLBACK)
    for language in ("en", "fr"):
        texts = i18n.catalog(language)
        assert set(texts["buttons"]) == set(vocab.Button)
        assert set(texts["lengths"]) == set(vocab.LengthChoice)
        for dimension, values in vocab.DIMENSION_VALUES.items():
            assert set(texts["dimensions"][dimension]["values"]) == set(values)
        dashboard = texts["dashboard"]
        for section, identifiers in (
            ("outcomes", vocab.Outcome),
            ("format_issues", vocab.FormatIssue),
            ("groups", vocab.Arm),
            ("modes", vocab.ReplyMode),
            ("clarify", vocab.Clarify),
            ("reading", reading.COMPONENTS),
        ):
            assert set(dashboard[section]) == set(identifiers)
    french = json.loads((i18n.LOCALES / "fr.json").read_text())["dashboard"]
    for section, texts in english["dashboard"].items():
        if isinstance(texts, dict):  # Every dashboard text is translated.
            assert set(french[section]) == set(texts)
    fragments = english["prompts"]["fragments"]
    for dimension, values in vocab.DIMENSION_VALUES.items():
        assert set(fragments[dimension]) == set(values) - {vocab.PrefValue.AUTO}
    assert set(english["prompts"]["intents"]) == set(vocab.Button)
    assert set(english["prompts"]["arms"]) == set(vocab.Arm) - {
        vocab.Arm.TREATMENT
    }
    assert i18n.catalog("../etc") == english  # Not a language: English.


def test_topic_clustering_is_multilingual_and_stable():
    rng = np.random.default_rng(0)
    centers = topics.unit(rng.normal(size=(3, 384)))
    vectors = topics.unit(
        np.repeat(centers, 40, axis=0) + 0.05 * rng.normal(size=(120, 384))
    )
    found, labels = topics._best_k_means(vectors)  # pylint: disable=protected-access
    assert len(found) == 3
    assert len(set(labels[:40])) == len(set(labels[40:80])) == 1
    ids = matching.stable_ids(found[::-1] @ found[:2].T, [7, 9])
    assert ids == [ids[0], 9, 7] and ids[0] not in (7, 9)
