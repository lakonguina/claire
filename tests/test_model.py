"""Model behaviour and recovery on synthetic data (SPEC 12)."""

import math
import random

import numpy as np
import pytest

from app import i18n
from app import text
from app import topics
from app import vocab
from app.model import bandits
from app.model import family
from app.model import length
from app.model import policy
from app.model import prefs
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta

CONF = reaction.default_confusions()
FAMILY_PROBS = {"claude-opus": 0.5, "mistral-large": 0.5}


def outcome_probs(verdict: str) -> list[float]:
    prior = [1 / len(reaction.OUTCOMES)] * len(reaction.OUTCOMES)
    obs = reaction.Observation(
        vocab.Source.ASSISTANT, {verdict: 1.0}, FAMILY_PROBS
    )
    return reaction.posterior(prior, [obs], CONF)


def posterior_mean_after(verdict: str) -> float:
    hyper = theta.Hyper()
    grid = stats.normal_grid(math.log(0.6), 0.25)
    grid, _ = theta.update(
        grid,
        theta.alpha_prior(hyper),
        "fact",
        math.log(0.6),
        0.0,
        (0.0, 0.0),
        outcome_probs(verdict),
        hyper,
    )
    return stats.grid_mean_var(grid)[0]


def test_outcomes_move_the_posterior_the_right_way():
    before = stats.grid_mean_var(stats.normal_grid(math.log(0.6), 0.25))[0]
    too_long, too_short, moved_on = map(
        posterior_mean_after, ("too_long", "too_short", "moved_on")
    )
    assert too_long < moved_on < before < too_short


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
        stats.normal_grid(math.log(0.6), 0.01),
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


def test_family_names_are_normalised_without_a_vendor_list():
    cases = {
        "mistral-large-2411": "mistral-large",
        "Mistral Large 2": "mistral-large",
        "gemini-2.5-pro": "gemini-pro",
        "claude-opus-5-5": "claude-opus",
        "llama-3.1-70b-instruct": "llama-instruct",
        None: family.UNKNOWN,
        "4o": family.UNKNOWN,
    }
    for name, expected in cases.items():
        assert family.family_of(name) == expected


def test_new_families_start_from_the_learned_average():
    families = family.Families(
        params={
            "a": {**family.POPULATION, "speed_mu": math.log(40)},
            "b": {**family.POPULATION, "speed_mu": math.log(60)},
        }
    )
    unseen = families.of("mistral-large")
    assert unseen["speed_mu"] == pytest.approx(
        (math.log(40) + math.log(60)) / 2
    )
    probs = families.initial_probs("Le Chat", "mistral-large")
    assert set(probs) == {"a", "b", "mistral-large", family.UNKNOWN}
    assert max(probs, key=probs.get) == "mistral-large"


def test_family_posterior_sees_through_wrong_self_reports():
    rng = random.Random(11)
    speeds = {"slow": 35, "medium": 50, "fast": 90}
    families = family.Families(
        params={
            f: {**family.POPULATION, "speed_mu": math.log(v), "speed_sd": 0.3}
            for f, v in speeds.items()
        }
    )
    hits = 0
    for _ in range(100):
        reported = "medium" if rng.random() < 0.3 else "fast"
        probs = families.initial_probs("host", reported)
        for _ in range(8):
            probs = families.update_with_speed(
                probs, rng.gauss(math.log(90), 0.3)
            )
        hits += max(probs, key=probs.get) == "fast"
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
    counts = {("cell", d, v): c for (d, v), c in signal.items()}
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


def test_length_cells_and_profile_edits():
    assert theta.cell_of("code", vocab.Platform.MOBILE) == "code@mobile"
    assert theta.cell_of("code", vocab.Platform.WEB) == "code"
    assert theta.request_type_of("code@mobile") == "code"
    grid = theta.edit(
        stats.normal_grid(math.log(1.0), 0.25), vocab.LengthChoice.SHORT
    )
    assert math.exp(stats.grid_mean_var(grid)[0]) < 0.5


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
        assert set(texts["request_types"]) == set(vocab.RequestType)
        assert set(texts["lengths"]) == set(vocab.LengthChoice)
        for dimension, values in vocab.DIMENSION_VALUES.items():
            assert set(texts["dimensions"][dimension]["values"]) == set(values)
    fragments = english["prompts"]["fragments"]
    for dimension, values in vocab.DIMENSION_VALUES.items():
        assert set(fragments[dimension]) == set(values) - {vocab.PrefValue.AUTO}
    assert set(english["prompts"]["arms"]) == set(vocab.Arm) - {
        vocab.Arm.TREATMENT
    }
    assert i18n.catalog("../etc") == english  # Not a language: English.


def test_topic_clustering_is_multilingual_and_stable():
    rng = np.random.default_rng(0)
    centers = topics._unit(rng.normal(size=(3, 384)))  # pylint: disable=protected-access
    vectors = topics._unit(  # pylint: disable=protected-access
        np.repeat(centers, 40, axis=0) + 0.05 * rng.normal(size=(120, 384))
    )
    found, labels = topics._best_k_means(vectors)  # pylint: disable=protected-access
    assert len(found) == 3
    assert len(set(labels[:40])) == len(set(labels[40:80])) == 1
    ids = topics.stable_ids(found[::-1], {7: found[0], 9: found[1]})
    assert ids == [ids[0], 9, 7] and ids[0] not in (7, 9)
