import math
import random

from app.model import family, length, policy, reading, theta
from app.model.bandits import BUTTONS, active_styles, choose_buttons
from app.model.reaction import OUTCOMES, Item, Observation, default_confusions, dawid_skene, posterior
from app.model.stats import grid_mean_var, normal_grid
from app.text import count_words, detect_language

CONF = default_confusions(family.FAMILIES)
PROBS = {f: 1 / len(family.FAMILIES) for f in family.FAMILIES}


def reaction_from_verdict(verdict: str) -> list[float]:
    prior = [1 / len(OUTCOMES)] * len(OUTCOMES)
    return posterior(prior, [Observation("assistant", {verdict: 1.0}, PROBS)], CONF)


def posterior_mean_after(verdict: str) -> float:
    hyper = theta.Hyper()
    grid = normal_grid(math.log(0.6), 0.25)
    alpha = theta.alpha_prior(hyper)
    r_hat = math.log(0.6)  # reply exactly at the prior mean ratio
    grid, _ = theta.update(grid, alpha, "fact", r_hat, 0.0, 0.0, 0.0, reaction_from_verdict(verdict), hyper)
    return grid_mean_var(grid)[0]


def test_outcomes_move_the_posterior_the_right_way():
    before = grid_mean_var(normal_grid(math.log(0.6), 0.25))[0]
    too_long, too_short, moved_on = (posterior_mean_after(v) for v in ("too_long", "too_short", "moved_on"))
    assert too_long < moved_on < before
    assert too_short > before


def test_rounding_and_clamps():
    assert policy.round_words(47) == 45
    assert policy.round_words(73) == 70
    assert policy.round_words(2) == 5
    assert policy.clamp_target(100, 3.0) == 150
    assert policy.clamp_target(100, 0.1) == 50
    assert policy.clamp_target(8, 0.5) == 10


def test_policy_decision_and_propensity():
    ctx = policy.Context(200, normal_grid(math.log(0.6), 0.01), 0.0, 0.0, math.log(0.8), 0.35, 0.8)
    decision, propensity = policy.BayesPolicy(random.Random(1)).decide(ctx)
    assert 100 <= decision.target_words <= 300
    assert decision.requested_words > decision.target_words  # compliance below 1 asks for more
    assert 0 < propensity <= 1
    fixed, p = policy.FixedRatioPolicy().decide(ctx)
    assert (fixed.target_words, p) == (160, 1.0)


def test_three_cornered_hat_recovers_variances():
    rng = random.Random(7)
    sigmas = (0.3, 0.5, 0.2)
    truth = [rng.gauss(5, 1) for _ in range(2000)]
    series = [[x + rng.gauss(0, s) for x in truth] for s in sigmas]
    for est, s in zip(length.three_cornered_hat(*series), sigmas):
        assert abs(est - s * s) / (s * s) < 0.2


def test_sensor_fusion_exact_length_wins():
    prior = length.Gaussian3.independent([math.log(100), 0.0, math.log(50)], [1.0, 0.09, 0.16])
    post = length.fuse(prior, length.Sensors(exact_words=80, writing_seconds=2.0, requested_words=100),
                       {"self_report": 0.3, "writing_time": 0.5, "requested": 0.25})
    assert abs(post.mean[0] - math.log(80)) < 1e-3
    assert post.mean[1] < -0.1  # wrote 80 for 100 requested: compliance log(80/100) < 0


def test_dawid_skene_recovers_confusions():
    rng = random.Random(3)
    def matrix(diag):
        return {y: {o: diag if o == y else (1 - diag) / 5 for o in OUTCOMES} for y in OUTCOMES}
    true = {"a": matrix(0.85), "b": matrix(0.7), "c": matrix(0.6)}
    items = []
    for _ in range(2000):
        y = rng.choice(OUTCOMES)
        obs = []
        for name, m in true.items():
            o = rng.choices(OUTCOMES, weights=[m[y][x] for x in OUTCOMES])[0]
            obs.append(Observation("assistant", {o: 1.0}, {name: 1.0}))
        items.append(Item(obs))
    init = {"assistant": {name: matrix(0.75) for name in true}}
    learned, _, _ = dawid_skene(items, init)
    errors = [abs(learned["assistant"][n][y][o] - true[n][y][o]) for n in true for y in OUTCOMES for o in OUTCOMES]
    # Mean absolute error: at 2,000 turns even the true labels give single entries off by more than 0.05.
    assert sum(errors) / len(errors) < 0.05
    assert max(errors) < 0.1


def test_family_posterior_sees_through_wrong_self_reports():
    rng = random.Random(11)
    params = {f: {"speed_mu": mu, "speed_sd": 0.3} for f, mu in
              zip(family.FAMILIES, [math.log(35), math.log(50), math.log(90), math.log(50), math.log(40)])}
    reliability = {r: {f: 0.85 if f == r else 0.0375 for f in family.FAMILIES} for r in family.FAMILIES}
    hits = 0
    for _ in range(100):
        reported = "claude-sonnet" if rng.random() < 0.3 else "claude-haiku"
        probs = family.initial_probs({f: 0.2 for f in family.FAMILIES}, reported, reliability)
        for _ in range(8):
            probs = family.update_with_speed(probs, rng.gauss(math.log(90), 0.3), params)
        hits += max(probs, key=probs.get) == "claude-haiku"
    assert hits >= 95


def test_reading_mixture_separates_components():
    rng = random.Random(5)
    params = reading.ReadingParams(s_r=0.3, mu_away=math.log(900), s_away=0.8)
    data, labels = [], []
    for _ in range(1500):
        full = rng.uniform(2.5, 4.0)  # log seconds of a full read
        kind = rng.choices(["read", "skim", "away"], weights=[0.5, 0.3, 0.2])[0]
        lt = {"read": rng.gauss(full, 0.3), "skim": rng.uniform(0, full), "away": rng.gauss(math.log(900), 0.8)}[kind]
        data.append((lt, full))
        labels.append(kind)
    fitted, weights = reading.fit_mixture(data, reading.ReadingParams())
    assert abs(fitted.mu_away - math.log(900)) < 0.3
    assert abs(weights[2] - 0.2) < 0.05
    correct = sum(reading.COMPONENTS[max(range(3), key=r.__getitem__)] == label for r, label in
                  zip((reading.responsibilities(lt, fr, weights, fitted) for lt, fr in data), labels))
    assert correct / len(data) > 0.75


def test_buttons_and_styles():
    shown = choose_buttons({}, {"rephrase_request"}, random.Random(2))
    assert len({s.button for s in shown}) == 3 and [s.position for s in shown] == [0, 1, 2]
    assert all(s.button in BUTTONS and s.button != "rephrase_request" for s in shown)
    assert active_styles({"style:simple": (3.0, 1.0)}) == ["simple"]
    assert active_styles({"style:simple": (2.0, 1.0)}) == []  # one signal only


def test_text_helpers():
    assert count_words("## Title\n\n- **bold** item\n- [a link](https://x.y)\n\n```py\nx = 1\n```") == 7
    assert detect_language("comment configurer le serveur pour la prod") == "fr"
    assert detect_language("how to deploy the server") == "en"
    assert detect_language("Kubernetes") == "unknown"
