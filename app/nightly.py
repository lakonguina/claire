"""Nightly job (SPEC 6): refit every noise model, hyperparameter and topic
cluster, then rebuild the derived state by replaying the events."""

import asyncio
import collections
import logging
import math

import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext import asyncio as sa_asyncio

from app import db
from app import engine
from app import models
from app import params as params_lib
from app import topics
from app import vocab
from app.model import family
from app.model import length
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta

Session = sa_asyncio.AsyncSession
log = logging.getLogger(__name__)
DERIVED = models.DERIVED
MIN_WEIGHT = 20.0  # Evidence needed before a parameter leaves its prior.
MIN_ITEMS = 50  # Turns needed before Dawid-Skene runs.
MIN_USERS = 5  # Users needed before the hyperparameters are refit.
MIN_CELLS = 3  # Cells of a request type needed to estimate its gamma.
MIN_VAR = 0.01
HOST_SMOOTHING = 0.5


async def _first_by_turn(
    session: Session, model, *where, latest: bool = False
) -> dict[int, object]:
    """Returns each turn's first (or latest) event of a model."""
    order = model.created_at.desc() if latest else model.created_at
    out = {}
    for event in await session.scalars(
        sa.select(model).where(*where).order_by(order)
    ):
        out.setdefault(event.turn_id, event)
    return out


async def finalized_turns(session: Session) -> list[dict]:
    """Returns every closed turn with its first events and inferred state."""
    rows = (
        await session.execute(
            sa.select(
                models.Turn, models.TurnInference, models.ConversationState
            )
            .join(
                models.TurnInference,
                models.TurnInference.turn_id == models.Turn.id,
            )
            .join(
                models.ConversationState,
                models.ConversationState.conversation_id
                == models.Turn.conversation_id,
            )
        )
    ).all()
    replies = await _first_by_turn(session, models.Reply)
    verdicts = await _first_by_turn(
        session, models.Outcome, models.Outcome.source == vocab.Source.ASSISTANT
    )
    clicks = await _first_by_turn(session, models.Click)
    golds = await _first_by_turn(session, models.GoldLabel, latest=True)
    shown = set(await session.scalars(sa.select(models.ButtonShown.turn_id)))
    out = []
    for turn, inference, state in rows:
        reply, verdict = replies.get(turn.id), verdicts.get(turn.id)
        click, gold = clicks.get(turn.id), golds.get(turn.id)
        out.append(
            {
                "user_id": turn.user_id,
                "cell": engine.style_of(turn)["cell"],
                "needed_words": turn.needed_words,
                "requested_words": turn.requested_words,
                "length_mean": inference.length_mean,
                "reading_probs": inference.reading_probs,
                "family_probs": state.family_probs,
                "kappa_mean": state.kappa_mean,
                "speed_mean": state.speed_mean,
                "duration": (
                    (reply.created_at - turn.created_at).total_seconds()
                    if reply
                    else None
                ),
                "exact": reply.reply_words_exact if reply else None,
                "verdict": verdict.observed if verdict else None,
                "self_report": verdict.last_reply_words if verdict else None,
                "click": (click.button, click.position) if click else None,
                "any_shown": turn.id in shown,
                "gold": gold.outcome if gold else None,
            }
        )
    return out


def refit_reactions(turns: list[dict], p: params_lib.Params) -> None:
    """Refits every annotator's confusion matrix by Dawid-Skene."""
    items = [
        reaction.Item(
            engine.observations(
                t["verdict"],
                t["family_probs"],
                t["click"],
                t["any_shown"],
                [t["reading_probs"][c] for c in reading.COMPONENTS],
            ),
            t["gold"],
        )
        for t in turns
    ]
    if len(items) >= MIN_ITEMS:
        p.confusions, _, _ = reaction.dawid_skene(items, p.confusions)


def _sd(values: list[float]) -> float:
    return math.sqrt(max(stats.mean_var(values)[1], MIN_VAR))


def refit_sensors(turns: list[dict], p: params_lib.Params) -> None:
    """Refits the length sensors' noise and the timing overhead (SPEC 6.2)."""
    writing, requested, self_reports, fallback_requested, overheads = (
        [],
        [],
        [],
        [],
        [],
    )
    for t in turns:
        w = (t["duration"] or 0) - p.overhead
        s2 = math.log(w) + t["speed_mean"] if w > 0 else None
        s3 = (
            math.log(t["requested_words"]) + t["kappa_mean"]
            if t["requested_words"]
            else None
        )
        if t["exact"]:
            truth = math.log(t["exact"])
            if s2 is not None:
                writing.append(s2 - truth)
            if s3 is not None:
                requested.append(s3 - truth)
            if t["duration"]:
                overheads.append(
                    t["duration"] - math.exp(truth - t["speed_mean"])
                )
        elif t["self_report"] and s3 is not None:
            self_reports.append(math.log(t["self_report"]))
            fallback_requested.append(s3)
    if len(writing) >= MIN_WEIGHT:
        p.noise["writing_time"] = _sd(writing)
    if len(requested) >= MIN_WEIGHT:
        p.noise["requested"] = _sd(requested)
    if len(self_reports) >= MIN_WEIGHT:
        p.noise["self_report"] = math.sqrt(
            length.self_report_variance(
                self_reports, fallback_requested, p.noise["requested"] ** 2
            )
        )
    if len(overheads) >= MIN_WEIGHT:
        p.overhead = max(sorted(overheads)[len(overheads) // 2], 0.5)


def _cell(turn: dict) -> tuple[int, str]:
    return turn["user_id"], turn["cell"]


def refit_families(
    turns: list[dict],
    conversations: list[dict],
    p: params_lib.Params,
    theta_means: dict[tuple[int, str], float],
) -> None:
    """Refits the behaviour of every family with enough evidence, and the
    host priors and self-report reliabilities (SPEC 6.3).

    The set of families is open: any family a conversation gave weight to is
    refit once its evidence reaches MIN_WEIGHT; the others keep the average
    of the learned families.
    """
    families = p.families
    seen = sorted(
        {f for t in turns for f in t["family_probs"]}
        | {f for c in conversations for f in c["family_probs"]}
        | set(families.params)
    )
    learned = {}
    for f in seen:
        speeds, speed_w, biases, bias_w = [], [], [], []
        for t in turns:
            weight = t["family_probs"].get(f, 0.0)
            w = (t["duration"] or 0) - p.overhead
            if w > 0:
                speeds.append(t["length_mean"] - math.log(w))
                speed_w.append(weight)
            if _cell(t) in theta_means:
                biases.append(
                    t["length_mean"]
                    - math.log(t["needed_words"])
                    - theta_means[_cell(t)]
                )
                bias_w.append(weight)
        weights = [c["family_probs"].get(f, 0.0) for c in conversations]
        fam = dict(families.of(f))
        if sum(speed_w) >= MIN_WEIGHT:
            mu, var = stats.mean_var(speeds, speed_w)
            fam.update(speed_mu=mu, speed_sd=math.sqrt(max(var, MIN_VAR)))
        if sum(bias_w) >= MIN_WEIGHT:
            b, var = stats.mean_var(biases, bias_w)
            fam.update(b=b, w=max(var, MIN_VAR))
        if sum(weights) >= MIN_WEIGHT:
            fam["kappa"] = stats.mean_var(
                [c["kappa_mean"] for c in conversations], weights
            )[0]
        if max(sum(speed_w), sum(bias_w), sum(weights)) >= MIN_WEIGHT:
            learned[f] = fam  # Only families with evidence are learned.
    if learned:
        # mu0 holds the population level, so the family biases are centred.
        center = sum(fam["b"] for fam in learned.values()) / len(learned)
        for fam in learned.values():
            fam["b"] -= center
    families.params = learned

    by_host, by_report = (
        collections.defaultdict(list),
        collections.defaultdict(list),
    )
    for c in conversations:
        by_host[c["host"]].append(c["family_probs"])
        reported = family.family_of(c["assistant_model"])
        if reported != family.UNKNOWN:
            by_report[reported].append(c["family_probs"])

    def average(probs: list[dict]) -> dict[str, float]:
        sums = [
            sum(x.get(f, 0.0) for x in probs) + HOST_SMOOTHING for f in seen
        ]
        return dict(zip(seen, stats.normalize(sums)))

    families.host_prior = {
        h: average(v) for h, v in by_host.items() if len(v) >= MIN_WEIGHT
    }
    families.reliability = {
        r: average(v) for r, v in by_report.items() if len(v) >= MIN_WEIGHT
    }


def refit_reading(
    turns: list[dict], rho: dict[int, float], p: params_lib.Params
) -> None:
    """Refits the population reading mixture (SPEC 6.6)."""
    data = [
        (
            math.log(t["reading_probs"]["seconds"]),
            t["length_mean"] - rho.get(t["user_id"], 0.0),
        )
        for t in turns
    ]
    if len(data) >= MIN_WEIGHT:
        p.reading, _ = reading.fit_mixture(data, p.reading)


def refit_hyper(
    alphas: dict[int, list[float]],
    thetas: dict[tuple[int, str], list[float]],
    p: params_lib.Params,
) -> None:
    """Refits mu0, sigma_alpha, gamma, sigma_gamma and sigma_eta (SPEC 6.5)."""
    if len(alphas) < MIN_USERS:
        return
    a_stats = {u: stats.grid_mean_var(g) for u, g in alphas.items()}
    hyper = p.hyper
    hyper.mu0, spread = stats.mean_var([m for m, _ in a_stats.values()])
    mean_var = sum(v for _, v in a_stats.values()) / len(a_stats)
    hyper.sigma_alpha = math.sqrt(spread + mean_var)
    residuals = collections.defaultdict(list)
    for (u, cell), grid in thetas.items():
        if u in a_stats:
            residuals[theta.request_type_of(cell)].append(
                stats.grid_mean_var(grid)[0] - a_stats[u][0]
            )
    hyper.gamma = {
        t: stats.mean_var(r)[0]
        for t, r in residuals.items()
        if len(r) >= MIN_CELLS
    }
    if len(hyper.gamma) >= 2:
        hyper.sigma_gamma = _sd(list(hyper.gamma.values()))
    eta = [
        r - hyper.gamma.get(t, 0.0) for t, rs in residuals.items() for r in rs
    ]
    if len(eta) >= MIN_USERS:
        hyper.sigma_eta = _sd(eta)


async def save_params(session: Session, p: params_lib.Params) -> None:
    """Replaces every learned parameter table."""
    for model in models.LEARNED:
        await session.execute(sa.delete(model))
    hyper = {name: getattr(p.hyper, name) for name in params_lib.HYPER_NAMES}
    hyper |= {
        n: getattr(p.reading, f) for n, f in params_lib.READING_NAMES.items()
    }
    hyper |= {
        f"{params_lib.GAMMA_PREFIX}{t}": g for t, g in p.hyper.gamma.items()
    }
    hyper["overhead"] = p.overhead
    session.add_all(
        [
            *(
                models.FamilyParam(model_family=f, **v)
                for f, v in p.families.params.items()
            ),
            *(
                models.SensorNoise(
                    sensor=s, model_family=params_lib.ALL_FAMILIES, sigma=sigma
                )
                for s, sigma in p.noise.items()
            ),
            *(
                models.AnnotatorConfusion(
                    annotator=a,
                    stratum=s,
                    true_outcome=y,
                    observed=o,
                    prob=prob,
                )
                for a, strata in p.confusions.items()
                for s, ys in strata.items()
                for y, obs in ys.items()
                for o, prob in obs.items()
            ),
            *(
                models.SelfReportReliability(
                    reported_family=r, true_family=f, prob=prob
                )
                for r, row in p.families.reliability.items()
                for f, prob in row.items()
            ),
            *(
                models.HostFamilyPrior(host=h, model_family=f, prob=prob)
                for h, row in p.families.host_prior.items()
                for f, prob in row.items()
            ),
            *(models.Hyperparam(name=n, value=v) for n, v in hyper.items()),
        ]
    )
    await session.flush()


async def replay(session: Session, p: params_lib.Params) -> int:
    """Rebuilds every derived table from the events, in their live order.

    Args:
        session: The session, inside the nightly transaction.
        p: The parameters to replay with.

    Returns:
        The number of turns closed.
    """
    for model in DERIVED:
        await session.execute(sa.delete(model))
    session.expunge_all()
    nxt = orm.aliased(models.Turn)
    closes = (
        await session.execute(
            sa.select(models.Turn.id, nxt.created_at).join(
                nxt,
                sa.and_(
                    nxt.conversation_id == models.Turn.conversation_id,
                    nxt.turn_index == models.Turn.turn_index + 1,
                ),
            )
        )
    ).all()
    edits = list(await session.scalars(sa.select(models.ProfileEdit)))
    events = sorted(
        [(at, 0, turn_id, None) for turn_id, at in closes]
        + [(e.created_at, 1, e.id, e) for e in edits],
        key=lambda e: e[:3],
    )
    for at, _, turn_id, edit in events:
        if edit:
            await engine.apply_edit(session, edit, p)
        else:
            await engine.finalize_turn(session, turn_id, at, p)
    return len(closes)


async def run(session: Session) -> int:
    """Refits everything, then replays; returns the number of turns closed."""
    p = await params_lib.load_params(session)
    turns = await finalized_turns(session)
    conversations = (
        (
            await session.execute(
                sa.select(
                    models.Conversation.host,
                    models.Conversation.assistant_model,
                    models.ConversationState.family_probs,
                    models.ConversationState.kappa_mean,
                ).join(
                    models.ConversationState,
                    models.ConversationState.conversation_id
                    == models.Conversation.id,
                )
            )
        )
        .mappings()
        .all()
    )
    thetas = {
        (t.user_id, t.request_type): t.grid
        for t in await session.scalars(sa.select(models.ThetaPosterior))
    }
    alphas = {
        a.user_id: a.grid
        for a in await session.scalars(sa.select(models.AlphaPosterior))
    }
    rho = {
        r.user_id: r.rho_mean
        for r in await session.scalars(sa.select(models.UserReading))
    }
    refit_reactions(turns, p)
    refit_sensors(turns, p)
    refit_families(
        turns,
        conversations,
        p,
        {k: stats.grid_mean_var(g)[0] for k, g in thetas.items()},
    )
    refit_reading(turns, rho, p)
    refit_hyper(alphas, thetas, p)
    await save_params(session, p)
    await topics.recluster(session)
    params_lib.cache.invalidate()
    return await replay(session, p)


async def lock_derived(session: Session) -> None:
    """Locks the derived tables for the rest of the transaction."""
    tables = ", ".join(model.__tablename__ for model in DERIVED)
    await session.execute(sa.text(f"LOCK TABLE {tables} IN EXCLUSIVE MODE"))


async def main() -> None:
    """Runs the nightly job in one transaction."""
    logging.basicConfig(level=logging.INFO)
    await db.migrate()
    async with db.session() as session, session.begin():
        await lock_derived(session)
        log.info("nightly: replayed %d turns", await run(session))
    await db.engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
