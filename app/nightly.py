"""Nightly job (SPEC 6): refit every noise model, hyperparameter and topic
cluster, then rebuild the derived state by replaying the events."""

import asyncio
import collections
import logging
import math

import numpy as np
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
                "conversation_id": turn.conversation_id,
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
    """Refits the behaviour classes and the host priors (SPEC 6.3).

    Each conversation with measured writing speeds is placed by its mean log
    speed and its compliance; the classes found keep their names across
    nights, and each gets the estimation bias of its conversations.
    """
    speeds = collections.defaultdict(list)
    for t in turns:
        w = (t["duration"] or 0) - p.overhead
        if w > 0:
            speeds[t["conversation_id"]].append(t["length_mean"] - math.log(w))
    measured = [c for c in conversations if c["id"] in speeds]
    features = np.array(
        [
            [stats.mean_var(speeds[c["id"]])[0], c["kappa_mean"]]
            for c in measured
        ]
    ).reshape(len(measured), 2)
    fit = family.fit_classes(features, p.families)
    if fit is None:
        p.families = family.Families()
        return
    membership = {
        c["id"]: dict(zip(fit.names, row))
        for c, row in zip(measured, fit.memberships)
    }
    params = {}
    for k, name in enumerate(fit.names):
        biases, weights = [], []
        for t in turns:
            weight = membership.get(t["conversation_id"], {}).get(name, 0.0)
            if weight and _cell(t) in theta_means:
                biases.append(
                    t["length_mean"]
                    - math.log(t["needed_words"])
                    - theta_means[_cell(t)]
                )
                weights.append(weight)
        b, w = family.PRIOR["b"], family.PRIOR["w"]
        if sum(weights) >= MIN_WEIGHT:
            b, w = stats.mean_var(biases, weights)
        params[name] = {
            "b": b,
            "w": max(w, MIN_VAR),
            "kappa": float(fit.means[k, 1]),
            "speed_mu": float(fit.means[k, 0]),
            "speed_sd": math.sqrt(fit.variances[k, 0]),
        }
    # mu0 holds the population level, so the class biases are centred.
    center = sum(c["b"] for c in params.values()) / len(params)
    for c in params.values():
        c["b"] -= center

    by_host = collections.defaultdict(list)
    for c in measured:
        by_host[c["host"]].append(membership[c["id"]])
    host_prior = {}
    for host, rows in by_host.items():
        if len(rows) >= MIN_WEIGHT:
            sums = [sum(r[n] for r in rows) + HOST_SMOOTHING for n in fit.names]
            host_prior[host] = dict(zip(fit.names, stats.normalize(sums)))
    p.families = family.Families(params, host_prior)


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
                    models.Conversation.id,
                    models.Conversation.host,
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
