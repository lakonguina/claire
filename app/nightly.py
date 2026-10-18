"""Nightly job (SPEC 6): refit every noise model, hyperparameter and topic
cluster, then rebuild the derived state by replaying the events."""

import asyncio
import collections
import logging
import math

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext import asyncio as sa_asyncio

from app import db
from app import engine
from app import models
from app import params as params_lib
from app import questions
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
TURN_COLUMNS = [
    "user_id",
    "conversation_id",
    "theta_mean",
    "needed_words",
    "requested_words",
    "length_mean",
    "reading_probs",
    "family_probs",
    "kappa_mean",
    "speed_mean",
    "duration",
    "exact",
    "verdict",
    "self_report",
    "click",
    "any_shown",
    "gold",
]
NUMERIC = [
    "needed_words",
    "requested_words",
    "length_mean",
    "theta_mean",
    "kappa_mean",
    "speed_mean",
    "duration",
    "exact",
    "self_report",
]
MIN_WEIGHT = 20.0  # Evidence needed before a parameter leaves its prior.
MIN_ITEMS = 50  # Turns needed before Dawid-Skene runs.
MIN_USERS = 5  # Users needed before the hyperparameters are refit.
MIN_VAR = 0.01
HOST_SMOOTHING = 0.5


async def finalized_turns(session: Session) -> pd.DataFrame:
    """Returns every closed turn with its first events and inferred state."""
    rows = (
        await session.execute(
            engine.with_events(
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
        )
    ).all()
    out = []
    for turn, inference, state in rows:
        reply = engine.first(turn.replies)
        verdict = engine.first(turn.outcomes, source=vocab.Source.ASSISTANT)
        click = engine.last(turn.clicks)
        out.append(
            {
                "user_id": turn.user_id,
                "conversation_id": turn.conversation_id,
                "theta_mean": inference.theta_mean,
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
                "any_shown": bool(turn.buttons_shown),
                "gold": turn.gold_labels[-1].outcome
                if turn.gold_labels
                else None,
            }
        )
    frame = pd.DataFrame(out, columns=TURN_COLUMNS)
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric)
    return frame


def refit_reactions(turns: pd.DataFrame, p: params_lib.Params) -> None:
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
        for t in turns.to_dict("records")
    ]
    if len(items) >= MIN_ITEMS:
        p.confusions, _, _ = reaction.dawid_skene(items, p.confusions)


def _sd(values) -> float:
    return math.sqrt(max(float(np.var(values)), MIN_VAR))


def _log_speed(turns: pd.DataFrame, p: params_lib.Params) -> pd.Series:
    """Each turn's measured log writing speed (NaN without a timing)."""
    writing = turns.duration - p.overhead
    return turns.length_mean - np.log(writing.where(writing > 0))


def refit_sensors(turns: pd.DataFrame, p: params_lib.Params) -> None:
    """Refits the length sensors' noise and the timing overhead (SPEC 6.2)."""
    truth = np.log(turns.exact)
    exact = turns.exact.notna()
    s2 = truth - _log_speed(turns, p) + turns.speed_mean
    s3 = np.log(turns.requested_words) + turns.kappa_mean
    residuals = {
        "writing_time": (s2 - truth)[exact].dropna(),
        "requested": (s3 - truth)[exact].dropna(),
    }
    for sensor, values in residuals.items():
        if len(values) >= MIN_WEIGHT:
            p.noise[sensor] = _sd(values)
    fallback = ~exact & turns.self_report.notna() & s3.notna()
    if fallback.sum() >= MIN_WEIGHT:
        p.noise["self_report"] = math.sqrt(
            length.self_report_variance(
                np.log(turns.self_report[fallback]),
                s3[fallback],
                p.noise["requested"] ** 2,
            )
        )
    overheads = (turns.duration - np.exp(truth - turns.speed_mean)).dropna()
    if len(overheads) >= MIN_WEIGHT:
        p.overhead = max(float(overheads.median()), 0.5)


def refit_families(
    turns: pd.DataFrame,
    conversations: pd.DataFrame,
    p: params_lib.Params,
) -> None:
    """Refits the behaviour classes and the host priors (SPEC 6.3).

    Each conversation with measured writing speeds is placed by its mean log
    speed and its compliance; the classes found keep their names across
    nights, and each gets the estimation bias of its conversations.
    """
    speed = _log_speed(turns, p).groupby(turns.conversation_id).mean().dropna()
    measured = conversations.set_index("id").loc[speed.index]
    fit = family.fit_classes(
        np.column_stack([speed.to_numpy(), measured.kappa_mean.to_numpy()]),
        p.families,
    )
    if fit is None:
        p.families = family.Families()
        return
    membership = pd.DataFrame(
        fit.memberships, index=speed.index, columns=fit.names
    )
    # The preference predicted before each turn, so no turn explains itself.
    bias = turns.length_mean - np.log(turns.needed_words) - turns.theta_mean
    weights = membership.reindex(turns.conversation_id.to_numpy()).fillna(0.0)
    params = {}
    for k, name in enumerate(fit.names):
        w = weights[name].to_numpy()
        known = bias.notna().to_numpy() & (w > 0)
        b, var = family.PRIOR["b"], family.PRIOR["w"]
        if w[known].sum() >= MIN_WEIGHT:
            b, var = stats.mean_var(bias[known], w[known])
        params[name] = {
            "b": b,
            "w": max(var, MIN_VAR),
            "kappa": float(fit.means[k, 1]),
            "speed_mu": float(fit.means[k, 0]),
            "speed_sd": math.sqrt(fit.variances[k, 0]),
        }
    # The prior's bias holds the population level: class biases are centred.
    center = np.mean([c["b"] for c in params.values()])
    for c in params.values():
        c["b"] -= center
    by_host = membership.groupby(measured.host)
    host_prior = {
        host: dict(zip(fit.names, stats.normalize(rows.sum() + HOST_SMOOTHING)))
        for host, rows in by_host
        if len(rows) >= MIN_WEIGHT
    }
    p.families = family.Families(params, host_prior)


def refit_reading(
    turns: pd.DataFrame, rho: dict[int, float], p: params_lib.Params
) -> None:
    """Refits the population reading mixture (SPEC 6.6)."""
    seconds = turns.reading_probs.map(lambda r: r["seconds"])
    full_read = turns.length_mean - turns.user_id.map(rho).fillna(0.0)
    if len(turns) >= MIN_WEIGHT:
        p.reading, _ = reading.fit_mixture(
            list(zip(np.log(seconds), full_read)), p.reading
        )


def refit_prior(beliefs: list[theta.Belief], p: params_lib.Params) -> None:
    """Refits the population prior of the length weights (SPEC 6.5).

    Empirical Bayes per feature: the mean of the users' posterior means, and
    their spread plus the mean posterior variance (the part of each user the
    data has not pinned down yet).
    """
    by_feature = collections.defaultdict(list)
    for belief in beliefs:
        for i, name in enumerate(belief.names):
            by_feature[name].append((belief.mean[i], belief.cov[i, i]))
    for name, values in by_feature.items():
        if len(values) >= MIN_USERS:
            means, variances = np.array(values).T
            p.prior.mean[name] = float(means.mean())
            p.prior.var[name] = max(
                float(means.var() + variances.mean()), MIN_VAR
            )
    domains = [
        v for n, v in p.prior.var.items() if n.startswith(topics.DOMAIN_PREFIX)
    ]
    if len(domains) >= 2:
        p.prior.default_var = float(np.mean(domains))


async def save_params(session: Session, p: params_lib.Params) -> None:
    """Replaces every learned parameter table."""
    for model in models.LEARNED:
        await session.execute(sa.delete(model))
    hyper = {
        n: getattr(p.reading, f) for n, f in params_lib.READING_NAMES.items()
    }
    hyper |= {f"{params_lib.PRIOR_MEAN}{f}": m for f, m in p.prior.mean.items()}
    hyper |= {f"{params_lib.PRIOR_VAR}{f}": v for f, v in p.prior.var.items()}
    hyper["tau"] = p.prior.tau
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
    conversations = pd.DataFrame(
        (
            await session.execute(
                sa.select(
                    models.Conversation.id,
                    models.Conversation.host,
                    models.ConversationState.kappa_mean,
                ).join(
                    models.ConversationState,
                    models.ConversationState.conversation_id
                    == models.Conversation.id,
                )
            )
        ).all(),
        columns=["id", "host", "kappa_mean"],
    )
    beliefs = [
        engine.belief_of(row)
        for row in await session.scalars(sa.select(models.LengthPosterior))
    ]
    rho = {
        r.user_id: r.rho_mean
        for r in await session.scalars(sa.select(models.UserReading))
    }
    refit_reactions(turns, p)
    refit_sensors(turns, p)
    refit_families(turns, conversations, p)
    refit_reading(turns, rho, p)
    refit_prior(beliefs, p)
    await save_params(session, p)
    await topics.recluster(session)
    await questions.rebuild(session)
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
