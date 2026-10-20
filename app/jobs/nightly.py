"""Nightly job: delete text past its retention, refit every noise model,
hyperparameter and topic cluster, then rebuild the derived state by
replaying the events."""

import asyncio
import collections
import datetime
import logging
import math

import numpy as np
import pandas as pd
from sklearn import linear_model
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config
from app.core import db
from app.core import vocab
from app.model import family
from app.model import length
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta
from app import models
from app.services import engine
from app.services import metrics
from app.services import params as params_lib
from app.services import topics

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
    "gold",
    "distance",
    "ambiguity",
    "reaction_probs",
    "satisfaction",
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
    "distance",
    "ambiguity",
]
MIN_WEIGHT = 20.0  # Evidence needed before a parameter leaves its prior.
MIN_ITEMS = 50  # Turns needed before Dawid-Skene runs.
MIN_USERS = 5  # Users needed before the hyperparameters are refit.
MIN_VAR = 0.01
HOST_SMOOTHING = 0.5
NONCOMPLIANT_RATE = 0.2  # Below it, a class stops being asked.
REQUESTED_FIELDS = {
    vocab.Requested.USER_MESSAGE: "user_message_when_asked",
    vocab.Requested.CHOICES: "choices_when_asked",
}
# A chat ends after the gap that this share of the gaps between its calls
# stays under.
CHAT_END_QUANTILE = 0.95
MIN_CHAT_END_DAYS = 1.0


async def late_evidence(session: Session) -> dict[int, dict]:
    """Returns, per turn, the evidence known only after it closed.

    The ground-truth answer is about the turn its survey's call closed.
    """
    out = collections.defaultdict(dict)
    answers = await session.execute(
        sa.select(
            models.Turn.follows, models.WidgetEvent.value["value"].as_string()
        )
        .join(models.Turn, models.Turn.id == models.WidgetEvent.turn_id)
        .where(
            models.WidgetEvent.kind == vocab.WidgetEvent.SATISFACTION,
            models.Turn.follows.is_not(None),
        )
        .order_by(models.WidgetEvent.id)
    )
    for turn_id, answer in answers:
        out[turn_id]["satisfaction"] = answer
    return out


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
    late = await late_evidence(session)
    out = []
    for turn, inference, state in rows:
        reply = engine.first(turn.replies)
        verdict = engine.first(turn.outcomes, source=vocab.Source.ASSISTANT)
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
                "gold": turn.gold_labels[-1].outcome
                if turn.gold_labels
                else None,
                "distance": inference.next_topic_distance,
                "ambiguity": turn.ambiguity,
                "reaction_probs": inference.reaction_probs,
                "satisfaction": late.get(turn.id, {}).get("satisfaction"),
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
                [t["reading_probs"][c] for c in reading.COMPONENTS],
                (
                    None
                    if pd.isna(t["distance"])
                    else t["distance"] < p.rephrase_distance
                ),
                t["satisfaction"],
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
    """Refits the length sensors' noise and the timing overhead."""
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
    """Refits the behaviour classes and the host priors.

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
    """Refits the population reading mixture."""
    seconds = turns.reading_probs.map(lambda r: r["seconds"])
    full_read = turns.length_mean - turns.user_id.map(rho).fillna(0.0)
    if len(turns) >= MIN_WEIGHT:
        p.reading, _ = reading.fit_mixture(
            list(zip(np.log(seconds), full_read)), p.reading
        )


def refit_prior(beliefs: list[theta.Belief], p: params_lib.Params) -> None:
    """Refits the population prior of the length weights.

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
    hyper |= {name: getattr(p, name) for name in params_lib.SCALARS}
    hyper |= {f"{params_lib.NONCOMPLIANT}{k}": 1.0 for k in p.noncompliant}
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


def refit_rephrase(turns: pd.DataFrame, p: params_lib.Params) -> None:
    """Learns under which topic distance the next request asks the same.

    The threshold that best tells the turns the assistant judged
    misunderstood (F1), among the consecutive requests' distances.
    """
    known = turns[turns.distance.notna() & turns.verdict.notna()]
    missed = (known.verdict == vocab.Outcome.MISUNDERSTOOD).to_numpy()
    if missed.sum() < MIN_WEIGHT:
        return
    distances = known.distance.to_numpy()
    best, best_f1 = p.rephrase_distance, -1.0
    for threshold in np.quantile(distances, np.linspace(0.01, 0.5, 50)):
        said = distances < threshold
        hits = (said & missed).sum()
        f1 = 2 * hits / (said.sum() + missed.sum())
        if f1 > best_f1:
            best, best_f1 = float(threshold), f1
    p.rephrase_distance = best


def refit_ambiguity(turns: pd.DataFrame, p: params_lib.Params) -> None:
    """Platt-scales the assistant's ambiguity on what followed.

    The target is whether the reply missed: P(misunderstood) or a rephrased
    next request, as soft labels.
    """
    sent = turns[turns.ambiguity.notna()]
    if len(sent) < MIN_WEIGHT:
        return
    missed = np.array(
        [
            max(
                r[vocab.Outcome.MISUNDERSTOOD],
                float(not pd.isna(d) and d < p.rephrase_distance),
            )
            for r, d in zip(sent.reaction_probs, sent.distance)
        ]
    )
    clipped = np.clip(sent.ambiguity.to_numpy(), 1e-4, 1 - 1e-4)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = linear_model.LogisticRegression()
    model.fit(
        np.vstack([logits, logits]),
        np.r_[np.ones(len(sent)), np.zeros(len(sent))],
        sample_weight=np.r_[missed, 1 - missed] + 1e-6,
    )
    p.ambiguity_a = float(model.coef_[0, 0])
    p.ambiguity_b = float(model.intercept_[0])


async def refit_compliance(session: Session, p: params_lib.Params) -> None:
    """Finds what each behaviour class does not do when the instruction asks.

    Its instruction then stops asking for it.
    """
    every = pd.DataFrame(await metrics.turns(session, p))
    p.noncompliant = set()
    if every.empty:
        return
    for behaviour, group in every.groupby("family"):
        for requested, name in REQUESTED_FIELDS.items():
            sent, due = metrics.FIELDS[name]
            if group[due].sum() >= MIN_WEIGHT:
                if group[sent].sum() / group[due].sum() < NONCOMPLIANT_RATE:
                    p.noncompliant.add(f"{behaviour}:{requested}")


async def refit_chat_end(session: Session, p: params_lib.Params) -> None:
    """Learns after how many days without a call a chat is over.

    The gap between consecutive calls of a chat that almost none exceeds:
    past it, the chat is almost never reopened.
    """
    t = models.Turn
    gap = t.created_at - sa.func.lag(t.created_at).over(
        partition_by=t.conversation_id, order_by=t.turn_index
    )
    gaps = sa.select(sa.func.extract("epoch", gap).label("seconds")).subquery()
    seconds = list(
        await session.scalars(
            sa.select(gaps.c.seconds).where(gaps.c.seconds.is_not(None))
        )
    )
    if len(seconds) >= MIN_WEIGHT:
        days = float(np.quantile(seconds, CHAT_END_QUANTILE)) / 86_400
        p.chat_end_days = max(days, MIN_CHAT_END_DAYS)


async def purge(session: Session) -> int:
    """Deletes the stored texts older than the retention; returns how many.

    Word counts stay: they are not text. Each purge is recorded.
    """
    before = await session.scalar(sa.select(sa.func.now())) - (
        datetime.timedelta(days=config.settings.retention_days)
    )
    replies = await session.execute(
        sa.update(models.Reply)
        .where(models.Reply.created_at < before, models.Reply.text.is_not(None))
        .values(text=None)
    )
    messages = await session.execute(
        sa.update(models.Message)
        .where(
            models.Message.created_at < before, models.Message.text.is_not(None)
        )
        .values(text=None)
    )
    rows = replies.rowcount + messages.rowcount
    session.add(models.Purge(before=before, rows=rows))
    await session.flush()
    return rows


async def replay(session: Session, p: params_lib.Params) -> int:
    """Rebuilds every derived table from the events, in their live order.

    A turn closes at the first call that follows it.

    Args:
        session: The session, inside the nightly transaction.
        p: The parameters to replay with.

    Returns:
        The number of turns closed.
    """
    for model in DERIVED:
        await session.execute(sa.delete(model))
    late = await late_evidence(session)
    session.expunge_all()
    first = (
        sa.select(
            models.Turn.follows,
            models.Turn.created_at,
            models.TopicEmbedding.embedding,
            sa.func.row_number()
            .over(
                partition_by=models.Turn.follows,
                order_by=(models.Turn.created_at, models.Turn.id),
            )
            .label("rank"),
        )
        .join(
            models.TopicEmbedding,
            models.TopicEmbedding.turn_id == models.Turn.id,
        )
        .where(models.Turn.follows.is_not(None))
        .subquery()
    )
    closes = (
        await session.execute(
            sa.select(first.c.follows, first.c.created_at, first.c.embedding)
            .where(first.c.rank == 1)
            .order_by(first.c.created_at, first.c.follows)
        )
    ).all()
    for turn_id, at, embedding in closes:
        await engine.finalize_turn(
            session, turn_id, at, p, embedding, late.get(turn_id)
        )
    return len(closes)


async def run(session: Session) -> int:
    """Purges, refits everything, then replays; returns the turns closed."""
    log.info("nightly: purged %d texts", await purge(session))
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
    refit_rephrase(turns, p)
    refit_ambiguity(turns, p)
    await refit_compliance(session, p)
    await refit_chat_end(session, p)
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
