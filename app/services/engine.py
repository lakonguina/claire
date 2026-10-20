"""Derived-state updates: length, assistant behaviour, reaction, form.

The live service and the nightly replay both go through finalize_turn, which
reads only events recorded up to the moment the turn closed, so a replay
reproduces the live state; the replay adds the evidence that comes later
(the ground-truth answer), the nightly refit's part.
"""

import collections
from collections.abc import Iterable
import dataclasses
import datetime
import math

import numpy as np
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import vocab
from app.model import length
from app.model import prefs
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta
from app import models
from app.services import params as params_lib
from app.services import topics

Session = sa_asyncio.AsyncSession
KAPPA_PRIOR_VAR = 0.3**2
LENGTH_PRIOR_VAR = 1.0


async def conversation_state(
    session: Session, conversation_id: int, p: params_lib.Params
) -> models.ConversationState:
    """Returns the conversation's assistant state, or a new one at its prior."""
    state = await session.get(models.ConversationState, conversation_id)
    if state:
        return state
    conv = await session.get(models.Conversation, conversation_id)
    probs = p.families.initial_probs(conv.host)
    speed_mean, speed_var = p.families.speed_prior(probs)
    return models.ConversationState(
        conversation_id=conversation_id,
        family_probs=probs,
        kappa_mean=p.families.mix(probs, "kappa"),
        kappa_var=KAPPA_PRIOR_VAR,
        speed_mean=speed_mean,
        speed_var=speed_var,
    )


async def user_reading(session: Session, user_id: int) -> models.UserReading:
    """Returns the user's reading state, or a new one at its prior."""
    return await session.get(models.UserReading, user_id) or models.UserReading(
        user_id=user_id,
        rho_mean=params_lib.RHO_PRIOR[0],
        rho_var=params_lib.RHO_PRIOR[1],
        mixture=list(params_lib.READING_PRIOR),
        reaction_prior=list(params_lib.REACTION_PRIOR),
    )


def belief_of(row: models.LengthPosterior | None) -> theta.Belief:
    """Returns the belief a stored length posterior holds."""
    if row is None:
        return theta.Belief.empty()
    size = len(row.features)
    return theta.Belief(
        list(row.features), row.mean, np.reshape(row.cov, (size, size))
    )


async def length_belief(session: Session, user_id: int) -> theta.Belief:
    """Returns the user's belief about their preferred length."""
    return belief_of(await session.get(models.LengthPosterior, user_id))


async def save_belief(
    session: Session, user_id: int, belief: theta.Belief
) -> None:
    await session.merge(
        models.LengthPosterior(
            user_id=user_id,
            features=belief.names,
            mean=belief.mean,
            cov=belief.cov.ravel(),
        )
    )


async def request_features(
    session: Session, turn: models.Turn, embedding
) -> dict[str, float]:
    """Returns a turn's length features from its topic and platform."""
    domains = (
        await topics.memberships(session, embedding)
        if embedding is not None
        else {}
    )
    return theta.features(domains, style_of(turn)["mobile"])


async def pref_counts(
    session: Session, user_id: int, scopes: Iterable[str]
) -> dict[str, prefs.Counts]:
    """Returns, for each scope, its counts, the user's and the population's.

    Args:
        session: The session.
        user_id: The user.
        scopes: The scopes needed (topic domains).

    Returns:
        Scope -> counts keyed by (level, dimension, value).
    """
    scopes = list(scopes)
    pc = models.PrefCount
    rows = await session.scalars(
        sa.select(pc).where(
            sa.or_(
                sa.and_(
                    pc.user_id == user_id,
                    pc.scope.in_([*scopes, prefs.USER_SCOPE]),
                ),
                sa.and_(
                    pc.user_id == prefs.POPULATION,
                    pc.scope == prefs.USER_SCOPE,
                ),
            )
        )
    )
    out = {s: {} for s in scopes}
    for r in rows:
        key = (r.dimension, r.value)
        for scope, counts in out.items():
            if r.user_id == prefs.POPULATION:
                counts[("population", *key)] = r.count
                continue
            if r.scope == scope:
                counts[("scope", *key)] = r.count
            if r.scope == prefs.USER_SCOPE:
                counts[("user", *key)] = r.count
    return out


async def add_prefs(
    session: Session,
    user_id: int,
    scope: str,
    counts: dict[tuple[str, str], float],
) -> None:
    """Adds counts to a scope, its user level and the population."""
    levels = {
        (user_id, scope),
        (user_id, prefs.USER_SCOPE),
        (prefs.POPULATION, prefs.USER_SCOPE),
    }
    rows = [
        {
            "user_id": u,
            "scope": s,
            "dimension": dimension,
            "value": value,
            "count": count,
        }
        for (dimension, value), count in counts.items()
        for u, s in levels
    ]
    if not rows:
        return
    insert = postgresql.insert(models.PrefCount)
    await session.execute(
        insert.on_conflict_do_update(
            index_elements=["user_id", "scope", "dimension", "value"],
            set_={"count": models.PrefCount.count + insert.excluded.count},
        ),
        rows,
    )


def observations(
    verdict: str | None,
    family_probs: dict,
    reading_probs: list[float] | None,
    rephrased: bool | None = None,
    satisfaction: str | None = None,
) -> list[reaction.Observation]:
    """Returns the annotator observations of one turn.

    Args:
        verdict: The assistant verdict, if any.
        family_probs: The conversation's model-family probabilities.
        reading_probs: P(read), P(skim), P(away), if known.
        rephrased: Whether the next request asked the same thing, if known.
        satisfaction: The person's answer to the ground-truth question. It
            arrives after the turn closed: the nightly replay adds it, the
            live path never has it.

    Returns:
        The observations.
    """
    obs = []
    if rephrased is not None:
        token = reaction.REPHRASED if rephrased else reaction.KEPT
        obs.append(reaction.Observation(vocab.Source.REPHRASE, {token: 1.0}))
    if satisfaction:
        obs.append(
            reaction.Observation(vocab.Source.SATISFACTION, {satisfaction: 1.0})
        )
    if verdict and verdict != vocab.Outcome.NO_SIGNAL:
        obs.append(
            reaction.Observation(
                vocab.Source.ASSISTANT, {verdict: 1.0}, dict(family_probs)
            )
        )
    if reading_probs is not None:
        read, skim, away = reading_probs
        obs.append(
            reaction.Observation(
                vocab.Source.READING,
                {"read": read, "skim": skim, reaction.NO_INFO: away},
            )
        )
    return obs


TURN_EVENTS = (
    models.Turn.replies,
    models.Turn.outcomes,
    models.Turn.widget_events,
    models.Turn.gold_labels,
    models.Turn.contexts,
    models.Turn.embeddings,
    models.Turn.shadow_decisions,
)


def with_events(statement: sa.Select) -> sa.Select:
    """Loads every event relationship of the turns a statement selects."""
    return statement.options(
        *(orm.selectinload(r) for r in TURN_EVENTS)
    ).execution_options(populate_existing=True)


def first(events: list, at: datetime.datetime | None = None, **match):
    """Returns the first event, recorded up to `at`, matching every field."""
    return next(
        (
            e
            for e in events
            if (at is None or e.created_at <= at)
            and all(getattr(e, k) == v for k, v in match.items())
        ),
        None,
    )


def last(events: list, at: datetime.datetime | None = None, **match):
    """Returns the last event, recorded up to `at`, matching every field."""
    return first(events[::-1], at, **match)


@dataclasses.dataclass
class _TurnEvents:
    """One turn and its events recorded up to the moment it was closed."""

    turn: models.Turn
    reply: models.Reply | None
    verdict: models.Outcome | None
    domain: str
    features: dict[str, float]
    embedding: object  # The topic's, or None.


async def _turn_events(
    session: Session, turn_id: int, at: datetime.datetime
) -> _TurnEvents:
    turn = await session.scalar(
        with_events(sa.select(models.Turn).where(models.Turn.id == turn_id))
    )
    # Domains are recomputed, so replays follow the latest clusters.
    embedding = turn.embeddings[0].embedding if turn.embeddings else None
    return _TurnEvents(
        turn=turn,
        reply=first(turn.replies, at),
        verdict=first(turn.outcomes, at, source=vocab.Source.ASSISTANT),
        domain=(
            await topics.domain_of(session, embedding)
            if embedding is not None
            else topics.ANY_DOMAIN
        ),
        features=await request_features(session, turn, embedding),
        embedding=embedding,
    )


def style_of(turn: models.Turn) -> dict:
    """Returns the decisions logged on a turn, with their propensities."""
    style = turn.style if isinstance(turn.style, dict) else {}
    return {
        "form": {},
        "clarify": None,
        "expertise": None,
        "survey": False,
        "mobile": False,
        "domain": topics.ANY_DOMAIN,
        "propensities": {},
        **style,
    }


async def finalize_turn(
    session: Session,
    turn_id: int,
    at: datetime.datetime,
    p: params_lib.Params,
    next_embedding=None,
    late: dict | None = None,
) -> None:
    """Closes a turn when the next plan_reply of its conversation arrives.

    Args:
        session: The session, inside the caller's transaction.
        turn_id: The turn to close.
        at: When the next plan_reply arrived.
        p: The global parameters.
        next_embedding: The topic embedding of that next request.
        late: Evidence known only after the turn closed, which the nightly
            replay adds: "satisfaction".
    """
    ev = await _turn_events(session, turn_id, at)
    turn, reply, verdict = ev.turn, ev.reply, ev.verdict
    style = style_of(turn)
    conv = await conversation_state(session, turn.conversation_id, p)

    # Length and assistant behaviour.
    duration = (
        (reply.created_at - turn.created_at).total_seconds() if reply else None
    )
    writing = duration - p.overhead if duration is not None else None
    post = length.fuse(
        length.Gaussian3.independent(
            [math.log(turn.needed_words), conv.kappa_mean, conv.speed_mean],
            [LENGTH_PRIOR_VAR, conv.kappa_var, conv.speed_var],
        ),
        length.Sensors(
            exact_words=reply.reply_words_exact if reply else None,
            self_report_words=verdict.last_reply_words if verdict else None,
            writing_seconds=writing,
            requested_words=turn.requested_words,
        ),
        p.noise,
    )
    log_len, len_var = post.mean[0], post.cov[0][0]
    if writing and writing > 0:
        conv.family_probs = p.families.update_with_speed(
            conv.family_probs, log_len - math.log(writing)
        )
    conv.kappa_mean, conv.kappa_var = post.mean[1], post.cov[1][1]
    conv.speed_mean, conv.speed_var = post.mean[2], post.cov[2][2]

    # Reading behaviour.
    ur = await user_reading(session, turn.user_id)
    reply_end = (
        reply.created_at
        if reply
        else turn.created_at
        + datetime.timedelta(
            seconds=math.exp(log_len - conv.speed_mean) + p.overhead
        )
    )
    log_t = math.log(max((at - reply_end).total_seconds(), 1.0))
    resp = reading.responsibilities(
        log_t, log_len - ur.rho_mean, stats.normalize(ur.mixture), p.reading
    )
    ur.rho_mean, ur.rho_var = reading.update_rho(
        (ur.rho_mean, ur.rho_var),
        log_t,
        (log_len, len_var),
        resp[0],
        p.reading,
    )
    ur.mixture = [m + r for m, r in zip(ur.mixture, resp)]

    # Reaction, then the censored length update.
    distance = (
        topics.distance(ev.embedding, next_embedding)
        if ev.embedding is not None and next_embedding is not None
        else None
    )
    outcome_probs = reaction.posterior(
        stats.normalize(ur.reaction_prior),
        observations(
            verdict.observed if verdict else None,
            conv.family_probs,
            resp,
            None if distance is None else distance < p.rephrase_distance,
            **(late or {}),
        ),
        p.confusions,
    )
    ur.reaction_prior = [
        c + q for c, q in zip(ur.reaction_prior, outcome_probs)
    ]
    belief = await length_belief(session, turn.user_id)
    theta_mean, _ = belief.predict(ev.features, p.prior)
    theta.update(
        belief,
        ev.features,
        log_len - math.log(turn.needed_words),
        len_var,
        p.families.bias(conv.family_probs),
        outcome_probs,
        p.prior,
    )
    await save_belief(session, turn.user_id, belief)

    # Form preferences, clarification and expertise.
    await _learn_preferences(
        session, ev, style, dict(zip(reaction.OUTCOMES, outcome_probs))
    )
    session.add_all([conv, ur])
    await session.merge(
        models.TurnInference(
            turn_id=turn_id,
            length_mean=log_len,
            length_var=len_var,
            theta_mean=theta_mean,
            reaction_probs=dict(zip(reaction.OUTCOMES, outcome_probs)),
            reading_probs={
                **dict(zip(reading.COMPONENTS, resp)),
                "seconds": math.exp(log_t),
            },
            next_topic_distance=distance,
        )
    )
    await session.flush()


async def _learn_preferences(
    session: Session, ev: _TurnEvents, style: dict, probs: dict[str, float]
) -> None:
    verdict = ev.verdict
    explicit = (
        [vocab.FORMAT_ISSUES[verdict.format_issue]]
        if verdict and verdict.format_issue
        else []
    )
    counts = collections.defaultdict(float)
    signals = prefs.signals(
        style["form"],
        explicit,
        probs[vocab.Outcome.MOVED_ON],
        probs[vocab.Outcome.WRONG_FORMAT],
    )
    for key, count in signals.items():
        counts[key] += count
    # Asking first, or raising a blocking point, worked when the person
    # moved on.
    moved_on = probs[vocab.Outcome.MOVED_ON]
    for kind in (vocab.ArmKind.CLARIFY, vocab.ArmKind.EXPERTISE):
        if style[kind]:
            dimension = prefs.arm(kind, style[kind])
            counts[(dimension, vocab.PrefValue.YES)] += moved_on
            counts[(dimension, vocab.PrefValue.NO)] += 1 - moved_on
    await add_prefs(
        session,
        ev.turn.user_id,
        prefs.scope_of(ev.domain, style["mobile"]),
        counts,
    )
