"""Derived-state updates (SPEC 6 and 7).

The live service and the nightly replay both go through finalize_turn and
apply_edit, which read only events recorded up to the moment they happened,
so a replay reproduces the live state.
"""

import collections
from collections.abc import Iterable
import dataclasses
import datetime
import math

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext import asyncio as sa_asyncio

from app import models
from app import params as params_lib
from app import topics
from app import vocab
from app.model import bandits
from app.model import family
from app.model import length
from app.model import prefs
from app.model import reaction
from app.model import reading
from app.model import stats
from app.model import theta

Session = sa_asyncio.AsyncSession
KAPPA_PRIOR_VAR = 0.3**2
LENGTH_PRIOR_VAR = 1.0
NO_CLICK = "no_click"


async def conversation_state(
    session: Session, conversation_id: int, p: params_lib.Params
) -> models.ConversationState:
    """Returns the conversation's assistant state, or a new one at its prior."""
    state = await session.get(models.ConversationState, conversation_id)
    if state:
        return state
    conv = await session.get(models.Conversation, conversation_id)
    probs = p.families.initial_probs(
        conv.host, family.family_of(conv.assistant_model)
    )
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


async def grids(
    session: Session, user_id: int, cell: str, p: params_lib.Params
) -> tuple[models.ThetaPosterior, models.AlphaPosterior]:
    """Returns the cell's theta posterior and the user's alpha posterior."""
    alpha = await session.get(models.AlphaPosterior, user_id)
    if alpha is None:
        alpha = models.AlphaPosterior(
            user_id=user_id, grid=theta.alpha_prior(p.hyper)
        )
    cell_theta = await session.get(models.ThetaPosterior, (user_id, cell))
    if cell_theta is None:
        cell_theta = models.ThetaPosterior(
            user_id=user_id,
            request_type=cell,
            grid=theta.new_cell(alpha.grid, cell, p.hyper),
        )
    return cell_theta, alpha


async def pref_counts(
    session: Session, user_id: int, scopes: Iterable[str]
) -> dict[str, prefs.Counts]:
    """Returns, for each scope, the counts of its cell, user and population.

    Args:
        session: The session.
        user_id: The user.
        scopes: The cell scopes needed.

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
                counts[("cell", *key)] = r.count
            if r.scope == prefs.USER_SCOPE:
                counts[("user", *key)] = r.count
    return out


async def add_prefs(
    session: Session,
    user_id: int,
    scope: str,
    counts: dict[tuple[str, str], float],
) -> None:
    """Adds counts to a cell, its user level and the population."""
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
    click: tuple[str, int] | None,
    any_shown: bool,
    reading_probs: list[float] | None,
) -> list[reaction.Observation]:
    """Returns the annotator observations of one turn.

    A click makes the assistant verdict dependent (SPEC 6.4), so the verdict
    is dropped when there is one. A click on a retired button tells nothing.

    Args:
        verdict: The assistant verdict, if any.
        family_probs: The conversation's model-family probabilities.
        click: The clicked button and its position, if any.
        any_shown: Whether buttons were shown.
        reading_probs: P(read), P(skim), P(away), if known.

    Returns:
        The observations.
    """
    obs = []
    if verdict and verdict != vocab.Outcome.NO_SIGNAL and click is None:
        obs.append(
            reaction.Observation(
                vocab.Source.ASSISTANT, {verdict: 1.0}, dict(family_probs)
            )
        )
    effect = bandits.EFFECTS.get(click[0]) if click else None
    if any_shown and (click is None or effect):
        token = f"{click[1]}:{effect.outcome}" if click else NO_CLICK
        obs.append(reaction.Observation(vocab.Source.BUTTON, {token: 1.0}))
    if reading_probs:
        read, skim, away = reading_probs
        obs.append(
            reaction.Observation(
                vocab.Source.READING,
                {"read": read, "skim": skim, reaction.NO_INFO: away},
            )
        )
    return obs


@dataclasses.dataclass
class _TurnEvents:
    """One turn and its events recorded up to the moment it was closed."""

    turn: models.Turn
    reply: models.Reply | None
    verdict: models.Outcome | None
    click: models.Click | None
    shown: list[models.ButtonShown]
    domain: str


async def _first(
    session: Session, model, turn_id: int, at: datetime.datetime, *where
):
    return await session.scalar(
        sa.select(model)
        .where(model.turn_id == turn_id, model.created_at <= at, *where)
        .order_by(model.created_at)
        .limit(1)
    )


async def _turn_events(
    session: Session, turn_id: int, at: datetime.datetime
) -> _TurnEvents:
    embedding = await session.scalar(
        sa.select(models.TopicEmbedding.embedding).where(
            models.TopicEmbedding.turn_id == turn_id
        )
    )
    shown = await session.scalars(
        sa.select(models.ButtonShown).where(
            models.ButtonShown.turn_id == turn_id
        )
    )
    return _TurnEvents(
        turn=await session.get(models.Turn, turn_id),
        reply=await _first(session, models.Reply, turn_id, at),
        verdict=await _first(
            session,
            models.Outcome,
            turn_id,
            at,
            models.Outcome.source == vocab.Source.ASSISTANT,
        ),
        click=await _first(session, models.Click, turn_id, at),
        shown=list(shown),
        domain=(
            await topics.domain_of(session, embedding)
            if embedding is not None
            else topics.ANY_DOMAIN
        ),
    )


def style_of(turn: models.Turn) -> dict:
    """Returns the decisions logged on a turn (SPEC 5.5)."""
    style = turn.style if isinstance(turn.style, dict) else {}
    return {"form": {}, "clarify": None, "cell": turn.request_type, **style}


async def finalize_turn(
    session: Session,
    turn_id: int,
    at: datetime.datetime,
    p: params_lib.Params,
) -> None:
    """Closes a turn when the next plan_reply of its conversation arrives.

    Args:
        session: The session, inside the caller's transaction.
        turn_id: The turn to close.
        at: When the next plan_reply arrived.
        p: The global parameters.
    """
    ev = await _turn_events(session, turn_id, at)
    turn, reply, verdict = ev.turn, ev.reply, ev.verdict
    style = style_of(turn)
    conv = await conversation_state(session, turn.conversation_id, p)

    # Length (6.2) and assistant behaviour (6.3).
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

    # Reading behaviour (6.6).
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

    # Reaction (6.4), then the censored length update (6.5).
    click = (ev.click.button, ev.click.position) if ev.click else None
    outcome_probs = reaction.posterior(
        stats.normalize(ur.reaction_prior),
        observations(
            verdict.observed if verdict else None,
            conv.family_probs,
            click,
            bool(ev.shown),
            resp,
        ),
        p.confusions,
    )
    ur.reaction_prior = [
        c + q for c, q in zip(ur.reaction_prior, outcome_probs)
    ]
    cell_theta, alpha = await grids(session, turn.user_id, style["cell"], p)
    cell_theta.grid, alpha.grid = theta.update(
        cell_theta.grid,
        alpha.grid,
        style["cell"],
        log_len - math.log(turn.needed_words),
        len_var,
        p.families.bias(conv.family_probs),
        outcome_probs,
        p.hyper,
    )

    # Form preferences (7), buttons (8) and clarification (4.1).
    await _learn_preferences(
        session, ev, style, dict(zip(reaction.OUTCOMES, outcome_probs))
    )
    session.add_all([conv, ur, cell_theta, alpha])
    await session.merge(
        models.TurnInference(
            turn_id=turn_id,
            length_mean=log_len,
            length_var=len_var,
            reaction_probs=dict(zip(reaction.OUTCOMES, outcome_probs)),
            reading_probs={
                **dict(zip(reading.COMPONENTS, resp)),
                "seconds": math.exp(log_t),
            },
        )
    )
    await session.flush()


async def _learn_preferences(
    session: Session, ev: _TurnEvents, style: dict, probs: dict[str, float]
) -> None:
    turn, click, verdict = ev.turn, ev.click, ev.verdict
    effect = bandits.EFFECTS.get(click.button) if click else None
    explicit = []
    if effect and effect.preference:
        explicit.append(effect.preference)
    elif not click and verdict and verdict.format_issue:
        explicit.append(vocab.FORMAT_ISSUES[verdict.format_issue])
    by_scope = collections.defaultdict(dict)
    signals = prefs.signals(
        style["form"],
        explicit,
        probs[vocab.Outcome.MOVED_ON],
        probs[vocab.Outcome.WRONG_FORMAT],
    )
    for (dimension, value), count in signals.items():
        scope = prefs.scope_of(dimension, turn.request_type, ev.domain)
        by_scope[scope][(dimension, value)] = count
    for shown in ev.shown:
        clicked = bool(click) and click.button == shown.button
        dimension = prefs.arm(vocab.ArmKind.BUTTON, shown.button)
        value = vocab.PrefValue.YES if clicked else vocab.PrefValue.NO
        by_scope[turn.request_type][(dimension, value)] = 1.0
    if style["clarify"]:
        dimension = prefs.arm(vocab.ArmKind.CLARIFY, style["clarify"])
        moved_on = probs[vocab.Outcome.MOVED_ON]
        by_scope[turn.request_type][(dimension, vocab.PrefValue.YES)] = moved_on
        by_scope[turn.request_type][(dimension, vocab.PrefValue.NO)] = (
            1 - moved_on
        )
    for scope, counts in by_scope.items():
        await add_prefs(session, turn.user_id, scope, counts)


async def apply_edit(
    session: Session, edit: models.ProfileEdit, p: params_lib.Params
) -> None:
    """Applies a profile-page correction (SPEC 4.6), a very strong signal."""
    if edit.dimension == vocab.LENGTH:
        cell_theta, _ = await grids(session, edit.user_id, edit.scope, p)
        cell_theta.grid = theta.edit(cell_theta.grid, edit.value)
        session.add(cell_theta)
        await session.flush()
        return
    await add_prefs(
        session,
        edit.user_id,
        edit.scope,
        {(edit.dimension, edit.value): prefs.EDIT},
    )
