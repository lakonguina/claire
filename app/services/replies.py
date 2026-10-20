"""Tool logic, independent of the MCP layer."""

import dataclasses
import datetime
import enum
import hashlib
import json
import math
import random
import secrets
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config
from app.core import crypto
from app.core import db
from app.core import i18n
from app.core import text
from app.core import vocab
from app.model import family as family_lib
from app.model import policy
from app.model import prefs
from app.model import theta
from app import models
from app.services import engine
from app.services import params as params_lib
from app.services import topics

Session = sa_asyncio.AsyncSession
CONVERSATION_PREFIX = "c_"
TURN_PREFIX = "t_"
QUOTE_FREE = frozenset({vocab.Outcome.MOVED_ON, vocab.Outcome.NO_SIGNAL})
MIN_NEEDED, MAX_NEEDED = 5, 2000
MIN_CHOICES, MAX_CHOICES = 2, 6
MAX_CHOICE_WORDS, MAX_CHOICE_CHARS = 12, 100
MAX_VISIBLE_SECONDS = 86_400.0
ACTIVE = datetime.timedelta(days=30)
# A choice answered in the free field rather than by an option.
FREE_CHOICE = "free"
rng = random.Random()
SECRET_BYTES = 32


class HandleError(Exception):
    """A handle that is malformed or belongs to another user."""


class ReplyShownError(Exception):
    """The turn's reply is already shown: one show_reply per turn."""


@dataclasses.dataclass(frozen=True)
class Caller:
    """Who calls a tool, and through which host."""

    user_id: int
    host: str
    ui: bool
    protocol_version: str | None = None
    transport_session: str | None = None
    host_version: str | None = None
    capabilities: dict = dataclasses.field(default_factory=dict)


def new_secret() -> str:
    """Returns a random, unguessable token."""
    return secrets.token_urlsafe(SECRET_BYTES)


def token_hash(token: str) -> str:
    """Returns the stored hash of a token: tokens are never stored."""
    return hashlib.sha256(token.encode()).hexdigest()


def compact(data: dict) -> str:
    """Serializes a tool result without any whitespace, for the prompt cache."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def _parse(handle: Any, prefix: str) -> int:
    if not isinstance(handle, str) or not handle.startswith(prefix):
        raise HandleError
    number = handle.removeprefix(prefix)
    if not number.isdigit():
        raise HandleError
    return int(number)


async def _lock(session: Session, caller: Caller) -> None:
    """Serialises one user's writes for the rest of the transaction."""
    await session.execute(
        sa.select(sa.func.pg_advisory_xact_lock(caller.user_id))
    )


async def _owned_turn(
    session: Session, caller: Caller, handle: Any
) -> tuple[models.Turn, models.Conversation]:
    turn = await session.get(models.Turn, _parse(handle, TURN_PREFIX))
    if not turn or turn.user_id != caller.user_id:
        raise HandleError
    await _lock(session, caller)
    return turn, await session.get(models.Conversation, turn.conversation_id)


async def by_user(session: Session, now: datetime.datetime) -> bool:
    """Returns whether arms are drawn per person rather than per chat.

    Per chat, a person who learns brevity in their Claire chats keeps it in
    their holdout chats, which underestimates the gain; per person needs
    enough people to afford it.
    """
    unit = config.settings.control_unit
    if unit != vocab.RandomizationUnit.AUTO:
        return unit == vocab.RandomizationUnit.USER
    active = await session.scalar(
        sa.select(sa.func.count(sa.distinct(models.Turn.user_id))).where(
            models.Turn.created_at > now - ACTIVE
        )
    )
    return active >= config.settings.per_user_from


def _arm(user_id: int, per_user: bool) -> tuple[vocab.Arm, float]:
    """Draws the arm of a new conversation; returns it and its propensity."""
    draw = (
        random.Random(f"arm:{user_id}").random() if per_user else rng.random()
    )
    shares = config.settings.arm_shares
    for arm, share in shares.items():
        if draw < share:
            return arm, share
        draw -= share
    return vocab.Arm.TREATMENT, 1 - sum(shares.values())


def _ablation() -> tuple[str | None, float]:
    """Draws the field a new conversation never asks, with its propensity."""
    settings = config.settings
    fields = settings.ablation_fields
    if fields and rng.random() < settings.ablation_share:
        return rng.choice(fields), settings.ablation_share / len(fields)
    return None, 1 - settings.ablation_share if fields else 1.0


def _catalog() -> tuple[vocab.Catalog, float]:
    """Draws the prompt catalog of a new conversation, with its propensity."""
    share = config.settings.candidate_share if i18n.has_candidate() else 0.0
    if rng.random() < share:
        return vocab.Catalog.CANDIDATE, share
    return vocab.Catalog.CURRENT, 1 - share


async def _conversation(
    session: Session, caller: Caller, args: dict, now: datetime.datetime
) -> models.Conversation:
    handle = args.get("conversation_id")
    if handle is not None:
        conv = await session.get(
            models.Conversation, _parse(handle, CONVERSATION_PREFIX)
        )
        if not conv or conv.user_id != caller.user_id:
            raise HandleError
        # A conversation lives in one host: an id carried over from another
        # (e.g. the web app's, reused in Claude Code) starts a new one there.
        if conv.host == caller.host:
            return conv
    arm, propensity = _arm(caller.user_id, await by_user(session, now))
    ablated, ablation_p = _ablation()
    catalog, catalog_p = _catalog()
    conv = models.Conversation(
        user_id=caller.user_id,
        grp=arm,
        arm_propensity=propensity,
        host=caller.host,
        host_version=caller.host_version,
        capabilities=caller.capabilities,
        ui_supported=caller.ui,
        protocol_version=caller.protocol_version,
        sampled=rng.random() < config.settings.message_sample,
        ablated_field=ablated,
        ablation_propensity=ablation_p,
        catalog=catalog,
        catalog_propensity=catalog_p,
    )
    session.add(conv)
    await session.flush()
    return conv


def _form(counts: prefs.Counts) -> tuple[dict[str, str], dict[str, float]]:
    """Draws a value for every form dimension, with its propensity."""
    drawn = {
        d: prefs.with_propensity(lambda d=d: prefs.thompson(counts, d, rng))
        for d in vocab.Dimension
    }
    return {d: v for d, (v, _) in drawn.items()}, {
        d: p for d, (_, p) in drawn.items()
    }


def _bandit(
    counts: prefs.Counts, kind: vocab.ArmKind, arms: type[enum.StrEnum]
) -> tuple[str, float]:
    """Draws an arm of a two-value bandit, with its propensity."""
    return prefs.with_propensity(
        lambda: prefs.best_arm(counts, kind, list(arms), rng)
    )


def calibrated(ambiguity: float, p: params_lib.Params) -> float:
    """Returns the assistant's ambiguity, recalibrated on what followed.

    Platt scaling, refit nightly against misunderstood replies and rephrased
    requests: assistants are poorly calibrated.
    """
    clipped = min(max(ambiguity, 1e-4), 1 - 1e-4)
    logit = math.log(clipped / (1 - clipped))
    return 1 / (1 + math.exp(-(p.ambiguity_a * logit + p.ambiguity_b)))


@dataclasses.dataclass
class _Asks:
    """What the instruction asks beyond the reply itself."""

    message: bool = False
    choices: bool = False
    omit: str | None = None


def _instruction(
    conv: models.Conversation,
    requested: int | None,
    style: dict,
    ui: bool,
    asks: _Asks,
) -> str:
    """Composes the instruction from the conversation's prompt catalog."""
    prompts = i18n.prompts(conv.catalog == vocab.Catalog.CANDIDATE)
    if conv.grp != vocab.Arm.TREATMENT:
        parts = [prompts["arms"][conv.grp]]
    elif style["clarify"] == vocab.Clarify.ASK:
        parts = [prompts["clarify_widget" if asks.choices else "clarify"]]
    else:
        fragments = prompts["fragments"]
        parts = [
            prompts["reply_in_words"].format(words=requested),
            prompts["no_preamble"],
            *(
                fragments[d][v]
                for d, v in style["form"].items()
                if v != vocab.PrefValue.AUTO
            ),
        ]
        if style["expertise"] == vocab.Expertise.RAISE:
            parts.append(prompts["expertise"])
    if ui:
        parts.append(prompts["widget_mode"])
    if asks.message:
        parts.append(prompts["send_message"])
    if asks.omit:
        parts.append(prompts["omit_field"].format(field=asks.omit))
    return " ".join(parts)


def _log_call(session: Session, caller: Caller, tool: str) -> None:
    session.add(models.ToolCall(user_id=caller.user_id, tool=tool))


async def _previous(
    session: Session, conv: models.Conversation, handle: Any
) -> tuple[models.Turn | None, models.Turn | None]:
    """Returns the conversation's latest turn and the turn this call closes.

    The closed turn is the assistant's previous one, last_turn_id, when it
    names a turn of this conversation; otherwise the latest. They differ when
    the person edited or regenerated an earlier message.
    """
    latest = await session.scalar(
        sa.select(models.Turn)
        .where(models.Turn.conversation_id == conv.id)
        .order_by(models.Turn.turn_index.desc())
        .limit(1)
    )
    if handle is None or latest is None:
        return latest, latest
    try:
        named = await session.get(models.Turn, _parse(handle, TURN_PREFIX))
    except HandleError:
        return latest, latest
    if named is None or named.conversation_id != conv.id:
        return latest, latest
    return latest, named


async def _close(
    session: Session,
    prev: models.Turn,
    args: dict,
    now: datetime.datetime,
    p: params_lib.Params,
    embedding,
) -> None:
    """Records the verdict on the closed turn; closes it the first time."""
    already = await session.scalar(
        sa.select(sa.exists().where(models.Turn.follows == prev.id))
    )
    outcome = args.get("last_reply_outcome") or vocab.Outcome.NO_SIGNAL
    if outcome not in QUOTE_FREE and not (args.get("user_quote") or "").strip():
        outcome = vocab.Outcome.NO_SIGNAL
    session.add(
        models.Outcome(
            turn_id=prev.id,
            source=vocab.Source.ASSISTANT,
            observed=outcome,
            user_quote=args.get("user_quote"),
            last_reply_words=args.get("last_reply_words"),
            format_issue=(
                args.get("format_issue")
                if outcome == vocab.Outcome.WRONG_FORMAT
                else None
            ),
            goal_reached=args.get("goal_reached"),
            model_version=prev.model_version,
            schema_version=config.SCHEMA_VERSION,
        )
    )
    await session.flush()
    if not already:
        await engine.finalize_turn(session, prev.id, now, p, embedding)


async def _mobile(session: Session, conv: models.Conversation) -> bool:
    """Returns whether the widget last reported this conversation on a phone."""
    platform = await session.scalar(
        sa.select(models.TurnContext.platform)
        .join(models.Turn, models.Turn.id == models.TurnContext.turn_id)
        .where(models.Turn.conversation_id == conv.id)
        .order_by(models.TurnContext.created_at.desc())
        .limit(1)
    )
    return platform == vocab.Platform.MOBILE


def _message(session: Session, turn_id: int, message: Any) -> int | None:
    """Records the user message the assistant copied; returns its length."""
    if not isinstance(message, str) or not message.strip():
        return None
    words = text.count_words(message)
    keep = words <= config.settings.max_message_words
    session.add(
        models.Message(
            turn_id=turn_id, words=words, text=message if keep else None
        )
    )
    return words


def _complies(
    p: params_lib.Params, family: str, asked: vocab.Requested
) -> bool:
    """Returns whether this behaviour class does what the instruction asks."""
    return f"{family}:{asked}" not in p.noncompliant


async def plan_reply(caller: Caller, args: dict) -> dict:
    """Closes the previous turn and decides this one.

    Args:
        caller: Who calls.
        args: The tool arguments, as sent (stored as the raw payload, except
            the user message, stored apart and encrypted).

    Returns:
        The compact plan: conversation and turn ids, and the instruction.
    """
    settings = config.settings
    message = args.pop("user_message", None)
    embedding = await topics.embed(args["topic"])
    needed = min(max(int(args["needed_words"]), MIN_NEEDED), MAX_NEEDED)
    length_policy = policy.POLICIES[settings.policy]
    async with db.session() as session, session.begin():
        await _lock(session, caller)
        _log_call(session, caller, "plan_reply")
        p = await params_lib.cache.get(session)
        now = await session.scalar(sa.select(sa.func.now()))
        conv = await _conversation(session, caller, args, now)
        # A field this conversation never asks is ignored if sent anyway.
        args.pop(conv.ablated_field or "", None)
        latest, prev = await _previous(session, conv, args.get("last_turn_id"))
        if prev:
            await _close(session, prev, args, now, p, embedding)
        domain = await topics.domain_of(session, embedding)
        mobile = await _mobile(session, conv)
        x = theta.features(await topics.memberships(session, embedding), mobile)
        belief = await engine.length_belief(session, caller.user_id)
        theta_mean, theta_var = belief.predict(x, p.prior)
        state = await engine.conversation_state(session, conv.id, p)
        family = max(
            state.family_probs,
            key=state.family_probs.get,
            default=family_lib.POPULATION,
        )
        b_mean, b_var = p.families.bias(state.family_probs)
        ctx = policy.Context(
            needed,
            theta_mean,
            theta_var,
            b_mean,
            b_var,
            state.kappa_mean,
            p.prior.tau,
            settings.q,
        )
        decision, propensity = length_policy(rng).decide(ctx)
        scope = prefs.scope_of(domain, mobile)
        counts = (await engine.pref_counts(session, caller.user_id, [scope]))[
            scope
        ]
        treated = conv.grp == vocab.Arm.TREATMENT
        ambiguity = args.get("ambiguity")
        ambiguous = (
            None if ambiguity is None else calibrated(float(ambiguity), p)
        )
        clarify, clarify_p = (
            _bandit(counts, vocab.ArmKind.CLARIFY, vocab.Clarify)
            if treated
            and ambiguous is not None
            and ambiguous >= settings.ambiguity_threshold
            else (None, None)
        )
        answers = treated and clarify != vocab.Clarify.ASK
        expertise, expertise_p = (
            _bandit(counts, vocab.ArmKind.EXPERTISE, vocab.Expertise)
            if answers
            else (None, None)
        )
        form, form_p = _form(counts) if treated else ({}, {})
        asks = _Asks(
            message=conv.sampled
            and _complies(p, family, vocab.Requested.USER_MESSAGE),
            choices=caller.ui and _complies(p, family, vocab.Requested.CHOICES),
            omit=conv.ablated_field,
        )
        # The ground-truth question follows a sample of the goals reached.
        share = settings.ground_truth_share
        eligible = caller.ui and bool(args.get("goal_reached"))
        survey = eligible and rng.random() < share
        style = {
            "form": form,
            "clarify": clarify,
            "expertise": expertise,
            "survey": survey,
            "mobile": mobile,
            "domain": domain,
            "ambiguity_calibrated": ambiguous,
            "asks": {
                vocab.Requested.USER_MESSAGE: asks.message,
                vocab.Requested.CHOICES: asks.choices
                and clarify == vocab.Clarify.ASK,
            },
            "propensities": {
                "form": form_p,
                "clarify": clarify_p,
                "expertise": expertise_p,
                "survey": (
                    (share if survey else 1 - share) if eligible else None
                ),
            },
        }
        instruction = _instruction(
            conv, decision.requested_words, style, caller.ui, asks
        )
        turn = models.Turn(
            user_id=caller.user_id,
            conversation_id=conv.id,
            turn_index=latest.turn_index + 1 if latest else 1,
            reply_mode=(
                vocab.ReplyMode.WIDGET
                if caller.ui
                else vocab.ReplyMode.FALLBACK
            ),
            model_version=length_policy.version,
            schema_version=config.SCHEMA_VERSION,
            catalog_version=i18n.catalog_version(
                conv.catalog == vocab.Catalog.CANDIDATE
            ),
            raw_payload=args,
            instruction=instruction,
            topic=args["topic"],
            needed_words=needed,
            language=args["language"],
            user_message_words=args.get("user_message_words"),
            ambiguity=ambiguity,
            messages_since_last_call=args.get("messages_since_last_call"),
            follows=prev.id if prev else None,
            branched=prev is not latest,
            target_words=decision.target_words if treated else None,
            requested_words=decision.requested_words if treated else None,
            propensity=propensity if treated else None,
            theta_sample=decision.theta_sample,
            b_sample=decision.b_sample,
            post_mean=decision.post_mean,
            post_sd=decision.post_sd,
            style=style,
            transport_session=caller.transport_session,
        )
        session.add(turn)
        await session.flush()
        words = _message(session, turn.id, message)
        if words is not None and turn.user_message_words is None:
            turn.user_message_words = words
        if isinstance(args.get("goal"), str) and args["goal"].strip():
            session.add(
                models.Goal(
                    turn_id=turn.id,
                    conversation_id=conv.id,
                    text=args["goal"].strip(),
                )
            )
        session.add(
            models.TopicEmbedding(
                turn_id=turn.id,
                model=config.settings.embedding_model,
                embedding=embedding,
            )
        )
        for version in settings.shadow_policies:
            shadow, shadow_p = policy.POLICIES[version](rng).decide(ctx)
            session.add(
                models.ShadowDecision(
                    turn_id=turn.id,
                    model_version=version,
                    target_words=shadow.target_words,
                    propensity=shadow_p,
                    details={"requested_words": shadow.requested_words},
                )
            )
    return {
        "conversation_id": f"{CONVERSATION_PREFIX}{conv.id}",
        "turn_id": f"{TURN_PREFIX}{turn.id}",
        "instruction": instruction,
    }


def _choice(value: Any) -> str | None:
    """Returns an option of a choice question that passes the checks."""
    if not isinstance(value, str):
        return None
    option = " ".join(value.split())
    valid = 0 < len(option.split()) <= MAX_CHOICE_WORDS
    return option if valid and len(option) <= MAX_CHOICE_CHARS else None


def _choices(style: dict, values: Any) -> list[str]:
    """Returns the options to show: only for a clarifying question."""
    if style.get("clarify") != vocab.Clarify.ASK or not isinstance(
        values, list
    ):
        return []
    options = [o for o in map(_choice, values[:MAX_CHOICES]) if o]
    return options if len(options) >= MIN_CHOICES else []


async def show_reply(
    caller: Caller, turn_handle: Any, reply: Any, choices: Any = None
) -> dict:
    """Stores a reply, encrypted, with its word count; returns the widget data.

    Args:
        caller: Who calls.
        turn_handle: The turn id returned by plan_reply.
        reply: The reply text.
        choices: The options of a clarifying question, when the
            instruction asked for one.

    Returns:
        The widget's structured content.

    Raises:
        HandleError: On an invalid reply or turn.
        ReplyShownError: When the turn already has its reply.
    """
    if not isinstance(reply, str):
        raise HandleError
    words = text.count_words(reply)
    async with db.session() as session, session.begin():
        turn, _ = await _owned_turn(session, caller, turn_handle)
        # _owned_turn holds the user's lock, so two calls cannot both pass.
        if await session.scalar(
            sa.select(sa.exists().where(models.Reply.turn_id == turn.id))
        ):
            raise ReplyShownError
        _log_call(session, caller, "show_reply")
        style = engine.style_of(turn)
        options = _choices(style, choices)
        session.add(
            models.Reply(
                turn_id=turn.id,
                reply_words_exact=words,
                text=reply,
                choices=len(options),
            )
        )
        return {
            "turn_id": turn_handle,
            "choices": options,
            "survey": bool(style["survey"]),
            "labels": i18n.catalog(turn.language)["widget"],
            "privacy": {
                "email": config.settings.privacy_email,
                "code": crypto.privacy_code(caller.user_id),
            },
        }


def _event_value(kind: Any, value: Any) -> Any:
    """Returns the value an event may carry, or raises HandleError."""
    if kind == vocab.WidgetEvent.VISIBLE:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HandleError
        return min(max(float(value), 0.0), MAX_VISIBLE_SECONDS)
    if kind == vocab.WidgetEvent.CHOICE:
        # The index of the option picked; a free answer is not stored here:
        # it reaches the chat as the person's next message.
        if value == FREE_CHOICE or (
            isinstance(value, int) and 0 <= value < MAX_CHOICES
        ):
            return value
        raise HandleError
    if kind == vocab.WidgetEvent.SATISFACTION:
        if value not in vocab.Satisfaction:
            raise HandleError
        return value
    if kind not in vocab.WidgetEvent:
        raise HandleError
    return None


async def record_event(
    caller: Caller, turn_handle: Any, kind: Any, value: Any = None
) -> None:
    """Records what the widget saw about a shown reply."""
    checked = _event_value(kind, value)
    async with db.session() as session, session.begin():
        turn, _ = await _owned_turn(session, caller, turn_handle)
        _log_call(session, caller, "record_event")
        session.add(
            models.WidgetEvent(
                turn_id=turn.id, kind=kind, value={"value": checked}
            )
        )


async def report_context(
    caller: Caller,
    turn_handle: Any,
    platform: Any,
    locale: Any,
    time_zone: Any = None,
) -> None:
    """Records the host context the widget saw."""
    async with db.session() as session, session.begin():
        turn, _ = await _owned_turn(session, caller, turn_handle)
        session.add(
            models.TurnContext(
                turn_id=turn.id,
                platform=platform if platform in vocab.Platform else None,
                locale=locale if isinstance(locale, str) else None,
                time_zone=time_zone if isinstance(time_zone, str) else None,
            )
        )
