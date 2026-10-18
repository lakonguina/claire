"""Tool logic, independent of the MCP layer (SPEC 4)."""

import dataclasses
import hashlib
import json
import random
import secrets
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import db
from app import engine
from app import i18n
from app import models
from app import params as params_lib
from app import questions
from app import text
from app import topics
from app import vocab
from app.model import bandits
from app.model import policy
from app.model import prefs
from app.model import theta

Session = sa_asyncio.AsyncSession
CONVERSATION_PREFIX = "c_"
TURN_PREFIX = "t_"
QUOTE_FREE = frozenset({vocab.Outcome.MOVED_ON, vocab.Outcome.NO_SIGNAL})
MIN_NEEDED, MAX_NEEDED = 5, 2000
MAX_LABEL_WORDS = 8
MAX_LABEL_CHARS = 60
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


def new_secret() -> str:
    """Returns a random, unguessable token."""
    return secrets.token_urlsafe(SECRET_BYTES)


def token_hash(token: str) -> str:
    """Returns the stored hash of a token: tokens are never stored."""
    return hashlib.sha256(token.encode()).hexdigest()


def new_user() -> models.User:
    """Returns a new anonymous user, with the secret key of their profile."""
    return models.User(profile_key=new_secret())


def profile_url(key: str) -> str:
    """Returns the profile page URL of a user's profile key (SPEC 4.6)."""
    return f"{config.settings.public_url.rstrip("/")}/me/{key}"


def compact(data: dict) -> str:
    """Serializes a tool result without any whitespace (SPEC 3)."""
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


def _arm(user_id: int) -> vocab.Arm:
    """Draws the arm of a new conversation (SPEC 4.1)."""
    settings = config.settings
    by_user = settings.control_unit == vocab.RandomizationUnit.USER
    draw = random.Random(f"arm:{user_id}").random() if by_user else rng.random()
    for arm, share in settings.arm_shares.items():
        if draw < share:
            return arm
        draw -= share
    return vocab.Arm.TREATMENT


async def _conversation(
    session: Session, caller: Caller, args: dict
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
    conv = models.Conversation(
        user_id=caller.user_id,
        grp=_arm(caller.user_id),
        host=caller.host,
        ui_supported=caller.ui,
        protocol_version=caller.protocol_version,
    )
    session.add(conv)
    await session.flush()
    return conv


def _form(counts: prefs.Counts) -> dict[str, str]:
    """Draws a value for every form dimension (SPEC 7)."""
    return {d: prefs.thompson(counts, d, rng) for d in vocab.Dimension}


def _clarify(counts: prefs.Counts) -> vocab.Clarify:
    """Draws between asking and answering (SPEC 4.1)."""
    draws = {
        a: prefs.draw(
            prefs.posterior(counts, prefs.arm(vocab.ArmKind.CLARIFY, a)), rng
        )[vocab.PrefValue.YES]
        for a in vocab.Clarify
    }
    return max(draws, key=draws.get)


def _instruction(arm: str, requested: int | None, style: dict, ui: bool) -> str:
    """Composes the instruction from the English prompt catalog."""
    prompts = i18n.prompts()
    if arm != vocab.Arm.TREATMENT:
        parts = [prompts["arms"][arm]]
    elif style["clarify"] == vocab.Clarify.ASK:
        parts = [prompts["clarify"]]
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
    if ui:
        parts.append(prompts["widget_mode"])
    to_write = [b["button"] for b in style["buttons"] if "label" not in b]
    if to_write:
        parts.append(
            prompts["buttons"].format(
                count=len(to_write),
                intents="; ".join(prompts["intents"][b] for b in to_write),
            )
        )
    return " ".join(parts)


def _log_call(session: Session, caller: Caller, tool: str) -> None:
    session.add(models.ToolCall(user_id=caller.user_id, tool=tool))


async def _close_previous(
    session: Session,
    conv: models.Conversation,
    args: dict,
    now: Any,
    p: params_lib.Params,
) -> models.Turn | None:
    """Records the verdict on the previous turn and closes it."""
    prev = await session.scalar(
        sa.select(models.Turn)
        .where(models.Turn.conversation_id == conv.id)
        .order_by(models.Turn.turn_index.desc())
        .limit(1)
    )
    if not prev:
        return None
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
            model_version=prev.model_version,
            schema_version=config.SCHEMA_VERSION,
        )
    )
    await session.flush()
    await engine.finalize_turn(session, prev.id, now, p)
    return prev


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


async def plan_reply(caller: Caller, args: dict) -> dict:
    """Closes the previous turn and decides this one (SPEC 4.1).

    Args:
        caller: Who calls.
        args: The tool arguments, as sent (stored as the raw payload).

    Returns:
        The compact plan: conversation and turn ids, target and instruction.
    """
    settings = config.settings
    embedding = await topics.embed(args["topic"])
    needed = min(max(int(args["needed_words"]), MIN_NEEDED), MAX_NEEDED)
    length_policy = policy.POLICIES[settings.policy]
    async with db.session() as session, session.begin():
        await _lock(session, caller)
        _log_call(session, caller, "plan_reply")
        p = await params_lib.cache.get(session)
        now = await session.scalar(sa.select(sa.func.now()))
        conv = await _conversation(session, caller, args)
        prev = await _close_previous(session, conv, args, now, p)
        domain = await topics.domain_of(session, embedding)
        mobile = await _mobile(session, conv)
        x = theta.features(await topics.memberships(session, embedding), mobile)
        belief = await engine.length_belief(session, caller.user_id)
        theta_mean, theta_var = belief.predict(x, p.prior)
        state = await engine.conversation_state(session, conv.id, p)
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
        counts = (await engine.pref_counts(session, caller.user_id, [domain]))[
            domain
        ]
        treated = conv.grp == vocab.Arm.TREATMENT
        clarify = (
            _clarify(counts) if treated and args.get("ambiguous") else None
        )
        # Buttons go under widget replies of the treatment arm, except when
        # the reply is a clarifying question.
        offered = treated and caller.ui and clarify != vocab.Clarify.ASK
        buttons = (
            await _plan_buttons(session, counts, embedding, args["language"])
            if offered
            else []
        )
        style = {
            "form": _form(counts) if treated else {},
            "clarify": clarify,
            "mobile": mobile,
            "domain": domain,
            "buttons": buttons,
        }
        turn = models.Turn(
            user_id=caller.user_id,
            conversation_id=conv.id,
            turn_index=prev.turn_index + 1 if prev else 1,
            reply_mode=(
                vocab.ReplyMode.WIDGET
                if caller.ui
                else vocab.ReplyMode.FALLBACK
            ),
            model_version=length_policy.version,
            schema_version=config.SCHEMA_VERSION,
            raw_payload=args,
            topic=args["topic"],
            needed_words=needed,
            language=args["language"],
            user_message_words=args.get("user_message_words"),
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
        "target_words": decision.target_words if treated else None,
        "instruction": _instruction(
            conv.grp, decision.requested_words, style, caller.ui
        ),
    }


async def _plan_buttons(
    session: Session, counts: prefs.Counts, embedding, language: str
) -> list[dict]:
    """Chooses the turn's button intents (SPEC 4.4, 8).

    The next question is a frequent one when people asked one (SPEC 4.8);
    the assistant writes every other label.
    """
    buttons = []
    for shown in bandits.choose(counts, rng):
        button = {
            "button": shown.button,
            "shuffled": shown.shuffled,
            "draw": shown.thompson_draw,
        }
        if shown.button == vocab.Button.RELATED:
            frequent = await questions.follow_up(
                session, embedding, language, rng
            )
            if frequent:
                button |= {
                    "label": frequent["text"],
                    "question": frequent["id"],
                }
        buttons.append(button)
    return buttons


def _label(value: Any) -> str | None:
    """Returns an assistant's button label that passes the checks, or None."""
    if not isinstance(value, str):
        return None
    label = " ".join(value.split())
    valid = 0 < len(label.split()) <= MAX_LABEL_WORDS
    return label if valid and len(label) <= MAX_LABEL_CHARS else None


async def _buttons(
    session: Session, turn: models.Turn, labels: Any
) -> list[dict]:
    """Logs a turn's buttons once, with their labels; later calls get them.

    A label the assistant did not write, or wrote wrong, falls back to the
    intent's label from the locale catalog; a next question without a label
    is dropped.
    """
    shown = list(
        await session.scalars(
            sa.select(models.ButtonShown)
            .where(models.ButtonShown.turn_id == turn.id)
            .order_by(models.ButtonShown.position)
        )
    )
    if not shown:
        written = iter(labels if isinstance(labels, list) else [])
        planned = [
            (b, b.get("label") or _label(next(written, None)))
            for b in engine.style_of(turn)["buttons"]
        ]
        shown = [
            models.ButtonShown(
                turn_id=turn.id,
                button=b["button"],
                position=position,
                shuffled=b["shuffled"],
                thompson_draw=b["draw"],
                label=label,
                question_id=b.get("question"),
            )
            for position, (b, label) in enumerate(
                # A next question has no generic fallback.
                (b, label)
                for b, label in planned
                if label or b["button"] != vocab.Button.RELATED
            )
        ]
        session.add_all(shown)
    texts = i18n.catalog(turn.language)["buttons"]
    return [
        {
            "id": s.button,
            "position": s.position,
            **(
                {"label": s.label, "message": s.label}
                if s.label
                else texts[s.button]
            ),
        }
        for s in shown
    ]


async def show_reply(
    caller: Caller, turn_handle: Any, reply: Any, labels: Any = None
) -> dict:
    """Counts a reply's words and returns what the widget needs (SPEC 4.2).

    Args:
        caller: Who calls.
        turn_handle: The turn id returned by plan_reply.
        reply: The reply text; counted, never stored.
        labels: The button labels the assistant wrote, in the order the
            instruction asked for them (SPEC 4.4).

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
        session.add(models.Reply(turn_id=turn.id, reply_words_exact=words))
        user = await session.get(models.User, caller.user_id)
        return {
            "turn_id": turn_handle,
            "buttons": await _buttons(session, turn, labels),
            "labels": i18n.catalog(turn.language)["widget"],
            "profile_url": profile_url(user.profile_key),
        }


async def record_click(
    caller: Caller, turn_handle: Any, button: Any, position: Any
) -> None:
    """Records a click on a shown button (SPEC 4.3)."""
    async with db.session() as session, session.begin():
        turn, _ = await _owned_turn(session, caller, turn_handle)
        shown = await session.scalar(
            sa.select(models.ButtonShown).where(
                models.ButtonShown.turn_id == turn.id,
                models.ButtonShown.button == button,
                models.ButtonShown.position == position,
            )
        )
        if not shown or button not in bandits.EFFECTS:
            raise HandleError
        _log_call(session, caller, "record_click")
        session.add_all(
            [
                models.Click(turn_id=turn.id, button=button, position=position),
                models.Outcome(
                    turn_id=turn.id,
                    source=vocab.Source.BUTTON,
                    observed=bandits.EFFECTS[button].outcome,
                    model_version=turn.model_version,
                    schema_version=config.SCHEMA_VERSION,
                ),
            ]
        )


async def report_context(
    caller: Caller, turn_handle: Any, platform: Any, locale: Any
) -> None:
    """Records the host context the widget saw (SPEC 4.3)."""
    async with db.session() as session, session.begin():
        turn, _ = await _owned_turn(session, caller, turn_handle)
        session.add(
            models.TurnContext(
                turn_id=turn.id,
                platform=platform if platform in vocab.Platform else None,
                locale=locale if isinstance(locale, str) else None,
            )
        )
