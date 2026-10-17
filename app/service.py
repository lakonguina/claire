"""Tool logic, independent of the MCP layer (SPEC 4)."""

import dataclasses
import hashlib
import json
import random
from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import db
from app import engine
from app import i18n
from app import models
from app import params as params_lib
from app import text
from app import topics
from app import vocab
from app.model import bandits
from app.model import family
from app.model import policy
from app.model import prefs
from app.model import theta

Session = sa_asyncio.AsyncSession
CONVERSATION_PREFIX = "c_"
TURN_PREFIX = "t_"
QUOTE_FREE = frozenset({vocab.Outcome.MOVED_ON, vocab.Outcome.NO_SIGNAL})
MIN_NEEDED, MAX_NEEDED = 5, 2000
rng = random.Random()


class HandleError(Exception):
    """A handle that is malformed or belongs to another user."""


@dataclasses.dataclass(frozen=True)
class Caller:
    """Who calls a tool, and through which host."""

    user_id: int
    host: str
    ui: bool
    protocol_version: str | None = None
    transport_session: str | None = None


def token_hash(token: str) -> str:
    """Returns the stored hash of a user token."""
    return hashlib.sha256(token.encode()).hexdigest()


def profile_url(key: str) -> str:
    """Returns the profile page URL of a user's token hash (SPEC 4.6)."""
    return f"{config.settings.public_url.rstrip("/")}/me/{key}"


def compact(data: dict) -> str:
    """Serializes a tool result without any whitespace (SPEC 3)."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


async def user_for_token(token: str) -> int | None:
    """Returns the user behind a connector token, or None."""
    async with db.session() as session:
        return await session.scalar(
            sa.select(models.User.id).where(
                models.User.token_hash == token_hash(token)
            )
        )


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
        return conv
    conv = models.Conversation(
        user_id=caller.user_id,
        grp=_arm(caller.user_id),
        host=caller.host,
        ui_supported=caller.ui,
        assistant_model=args.get("assistant_model"),
        protocol_version=caller.protocol_version,
    )
    session.add(conv)
    await session.flush()
    return conv


def _form(
    counts: dict[str, prefs.Counts], request_type: str, domain: str
) -> dict[str, str]:
    """Draws a value for every form dimension (SPEC 7)."""
    return {
        d: prefs.thompson(
            counts[prefs.scope_of(d, request_type, domain)], d, rng
        )
        for d in vocab.Dimension
    }


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


async def _platform(session: Session, conv: models.Conversation) -> str | None:
    """Returns the conversation's latest platform reported by the widget."""
    return await session.scalar(
        sa.select(models.TurnContext.platform)
        .join(models.Turn, models.Turn.id == models.TurnContext.turn_id)
        .where(models.Turn.conversation_id == conv.id)
        .order_by(models.TurnContext.created_at.desc())
        .limit(1)
    )


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
    request_type = args["request_type"]
    length_policy = policy.POLICIES[settings.policy]
    async with db.session() as session, session.begin():
        await _lock(session, caller)
        _log_call(session, caller, "plan_reply")
        p = await params_lib.cache.get(session)
        now = await session.scalar(sa.select(sa.func.now()))
        conv = await _conversation(session, caller, args)
        prev = await _close_previous(session, conv, args, now, p)
        domain = await topics.domain_of(session, embedding)
        cell = theta.cell_of(request_type, await _platform(session, conv))
        cell_theta, _ = await engine.grids(session, caller.user_id, cell, p)
        state = await engine.conversation_state(session, conv.id, p)
        b_mean, b_var = family.bias(state.family_probs, p.family)
        ctx = policy.Context(
            needed,
            cell_theta.grid,
            b_mean,
            b_var,
            state.kappa_mean,
            p.hyper.tau,
            settings.q,
        )
        decision, propensity = length_policy(rng).decide(ctx)
        counts = await engine.pref_counts(
            session, caller.user_id, {request_type, domain}
        )
        treated = conv.grp == vocab.Arm.TREATMENT
        style = {
            "form": _form(counts, request_type, domain) if treated else {},
            "clarify": (
                _clarify(counts[request_type])
                if treated and args.get("ambiguous")
                else None
            ),
            "cell": cell,
            "domain": domain,
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
            request_type=request_type,
            topic=args["topic"],
            needed_words=needed,
            language=args["language"],
            user_message_words=args.get("user_message_words"),
            assistant_model=args.get("assistant_model") or conv.assistant_model,
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


async def _buttons(
    session: Session, turn: models.Turn, conv: models.Conversation
) -> list[dict]:
    """Chooses and logs a turn's buttons once; later calls get the same."""
    shown = list(
        await session.scalars(
            sa.select(models.ButtonShown)
            .where(models.ButtonShown.turn_id == turn.id)
            .order_by(models.ButtonShown.position)
        )
    )
    if not shown and conv.grp == vocab.Arm.TREATMENT:
        counts = await engine.pref_counts(
            session, turn.user_id, [turn.request_type]
        )
        shown = [
            models.ButtonShown(
                turn_id=turn.id,
                button=s.button,
                position=s.position,
                shuffled=s.shuffled,
                thompson_draw=s.thompson_draw,
            )
            for s in bandits.choose(counts[turn.request_type], rng)
        ]
        session.add_all(shown)
    texts = i18n.catalog(turn.language)["buttons"]
    return [
        {"id": s.button, "position": s.position, **texts[s.button]}
        for s in shown
    ]


async def show_reply(caller: Caller, turn_handle: Any, reply: Any) -> dict:
    """Counts a reply's words and returns what the widget needs (SPEC 4.2).

    Args:
        caller: Who calls.
        turn_handle: The turn id returned by plan_reply.
        reply: The reply text; counted, never stored.

    Returns:
        The widget's structured content.

    Raises:
        HandleError: On an invalid reply or turn.
    """
    if not isinstance(reply, str):
        raise HandleError
    words = text.count_words(reply)
    async with db.session() as session, session.begin():
        turn, conv = await _owned_turn(session, caller, turn_handle)
        _log_call(session, caller, "show_reply")
        session.add(models.Reply(turn_id=turn.id, reply_words_exact=words))
        user = await session.get(models.User, caller.user_id)
        return {
            "turn_id": turn_handle,
            "buttons": await _buttons(session, turn, conv),
            "labels": i18n.catalog(turn.language)["widget"],
            "profile_url": profile_url(user.token_hash),
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
