"""Tool logic, independent of the MCP layer (SPEC 4)."""

import dataclasses
import hashlib
import json
import random
from typing import Any

from app import config
from app import db
from app import engine
from app import i18n
from app import params as params_lib
from app import text
from app import topics
from app import vocab
from app.model import bandits
from app.model import family
from app.model import policy
from app.model import prefs
from app.model import theta

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
    async with db.pool.connection() as conn:
        row = await db.fetchrow(
            conn,
            "SELECT id FROM users WHERE token_hash = %s",
            token_hash(token),
        )
    return row["id"] if row else None


def _parse(handle: Any, prefix: str) -> int:
    if not isinstance(handle, str) or not handle.startswith(prefix):
        raise HandleError
    number = handle.removeprefix(prefix)
    if not number.isdigit():
        raise HandleError
    return int(number)


async def _owned_turn(conn: db.Connection, caller: Caller, handle: Any) -> dict:
    row = await db.fetchrow(
        conn,
        "SELECT t.*, c.grp FROM turns t "
        "JOIN conversations c ON c.id = t.conversation_id WHERE t.id = %s",
        _parse(handle, TURN_PREFIX),
    )
    if not row or row["user_id"] != caller.user_id:
        raise HandleError
    await db.execute(conn, "SELECT pg_advisory_xact_lock(%s)", caller.user_id)
    return row


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
    conn: db.Connection,
    caller: Caller,
    handle: str | None,
    assistant_model: str | None,
) -> dict:
    if handle is not None:
        row = await db.fetchrow(
            conn,
            "SELECT * FROM conversations WHERE id = %s",
            _parse(handle, CONVERSATION_PREFIX),
        )
        if not row or row["user_id"] != caller.user_id:
            raise HandleError
        return row
    return await db.fetchrow(
        conn,
        "INSERT INTO conversations (user_id, grp, host, ui_supported, "
        "assistant_model, protocol_version) VALUES (%s, %s, %s, %s, %s, %s) "
        "RETURNING *",
        caller.user_id,
        _arm(caller.user_id),
        caller.host,
        caller.ui,
        assistant_model,
        caller.protocol_version,
    )


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


async def _log_call(conn: db.Connection, caller: Caller, tool: str) -> None:
    await db.execute(
        conn,
        "INSERT INTO tool_calls (user_id, tool) VALUES (%s, %s)",
        caller.user_id,
        tool,
    )


async def _close_previous(
    conn: db.Connection, conv: dict, args: dict, now: Any, p: params_lib.Params
) -> dict | None:
    """Records the verdict on the previous turn and closes it."""
    prev = await db.fetchrow(
        conn,
        "SELECT * FROM turns WHERE conversation_id = %s "
        "ORDER BY turn_index DESC LIMIT 1",
        conv["id"],
    )
    if not prev:
        return None
    outcome = args.get("last_reply_outcome") or vocab.Outcome.NO_SIGNAL
    if outcome not in QUOTE_FREE and not (args.get("user_quote") or "").strip():
        outcome = vocab.Outcome.NO_SIGNAL
    issue = args.get("format_issue")
    await db.execute(
        conn,
        "INSERT INTO outcomes (turn_id, source, observed, user_quote, "
        "last_reply_words, format_issue, model_version, schema_version) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
        prev["id"],
        vocab.Source.ASSISTANT,
        outcome,
        args.get("user_quote"),
        args.get("last_reply_words"),
        issue if outcome == vocab.Outcome.WRONG_FORMAT else None,
        prev["model_version"],
        config.SCHEMA_VERSION,
    )
    await engine.finalize_turn(conn, prev["id"], now, p)
    return prev


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
    async with db.pool.connection() as conn, conn.transaction():
        await db.execute(
            conn, "SELECT pg_advisory_xact_lock(%s)", caller.user_id
        )
        await _log_call(conn, caller, "plan_reply")
        p = await params_lib.cache.get(conn)
        now = (await db.fetchrow(conn, "SELECT now() AS now"))["now"]
        conv = await _conversation(
            conn,
            caller,
            args.get("conversation_id"),
            args.get("assistant_model"),
        )
        prev = await _close_previous(conn, conv, args, now, p)
        context = await db.fetchrow(
            conn,
            "SELECT x.platform FROM turn_context x "
            "JOIN turns t ON t.id = x.turn_id "
            "WHERE t.conversation_id = %s ORDER BY x.created_at DESC LIMIT 1",
            conv["id"],
        )
        domain = await topics.domain_of(conn, embedding)
        cell = theta.cell_of(request_type, context and context["platform"])
        theta_grid, _ = await engine.grids(conn, caller.user_id, cell, p)
        state = await engine.conversation_state(conn, conv["id"], p)
        b_mean, b_var = family.bias(state["family_probs"], p.family)
        ctx = policy.Context(
            needed,
            theta_grid,
            b_mean,
            b_var,
            state["kappa_mean"],
            p.hyper.tau,
            settings.q,
        )
        decision, propensity = length_policy(rng).decide(ctx)
        counts = await engine.pref_counts(
            conn, caller.user_id, {request_type, domain}
        )
        treated = conv["grp"] == vocab.Arm.TREATMENT
        clarify = (
            _clarify(counts[request_type])
            if treated and args.get("ambiguous")
            else None
        )
        style = {
            "form": _form(counts, request_type, domain) if treated else {},
            "clarify": clarify,
            "cell": cell,
            "domain": domain,
        }
        turn = await db.fetchrow(
            conn,
            "INSERT INTO turns (user_id, conversation_id, turn_index, "
            "reply_mode, model_version, schema_version, raw_payload, "
            "request_type, topic, needed_words, language, user_message_words, "
            "assistant_model, target_words, requested_words, propensity, "
            "theta_sample, b_sample, post_mean, post_sd, style, "
            "transport_session) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, "
            "%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            caller.user_id,
            conv["id"],
            prev["turn_index"] + 1 if prev else 1,
            vocab.ReplyMode.WIDGET if caller.ui else vocab.ReplyMode.FALLBACK,
            length_policy.version,
            config.SCHEMA_VERSION,
            db.Jsonb(args),
            request_type,
            args["topic"],
            needed,
            args["language"],
            args.get("user_message_words"),
            args.get("assistant_model") or conv["assistant_model"],
            decision.target_words if treated else None,
            decision.requested_words if treated else None,
            propensity if treated else None,
            decision.theta_sample,
            decision.b_sample,
            decision.post_mean,
            decision.post_sd,
            db.Jsonb(style),
            caller.transport_session,
        )
        await db.execute(
            conn,
            "INSERT INTO topic_embeddings VALUES (%s, %s, %s)",
            turn["id"],
            topics.MODEL,
            embedding,
        )
        for version in settings.shadow_policies:
            shadow, shadow_p = policy.POLICIES[version](rng).decide(ctx)
            await db.execute(
                conn,
                "INSERT INTO shadow_decisions VALUES (%s, %s, %s, %s, %s)",
                turn["id"],
                version,
                shadow.target_words,
                shadow_p,
                db.Jsonb({"requested_words": shadow.requested_words}),
            )
    return {
        "conversation_id": f"{CONVERSATION_PREFIX}{conv["id"]}",
        "turn_id": f"{TURN_PREFIX}{turn["id"]}",
        "target_words": decision.target_words if treated else None,
        "instruction": _instruction(
            conv["grp"], decision.requested_words, style, caller.ui
        ),
    }


async def _buttons(conn: db.Connection, turn: dict) -> list[dict]:
    """Chooses and logs a turn's buttons once; later calls get the same."""
    shown = await db.fetch(
        conn,
        "SELECT button, position FROM buttons_shown WHERE turn_id = %s "
        "ORDER BY position",
        turn["id"],
    )
    if not shown and turn["grp"] == vocab.Arm.TREATMENT:
        counts = await engine.pref_counts(
            conn, turn["user_id"], [turn["request_type"]]
        )
        for s in bandits.choose(counts[turn["request_type"]], rng):
            await db.execute(
                conn,
                "INSERT INTO buttons_shown VALUES (%s, %s, %s, %s, %s)",
                turn["id"],
                s.button,
                s.position,
                s.shuffled,
                s.thompson_draw,
            )
            shown.append({"button": s.button, "position": s.position})
    texts = i18n.catalog(turn["language"])["buttons"]
    return [
        {"id": s["button"], "position": s["position"], **texts[s["button"]]}
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
    async with db.pool.connection() as conn, conn.transaction():
        turn = await _owned_turn(conn, caller, turn_handle)
        await _log_call(conn, caller, "show_reply")
        await db.execute(
            conn,
            "INSERT INTO replies (turn_id, reply_words_exact) VALUES (%s, %s)",
            turn["id"],
            words,
        )
        user = await db.fetchrow(
            conn, "SELECT token_hash FROM users WHERE id = %s", caller.user_id
        )
        return {
            "turn_id": turn_handle,
            "buttons": await _buttons(conn, turn),
            "labels": i18n.catalog(turn["language"])["widget"],
            "profile_url": profile_url(user["token_hash"]),
        }


async def record_click(
    caller: Caller, turn_handle: Any, button: Any, position: Any
) -> None:
    """Records a click on a shown button (SPEC 4.3)."""
    async with db.pool.connection() as conn, conn.transaction():
        turn = await _owned_turn(conn, caller, turn_handle)
        shown = await db.fetchrow(
            conn,
            "SELECT 1 FROM buttons_shown "
            "WHERE turn_id = %s AND button = %s AND position = %s",
            turn["id"],
            button,
            position,
        )
        if not shown:
            raise HandleError
        await _log_call(conn, caller, "record_click")
        await db.execute(
            conn,
            "INSERT INTO clicks (turn_id, button, position) "
            "VALUES (%s, %s, %s)",
            turn["id"],
            button,
            position,
        )
        await db.execute(
            conn,
            "INSERT INTO outcomes (turn_id, source, observed, model_version, "
            "schema_version) VALUES (%s, %s, %s, %s, %s)",
            turn["id"],
            vocab.Source.BUTTON,
            bandits.EFFECTS[button].outcome,
            turn["model_version"],
            config.SCHEMA_VERSION,
        )


async def report_context(
    caller: Caller, turn_handle: Any, platform: Any, locale: Any
) -> None:
    """Records the host context the widget saw (SPEC 4.3)."""
    async with db.pool.connection() as conn, conn.transaction():
        turn = await _owned_turn(conn, caller, turn_handle)
        await db.execute(
            conn,
            "INSERT INTO turn_context (turn_id, platform, locale) "
            "VALUES (%s, %s, %s)",
            turn["id"],
            platform if platform in vocab.Platform else None,
            locale if isinstance(locale, str) else None,
        )
