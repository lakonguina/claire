"""Tool logic, independent of the MCP layer: plan_reply, show_reply, record_click."""

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any

from app import engine
from app.config import SCHEMA_VERSION, settings
from app.db import AsyncConnection, Jsonb, execute, fetch, fetchrow, pool
from app.model.bandits import BUTTONS, STYLES, active_styles, choose_buttons
from app.model.family import bias
from app.model.policy import POLICIES, Context
from app.model.reaction import BUTTON_OUTCOME
from app.params import cached_params
from app.text import count_words, detect_language

QUOTE_FREE = {"moved_on", "no_signal"}
rng = random.Random()


class HandleError(Exception):
    """A conversation or turn handle that is malformed or belongs to another user."""


@dataclass
class Caller:
    user_id: int
    host: str
    ui: bool
    protocol_version: str | None = None
    transport_session: str | None = None


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def user_for_token(token: str) -> int | None:
    async with pool.connection() as conn:
        row = await fetchrow(conn, "SELECT id FROM users WHERE token_hash = %s", token_hash(token))
    return row["id"] if row else None


def _parse(handle: Any, prefix: str) -> int:
    if not isinstance(handle, str) or not handle.startswith(prefix) or not handle[len(prefix):].isdigit():
        raise HandleError
    return int(handle[len(prefix):])


async def _owned_turn(conn: AsyncConnection, caller: Caller, handle: Any) -> dict:
    row = await fetchrow(conn, """SELECT t.*, c.grp FROM turns t JOIN conversations c ON c.id = t.conversation_id
                                  WHERE t.id = %s""", _parse(handle, "t_"))
    if not row or row["user_id"] != caller.user_id:
        raise HandleError
    await execute(conn, "SELECT pg_advisory_xact_lock(%s)", caller.user_id)
    return row


def _group(user_id: int) -> str:
    if settings.control_unit == "user":
        draw = random.Random(f"group:{user_id}").random()
    else:
        draw = rng.random()
    return "control" if draw < settings.control_share else "treatment"


async def _conversation(conn: AsyncConnection, caller: Caller, handle: str | None, assistant_model: str | None) -> dict:
    if handle is not None:
        row = await fetchrow(conn, "SELECT * FROM conversations WHERE id = %s", _parse(handle, "c_"))
        if not row or row["user_id"] != caller.user_id:
            raise HandleError
        return row
    return await fetchrow(conn, """
        INSERT INTO conversations (user_id, grp, host, ui_supported, assistant_model, protocol_version)
        VALUES (%s, %s, %s, %s, %s, %s) RETURNING *""",
        caller.user_id, _group(caller.user_id), caller.host, caller.ui, assistant_model, caller.protocol_version)


def _instruction(requested: int | None, styles: list[str], ui: bool) -> str:
    parts = ["Reply normally."] if requested is None else [
        f"Reply in about {requested} words.", "No preamble, no recap.", *(STYLES[s] for s in styles)]
    if ui:
        parts.append("Write the reply only in show_reply.")
    return " ".join(parts)


async def _log_call(conn: AsyncConnection, caller: Caller, tool: str) -> None:
    await execute(conn, "INSERT INTO tool_calls (user_id, tool) VALUES (%s, %s)", caller.user_id, tool)


async def plan_reply(caller: Caller, args: dict) -> dict:
    outcome = args.get("last_reply_outcome") or "no_signal"
    if outcome not in QUOTE_FREE and not (args.get("user_quote") or "").strip():
        outcome = "no_signal"
    needed = min(max(int(args["needed_words"]), 5), 2000)
    params_policy = POLICIES[settings.policy]
    async with pool.connection() as conn, conn.transaction():
        await execute(conn, "SELECT pg_advisory_xact_lock(%s)", caller.user_id)
        await _log_call(conn, caller, "plan_reply")
        params = await cached_params(conn)
        now = (await fetchrow(conn, "SELECT now() AS now"))["now"]
        conv = await _conversation(conn, caller, args.get("conversation_id"), args.get("assistant_model"))
        prev = await fetchrow(conn, "SELECT * FROM turns WHERE conversation_id = %s ORDER BY turn_index DESC LIMIT 1",
                              conv["id"])
        if prev:
            await execute(conn, """INSERT INTO outcomes (turn_id, source, observed, user_quote, last_reply_words,
                                   model_version, schema_version) VALUES (%s, 'assistant', %s, %s, %s, %s, %s)""",
                          prev["id"], outcome, args.get("user_quote"), args.get("last_reply_words"),
                          prev["model_version"], SCHEMA_VERSION)
            await engine.finalize_turn(conn, prev["id"], now, params)

        request_type = args["request_type"]
        theta_grid, _ = await engine.grids(conn, caller.user_id, request_type, params)
        state = await engine.conversation_state(conn, conv["id"], params)
        b_mean, b_var = bias(state["family_probs"], params.family)
        ctx = Context(needed, theta_grid, b_mean, b_var, state["kappa_mean"], params.hyper.tau, settings.q)
        decision, propensity = params_policy(rng).decide(ctx)
        counts = await engine.beta_counts(conn, caller.user_id, request_type)
        treated = conv["grp"] == "treatment"
        styles = active_styles(counts) if treated else []
        target = decision.target_words if treated else None
        requested = decision.requested_words if treated else None
        turn = await fetchrow(conn, """
            INSERT INTO turns (user_id, conversation_id, turn_index, reply_mode, model_version, schema_version,
                raw_payload, request_type, topic, needed_words, language, user_message_words, pasted_content,
                user_specified, assistant_model, target_words, requested_words, propensity, theta_sample, b_sample,
                post_mean, post_sd, style, transport_session)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id""",
            caller.user_id, conv["id"], (prev["turn_index"] + 1) if prev else 1,
            "widget" if caller.ui else "fallback", params_policy.version, SCHEMA_VERSION, Jsonb(args), request_type,
            args["topic"], needed, detect_language(args["topic"]), args.get("user_message_words"),
            args.get("pasted_content"), args.get("user_specified"), args.get("assistant_model") or conv["assistant_model"],
            target, requested, propensity if treated else None, decision.theta_sample, decision.b_sample,
            decision.post_mean, decision.post_sd, Jsonb(styles), caller.transport_session)
        for version in settings.shadow_policies:
            shadow, shadow_p = POLICIES[version](rng).decide(ctx)
            await execute(conn, "INSERT INTO shadow_decisions VALUES (%s, %s, %s, %s, %s)", turn["id"], version,
                          shadow.target_words, shadow_p, Jsonb({"requested_words": shadow.requested_words}))
    return {"conversation_id": f"c_{conv['id']}", "turn_id": f"t_{turn['id']}", "target_words": target,
            "instruction": _instruction(requested, styles, caller.ui)}


async def _buttons(conn: AsyncConnection, turn: dict) -> list[dict]:
    """Choose and log the buttons of a turn once; later calls return the same ones."""
    shown = await fetch(conn, "SELECT button, position FROM buttons_shown WHERE turn_id = %s ORDER BY position",
                        turn["id"])
    if not shown and turn["grp"] == "treatment":
        clicked = await fetch(conn, """SELECT 1 FROM clicks k JOIN turns t ON t.id = k.turn_id
                                       WHERE t.conversation_id = %s AND k.button = 'rephrase_request'""",
                              turn["conversation_id"])
        counts = await engine.beta_counts(conn, turn["user_id"], turn["request_type"])
        for s in choose_buttons(counts, {"rephrase_request"} if clicked else set(), rng):
            await execute(conn, "INSERT INTO buttons_shown VALUES (%s, %s, %s, %s, %s)",
                          turn["id"], s.button, s.position, s.shuffled, s.thompson_draw)
            shown.append({"button": s.button, "position": s.position})
    return [{"id": s["button"], "label": BUTTONS[s["button"]][0], "message": BUTTONS[s["button"]][1],
             "position": s["position"]} for s in shown]


async def show_reply(caller: Caller, turn_handle: Any, text: Any) -> dict:
    if not isinstance(text, str):
        raise HandleError
    words = count_words(text)
    async with pool.connection() as conn, conn.transaction():
        turn = await _owned_turn(conn, caller, turn_handle)
        await _log_call(conn, caller, "show_reply")
        await execute(conn, "INSERT INTO replies (turn_id, reply_words_exact) VALUES (%s, %s)", turn["id"], words)
        return {"turn_id": turn_handle, "buttons": await _buttons(conn, turn)}


async def record_click(caller: Caller, turn_handle: Any, button: Any, position: Any) -> None:
    async with pool.connection() as conn, conn.transaction():
        turn = await _owned_turn(conn, caller, turn_handle)
        shown = await fetchrow(conn, "SELECT 1 FROM buttons_shown WHERE turn_id = %s AND button = %s AND position = %s",
                               turn["id"], button, position)
        if not shown:
            raise HandleError
        await _log_call(conn, caller, "record_click")
        await execute(conn, "INSERT INTO clicks (turn_id, button, position) VALUES (%s, %s, %s)",
                      turn["id"], button, position)
        await execute(conn, """INSERT INTO outcomes (turn_id, source, observed, model_version, schema_version)
                               VALUES (%s, 'button', %s, %s, %s)""",
                      turn["id"], BUTTON_OUTCOME[button], turn["model_version"], SCHEMA_VERSION)


def compact(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)
