"""Derived-state updates. The online path and the nightly replay both go through finalize_turn,
which reads only events recorded up to the moment the turn ended, so replay reproduces online state."""

import math
from datetime import datetime, timedelta

from app.db import AsyncConnection, Jsonb, execute, fetch, fetchrow
from app.model import family, length, reading, theta
from app.model.bandits import STYLE_BUTTON
from app.model.reaction import BUTTON_OUTCOME, NO_INFO, OUTCOMES, Observation, posterior
from app.model.stats import normalize
from app.params import DEFAULT_REACTION_PRIOR, DEFAULT_READING_WEIGHTS, RHO_PRIOR, Params

KAPPA_PRIOR_VAR = 0.3**2
LENGTH_PRIOR_VAR = 1.0


async def conversation_state(conn: AsyncConnection, conversation_id: int, params: Params) -> dict:
    row = await fetchrow(conn, "SELECT * FROM conversation_state WHERE conversation_id = %s", conversation_id)
    if row:
        return row
    conv = await fetchrow(conn, "SELECT host, assistant_model FROM conversations WHERE id = %s", conversation_id)
    probs = family.initial_probs(params.host_family_prior(conv["host"]),
                                 family.reported_family(conv["assistant_model"]), params.reliability)
    speed_mean, speed_var = family.speed_prior(probs, params.family)
    return {"conversation_id": conversation_id, "family_probs": probs,
            "kappa_mean": family.mix(probs, params.family, "kappa"), "kappa_var": KAPPA_PRIOR_VAR,
            "speed_mean": speed_mean, "speed_var": speed_var}


async def user_reading(conn: AsyncConnection, user_id: int) -> dict:
    row = await fetchrow(conn, "SELECT * FROM user_reading WHERE user_id = %s", user_id)
    return row or {"user_id": user_id, "rho_mean": RHO_PRIOR[0], "rho_var": RHO_PRIOR[1],
                   "mixture": list(DEFAULT_READING_WEIGHTS), "reaction_prior": list(DEFAULT_REACTION_PRIOR)}


async def grids(conn: AsyncConnection, user_id: int, request_type: str, params: Params) -> tuple[list, list]:
    a = await fetchrow(conn, "SELECT grid FROM alpha_posteriors WHERE user_id = %s", user_id)
    alpha = a["grid"] if a else theta.alpha_prior(params.hyper)
    t = await fetchrow(conn, "SELECT grid FROM theta_posteriors WHERE user_id = %s AND request_type = %s",
                       user_id, request_type)
    return (t["grid"] if t else theta.new_theta_cell(alpha, request_type, params.hyper)), alpha


async def beta_counts(conn: AsyncConnection, user_id: int, request_type: str) -> dict[str, tuple[float, float]]:
    rows = await fetch(conn, "SELECT key, alpha, beta FROM beta_counts WHERE user_id = %s AND request_type = %s",
                       user_id, request_type)
    return {r["key"]: (r["alpha"], r["beta"]) for r in rows}


async def _bump(conn: AsyncConnection, user_id: int, request_type: str, key: str, success: bool) -> None:
    await execute(conn, """
        INSERT INTO beta_counts VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, request_type, key)
        DO UPDATE SET alpha = beta_counts.alpha + EXCLUDED.alpha - 1, beta = beta_counts.beta + EXCLUDED.beta - 1""",
        user_id, request_type, key, 2.0 if success else 1.0, 1.0 if success else 2.0)


def observations(verdict: str | None, family_probs: dict, click: dict | None, any_shown: bool,
                 reading_probs: list[float] | None) -> list[Observation]:
    """Annotator observations of one turn. A click makes the assistant verdict dependent, so it is dropped."""
    obs = []
    if verdict and verdict != "no_signal" and click is None:
        obs.append(Observation("assistant", {verdict: 1.0}, dict(family_probs)))
    if any_shown:
        token = f"{click['position']}:{BUTTON_OUTCOME[click['button']]}" if click else "no_click"
        obs.append(Observation("button", {token: 1.0}))
    if reading_probs:
        obs.append(Observation("reading", {"read": reading_probs[0], "skim": reading_probs[1], NO_INFO: reading_probs[2]}))
    return obs


async def turn_events(conn: AsyncConnection, turn_id: int, at: datetime) -> dict:
    """Events of one turn recorded up to `at`."""
    one = "SELECT * FROM {} WHERE turn_id = %s AND created_at <= %s {} ORDER BY created_at LIMIT 1"
    return {
        "turn": await fetchrow(conn, "SELECT * FROM turns WHERE id = %s", turn_id),
        "reply": await fetchrow(conn, one.format("replies", ""), turn_id, at),
        "verdict": await fetchrow(conn, one.format("outcomes", "AND source = 'assistant'"), turn_id, at),
        "click": await fetchrow(conn, one.format("clicks", ""), turn_id, at),
        "shown": await fetch(conn, "SELECT button, position FROM buttons_shown WHERE turn_id = %s", turn_id),
    }


async def finalize_turn(conn: AsyncConnection, turn_id: int, at: datetime, params: Params) -> None:
    """Close a turn when the next plan_reply of its conversation arrives at `at`."""
    ev = await turn_events(conn, turn_id, at)
    turn, reply, verdict, click, shown = ev["turn"], ev["reply"], ev["verdict"], ev["click"], ev["shown"]
    user_id, request_type = turn["user_id"], turn["request_type"]
    conv = await conversation_state(conn, turn["conversation_id"], params)

    # Length (6.2) and assistant behaviour (6.3).
    duration = (reply["created_at"] - turn["created_at"]).total_seconds() if reply else None
    writing = duration - params.overhead if duration is not None else None
    prior = length.Gaussian3.independent(
        [math.log(turn["needed_words"]), conv["kappa_mean"], conv["speed_mean"]],
        [LENGTH_PRIOR_VAR, conv["kappa_var"], conv["speed_var"]])
    post = length.fuse(prior, length.Sensors(
        exact_words=reply["reply_words_exact"] if reply else None,
        self_report_words=verdict["last_reply_words"] if verdict else None,
        writing_seconds=writing, requested_words=turn["requested_words"]), params.noise)
    log_len, len_var = post.mean[0], post.cov[0][0]
    if writing and writing > 0:
        conv["family_probs"] = family.update_with_speed(conv["family_probs"], log_len - math.log(writing), params.family)
    conv.update(kappa_mean=post.mean[1], kappa_var=post.cov[1][1], speed_mean=post.mean[2], speed_var=post.cov[2][2])

    # Reading behaviour (6.6).
    ur = await user_reading(conn, user_id)
    reply_end = reply["created_at"] if reply else turn["created_at"] + timedelta(
        seconds=math.exp(log_len - conv["speed_mean"]) + params.overhead)
    reading_s = max((at - reply_end).total_seconds(), 1.0)
    resp = reading.responsibilities(math.log(reading_s), log_len - ur["rho_mean"], normalize(ur["mixture"]), params.reading)
    ur["rho_mean"], ur["rho_var"] = reading.update_rho(ur["rho_mean"], ur["rho_var"], math.log(reading_s), log_len,
                                                       len_var, resp[0], params.reading)
    ur["mixture"] = [m + r for m, r in zip(ur["mixture"], resp)]

    # Reaction (6.4), then the censored length update (6.5).
    obs = observations(verdict["observed"] if verdict else None, conv["family_probs"], click, bool(shown), resp)
    reaction = posterior(normalize(ur["reaction_prior"]), obs, params.confusions)
    ur["reaction_prior"] = [c + p for c, p in zip(ur["reaction_prior"], reaction)]
    theta_grid, alpha_grid = await grids(conn, user_id, request_type, params)
    b, b_var = family.bias(conv["family_probs"], params.family)
    theta_grid, alpha_grid = theta.update(theta_grid, alpha_grid, request_type, log_len - math.log(turn["needed_words"]),
                                          len_var, b, b_var, reaction, params.hyper)

    # Buttons (8) and style (7): shown but not clicked counts as a failure.
    shown_buttons = {s["button"] for s in shown}
    for button in shown_buttons:
        await _bump(conn, user_id, request_type, button, bool(click) and click["button"] == button)
    for style, button in STYLE_BUTTON.items():
        if button in shown_buttons:
            await _bump(conn, user_id, request_type, "style:" + style, bool(click) and click["button"] == button)

    await execute(conn, """
        INSERT INTO conversation_state VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (conversation_id) DO UPDATE SET
        family_probs = EXCLUDED.family_probs, kappa_mean = EXCLUDED.kappa_mean, kappa_var = EXCLUDED.kappa_var,
        speed_mean = EXCLUDED.speed_mean, speed_var = EXCLUDED.speed_var""",
        conv["conversation_id"], Jsonb(conv["family_probs"]), conv["kappa_mean"], conv["kappa_var"],
        conv["speed_mean"], conv["speed_var"])
    await execute(conn, """
        INSERT INTO user_reading VALUES (%s, %s, %s, %s, %s) ON CONFLICT (user_id) DO UPDATE SET
        rho_mean = EXCLUDED.rho_mean, rho_var = EXCLUDED.rho_var, mixture = EXCLUDED.mixture,
        reaction_prior = EXCLUDED.reaction_prior""",
        user_id, ur["rho_mean"], ur["rho_var"], Jsonb(ur["mixture"]), Jsonb(ur["reaction_prior"]))
    await execute(conn, """
        INSERT INTO theta_posteriors VALUES (%s, %s, %s)
        ON CONFLICT (user_id, request_type) DO UPDATE SET grid = EXCLUDED.grid""", user_id, request_type, theta_grid)
    await execute(conn, """
        INSERT INTO alpha_posteriors VALUES (%s, %s) ON CONFLICT (user_id) DO UPDATE SET grid = EXCLUDED.grid""",
        user_id, alpha_grid)
    await execute(conn, """
        INSERT INTO turn_inference VALUES (%s, %s, %s, %s, %s) ON CONFLICT (turn_id) DO UPDATE SET
        length_mean = EXCLUDED.length_mean, length_var = EXCLUDED.length_var,
        reaction_probs = EXCLUDED.reaction_probs, reading_probs = EXCLUDED.reading_probs""",
        turn_id, log_len, len_var, Jsonb(dict(zip(OUTCOMES, reaction))),
        Jsonb({"read": resp[0], "skim": resp[1], "away": resp[2], "seconds": reading_s}))
