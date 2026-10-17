"""Derived-state updates (SPEC 6 and 7).

The live service and the nightly replay both go through finalize_turn and
apply_edit, which read only events recorded up to the moment they happened,
so a replay reproduces the live state.
"""

import collections
from collections.abc import Iterable
import datetime
import math

from app import db
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

KAPPA_PRIOR_VAR = 0.3**2
LENGTH_PRIOR_VAR = 1.0


async def conversation_state(
    conn: db.Connection, conversation_id: int, p: params_lib.Params
) -> dict:
    """Returns the conversation's assistant state, or its prior."""
    row = await db.fetchrow(
        conn,
        "SELECT * FROM conversation_state WHERE conversation_id = %s",
        conversation_id,
    )
    if row:
        return row
    conv = await db.fetchrow(
        conn,
        "SELECT host, assistant_model FROM conversations WHERE id = %s",
        conversation_id,
    )
    probs = family.initial_probs(
        p.host_family_prior(conv["host"]),
        family.reported_family(conv["assistant_model"]),
        p.reliability,
    )
    speed_mean, speed_var = family.speed_prior(probs, p.family)
    return {
        "conversation_id": conversation_id,
        "family_probs": probs,
        "kappa_mean": family.mix(probs, p.family, "kappa"),
        "kappa_var": KAPPA_PRIOR_VAR,
        "speed_mean": speed_mean,
        "speed_var": speed_var,
    }


async def user_reading(conn: db.Connection, user_id: int) -> dict:
    """Returns the user's reading state, or its prior."""
    row = await db.fetchrow(
        conn, "SELECT * FROM user_reading WHERE user_id = %s", user_id
    )
    return row or {
        "user_id": user_id,
        "rho_mean": params_lib.RHO_PRIOR[0],
        "rho_var": params_lib.RHO_PRIOR[1],
        "mixture": list(params_lib.READING_PRIOR),
        "reaction_prior": list(params_lib.REACTION_PRIOR),
    }


async def grids(
    conn: db.Connection, user_id: int, cell: str, p: params_lib.Params
) -> tuple[list[float], list[float]]:
    """Returns the cell's theta grid and the user's alpha grid."""
    a = await db.fetchrow(
        conn, "SELECT grid FROM alpha_posteriors WHERE user_id = %s", user_id
    )
    alpha = a["grid"] if a else theta.alpha_prior(p.hyper)
    t = await db.fetchrow(
        conn,
        "SELECT grid FROM theta_posteriors "
        "WHERE user_id = %s AND request_type = %s",
        user_id,
        cell,
    )
    return (t["grid"] if t else theta.new_cell(alpha, cell, p.hyper)), alpha


async def pref_counts(
    conn: db.Connection, user_id: int, scopes: Iterable[str]
) -> dict[str, prefs.Counts]:
    """Returns, for each scope, the counts of its cell, user and population.

    Args:
        conn: The connection.
        user_id: The user.
        scopes: The cell scopes needed.

    Returns:
        Scope -> counts keyed by (level, dimension, value).
    """
    scopes = list(scopes)
    rows = await db.fetch(
        conn,
        "SELECT user_id, scope, dimension, value, count FROM pref_counts "
        "WHERE (user_id = %s AND scope = ANY(%s)) "
        "OR (user_id = %s AND scope = %s)",
        user_id,
        [*scopes, prefs.USER_SCOPE],
        prefs.POPULATION,
        prefs.USER_SCOPE,
    )
    out = {s: {} for s in scopes}
    for r in rows:
        key = (r["dimension"], r["value"])
        if r["user_id"] == prefs.POPULATION:
            for counts in out.values():
                counts[("population", *key)] = r["count"]
            continue
        for scope, counts in out.items():
            if r["scope"] == scope:
                counts[("cell", *key)] = r["count"]
            if r["scope"] == prefs.USER_SCOPE:
                counts[("user", *key)] = r["count"]
    return out


async def add_prefs(
    conn: db.Connection,
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
    for (dimension, value), count in counts.items():
        for level_user, level_scope in levels:
            await db.execute(
                conn,
                "INSERT INTO pref_counts VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (user_id, scope, dimension, value) "
                "DO UPDATE SET count = pref_counts.count + EXCLUDED.count",
                level_user,
                level_scope,
                dimension,
                value,
                count,
            )


def observations(
    verdict: str | None,
    family_probs: dict,
    click: dict | None,
    any_shown: bool,
    reading_probs: list[float] | None,
) -> list[reaction.Observation]:
    """Returns the annotator observations of one turn.

    A click makes the assistant verdict dependent (SPEC 6.4), so the verdict
    is dropped when there is one.

    Args:
        verdict: The assistant verdict, if any.
        family_probs: The conversation's model-family probabilities.
        click: The turn's click, if any.
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
    if any_shown:
        token = (
            f"{click["position"]}:{bandits.EFFECTS[click["button"]].outcome}"
            if click
            else "no_click"
        )
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


async def _turn_events(
    conn: db.Connection, turn_id: int, at: datetime.datetime
) -> dict:
    """Returns one turn and its events recorded up to `at`."""
    first = (
        "SELECT * FROM {} WHERE turn_id = %s AND created_at <= %s "
        "ORDER BY created_at LIMIT 1"
    )
    embedding = await db.fetchrow(
        conn,
        "SELECT embedding FROM topic_embeddings WHERE turn_id = %s",
        turn_id,
    )
    return {
        "turn": await db.fetchrow(
            conn, "SELECT * FROM turns WHERE id = %s", turn_id
        ),
        "reply": await db.fetchrow(conn, first.format("replies"), turn_id, at),
        "verdict": await db.fetchrow(
            conn,
            "SELECT * FROM outcomes WHERE turn_id = %s AND created_at <= %s "
            "AND source = %s ORDER BY created_at LIMIT 1",
            turn_id,
            at,
            vocab.Source.ASSISTANT,
        ),
        "click": await db.fetchrow(conn, first.format("clicks"), turn_id, at),
        "shown": await db.fetch(
            conn,
            "SELECT button, position FROM buttons_shown WHERE turn_id = %s",
            turn_id,
        ),
        "domain": (
            await topics.domain_of(conn, embedding["embedding"])
            if embedding
            else topics.ANY_DOMAIN
        ),
    }


def style_of(turn: dict) -> dict:
    """Returns the decisions logged on a turn (SPEC 5.5)."""
    style = turn["style"] if isinstance(turn["style"], dict) else {}
    return {"form": {}, "clarify": None, "cell": turn["request_type"], **style}


async def finalize_turn(
    conn: db.Connection,
    turn_id: int,
    at: datetime.datetime,
    p: params_lib.Params,
) -> None:
    """Closes a turn when the next plan_reply of its conversation arrives.

    Args:
        conn: The connection, inside the caller's transaction.
        turn_id: The turn to close.
        at: When the next plan_reply arrived.
        p: The global parameters.
    """
    ev = await _turn_events(conn, turn_id, at)
    turn, reply, verdict, click = (
        ev["turn"],
        ev["reply"],
        ev["verdict"],
        ev["click"],
    )
    user_id, style = turn["user_id"], style_of(turn)
    conv = await conversation_state(conn, turn["conversation_id"], p)

    # Length (6.2) and assistant behaviour (6.3).
    duration = (
        (reply["created_at"] - turn["created_at"]).total_seconds()
        if reply
        else None
    )
    writing = duration - p.overhead if duration is not None else None
    prior = length.Gaussian3.independent(
        [
            math.log(turn["needed_words"]),
            conv["kappa_mean"],
            conv["speed_mean"],
        ],
        [LENGTH_PRIOR_VAR, conv["kappa_var"], conv["speed_var"]],
    )
    post = length.fuse(
        prior,
        length.Sensors(
            exact_words=reply["reply_words_exact"] if reply else None,
            self_report_words=verdict["last_reply_words"] if verdict else None,
            writing_seconds=writing,
            requested_words=turn["requested_words"],
        ),
        p.noise,
    )
    log_len, len_var = post.mean[0], post.cov[0][0]
    if writing and writing > 0:
        conv["family_probs"] = family.update_with_speed(
            conv["family_probs"], log_len - math.log(writing), p.family
        )
    conv.update(
        kappa_mean=post.mean[1],
        kappa_var=post.cov[1][1],
        speed_mean=post.mean[2],
        speed_var=post.cov[2][2],
    )

    # Reading behaviour (6.6).
    ur = await user_reading(conn, user_id)
    reply_end = (
        reply["created_at"]
        if reply
        else turn["created_at"]
        + datetime.timedelta(
            seconds=math.exp(log_len - conv["speed_mean"]) + p.overhead
        )
    )
    log_t = math.log(max((at - reply_end).total_seconds(), 1.0))
    resp = reading.responsibilities(
        log_t,
        log_len - ur["rho_mean"],
        stats.normalize(ur["mixture"]),
        p.reading,
    )
    ur["rho_mean"], ur["rho_var"] = reading.update_rho(
        (ur["rho_mean"], ur["rho_var"]),
        log_t,
        (log_len, len_var),
        resp[0],
        p.reading,
    )
    ur["mixture"] = [m + r for m, r in zip(ur["mixture"], resp)]

    # Reaction (6.4), then the censored length update (6.5).
    obs = observations(
        verdict["observed"] if verdict else None,
        conv["family_probs"],
        click,
        bool(ev["shown"]),
        resp,
    )
    outcome_probs = reaction.posterior(
        stats.normalize(ur["reaction_prior"]), obs, p.confusions
    )
    ur["reaction_prior"] = [
        c + q for c, q in zip(ur["reaction_prior"], outcome_probs)
    ]
    theta_grid, alpha_grid = await grids(conn, user_id, style["cell"], p)
    theta_grid, alpha_grid = theta.update(
        theta_grid,
        alpha_grid,
        style["cell"],
        log_len - math.log(turn["needed_words"]),
        len_var,
        family.bias(conv["family_probs"], p.family),
        outcome_probs,
        p.hyper,
    )

    # Form preferences (7), buttons (8) and clarification (4.1).
    probs = dict(zip(reaction.OUTCOMES, outcome_probs))
    await _learn_preferences(conn, ev, style, probs)

    await _save(
        conn,
        turn_id,
        conv,
        ur,
        style["cell"],
        theta_grid,
        alpha_grid,
        (log_len, len_var),
        outcome_probs,
        resp,
        log_t,
    )


async def _learn_preferences(
    conn: db.Connection, ev: dict, style: dict, probs: dict[str, float]
) -> None:
    turn, click, verdict = ev["turn"], ev["click"], ev["verdict"]
    user_id, request_type = turn["user_id"], turn["request_type"]
    explicit = []
    if click and bandits.EFFECTS[click["button"]].preference:
        explicit.append(bandits.EFFECTS[click["button"]].preference)
    elif not click and verdict and verdict["format_issue"]:
        explicit.append(vocab.FORMAT_ISSUES[verdict["format_issue"]])
    by_scope = collections.defaultdict(dict)
    for (dimension, value), count in prefs.signals(
        style["form"],
        explicit,
        probs[vocab.Outcome.MOVED_ON],
        probs[vocab.Outcome.WRONG_FORMAT],
    ).items():
        scope = prefs.scope_of(dimension, request_type, ev["domain"])
        by_scope[scope][(dimension, value)] = count
    for shown in ev["shown"]:
        clicked = bool(click) and click["button"] == shown["button"]
        dimension = prefs.arm(vocab.ArmKind.BUTTON, shown["button"])
        value = vocab.PrefValue.YES if clicked else vocab.PrefValue.NO
        by_scope[request_type][(dimension, value)] = 1.0
    if style["clarify"]:
        dimension = prefs.arm(vocab.ArmKind.CLARIFY, style["clarify"])
        moved_on = probs[vocab.Outcome.MOVED_ON]
        by_scope[request_type][(dimension, vocab.PrefValue.YES)] = moved_on
        by_scope[request_type][(dimension, vocab.PrefValue.NO)] = 1 - moved_on
    for scope, counts in by_scope.items():
        await add_prefs(conn, user_id, scope, counts)


async def _save(
    conn: db.Connection,
    turn_id: int,
    conv: dict,
    ur: dict,
    cell: str,
    theta_grid: list[float],
    alpha_grid: list[float],
    fused: tuple[float, float],
    outcome_probs: list[float],
    resp: list[float],
    log_t: float,
) -> None:
    user_id = ur["user_id"]
    await db.execute(
        conn,
        "INSERT INTO conversation_state VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (conversation_id) DO UPDATE SET "
        "family_probs = EXCLUDED.family_probs, "
        "kappa_mean = EXCLUDED.kappa_mean, "
        "kappa_var = EXCLUDED.kappa_var, speed_mean = EXCLUDED.speed_mean, "
        "speed_var = EXCLUDED.speed_var",
        conv["conversation_id"],
        db.Jsonb(conv["family_probs"]),
        conv["kappa_mean"],
        conv["kappa_var"],
        conv["speed_mean"],
        conv["speed_var"],
    )
    await db.execute(
        conn,
        "INSERT INTO user_reading VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET rho_mean = EXCLUDED.rho_mean, "
        "rho_var = EXCLUDED.rho_var, mixture = EXCLUDED.mixture, "
        "reaction_prior = EXCLUDED.reaction_prior",
        user_id,
        ur["rho_mean"],
        ur["rho_var"],
        db.Jsonb(ur["mixture"]),
        db.Jsonb(ur["reaction_prior"]),
    )
    await save_theta(conn, user_id, cell, theta_grid)
    await db.execute(
        conn,
        "INSERT INTO alpha_posteriors VALUES (%s, %s) "
        "ON CONFLICT (user_id) DO UPDATE SET grid = EXCLUDED.grid",
        user_id,
        alpha_grid,
    )
    reading_probs = dict(zip(reading.COMPONENTS, resp))
    await db.execute(
        conn,
        "INSERT INTO turn_inference VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (turn_id) DO UPDATE SET "
        "length_mean = EXCLUDED.length_mean, "
        "length_var = EXCLUDED.length_var, "
        "reaction_probs = EXCLUDED.reaction_probs, "
        "reading_probs = EXCLUDED.reading_probs",
        turn_id,
        fused[0],
        fused[1],
        db.Jsonb(dict(zip(reaction.OUTCOMES, outcome_probs))),
        db.Jsonb({**reading_probs, "seconds": math.exp(log_t)}),
    )


async def save_theta(
    conn: db.Connection, user_id: int, cell: str, grid: list[float]
) -> None:
    """Stores a length cell's grid."""
    await db.execute(
        conn,
        "INSERT INTO theta_posteriors VALUES (%s, %s, %s) "
        "ON CONFLICT (user_id, request_type) "
        "DO UPDATE SET grid = EXCLUDED.grid",
        user_id,
        cell,
        grid,
    )


async def apply_edit(
    conn: db.Connection, edit: dict, p: params_lib.Params
) -> None:
    """Applies a profile-page correction (SPEC 4.6), a very strong signal."""
    if edit["dimension"] == vocab.LENGTH:
        grid, _ = await grids(conn, edit["user_id"], edit["scope"], p)
        await save_theta(
            conn,
            edit["user_id"],
            edit["scope"],
            theta.edit(grid, edit["value"]),
        )
        return
    await add_prefs(
        conn,
        edit["user_id"],
        edit["scope"],
        {(edit["dimension"], edit["value"]): prefs.EDIT},
    )
