"""Admin dashboard: the page at /admin and its JSON API (SPEC 11)."""

import math
import pathlib

import fastapi
from fastapi import responses

from app import config
from app import db
from app import engine
from app import nightly
from app import params as params_lib
from app import vocab
from app.model import prefs
from app.model import reaction
from app.model import reading as reading_lib
from app.model import stats

PAGE = (pathlib.Path(__file__).resolve().parent / "dashboard.html").read_text()
MAX_TURNS = 500
LEARNED_TABLES = (
    "hyperparams",
    "family_params",
    "sensor_noise",
    "annotator_confusion",
    "self_report_reliability",
    "host_family_prior",
    "topic_clusters",
)
TURN_COLUMNS = """
    t.id, t.created_at, t.user_id, t.conversation_id, t.turn_index, c.host,
    c.grp, t.reply_mode, t.request_type, t.topic, t.language, t.needed_words,
    t.target_words, t.requested_words, t.model_version, t.transport_session,
    t.style, r.reply_words_exact,
    extract(epoch FROM r.created_at - t.created_at)::float8 AS writing_seconds,
    o.observed AS verdict, o.user_quote, o.format_issue, k.button AS clicked,
    i.reaction_probs,
    (SELECT array_agg(b.button ORDER BY b.position) FROM buttons_shown b
     WHERE b.turn_id = t.id) AS buttons
    FROM turns t JOIN conversations c ON c.id = t.conversation_id
    LEFT JOIN LATERAL (SELECT * FROM replies WHERE turn_id = t.id
        ORDER BY created_at LIMIT 1) r ON true
    LEFT JOIN LATERAL (SELECT * FROM outcomes WHERE turn_id = t.id
        AND source = %s ORDER BY created_at LIMIT 1) o ON true
    LEFT JOIN LATERAL (SELECT * FROM clicks WHERE turn_id = t.id
        ORDER BY created_at LIMIT 1) k ON true
    LEFT JOIN turn_inference i ON i.turn_id = t.id"""
FILTERS = {
    "user": "t.user_id = %s",
    "conversation": "t.conversation_id = %s",
    "host": "c.host = %s",
    "mode": "t.reply_mode = %s",
    "group": "c.grp = %s",
    "q": "t.topic ILIKE '%%' || %s || '%%'",
    "before": "t.id < %s",
}
BEARER = "Bearer "


def admin(request: fastapi.Request) -> None:
    """Requires the admin token as a bearer token."""
    supplied = request.headers.get("authorization", "").removeprefix(BEARER)
    token = config.settings.admin_token
    if not token or supplied != token:
        raise fastapi.HTTPException(status_code=401)


page = fastapi.APIRouter()
api = fastapi.APIRouter(
    prefix="/admin/api", dependencies=[fastapi.Depends(admin)]
)


@page.get(
    "/admin", response_class=responses.HTMLResponse, include_in_schema=False
)
async def dashboard_page() -> str:
    return PAGE  # Holds no data: the API below needs the admin token.


def _inferred(row: dict) -> dict:
    probs = row.pop("reaction_probs", None) or {}
    row["inferred"] = max(probs, key=probs.get, default=None)
    row["inferred_p"] = max(probs.values(), default=None)
    return row


@api.get("/overview")
async def overview() -> dict:
    settings = config.settings
    async with db.pool.connection() as conn:
        counts = await db.fetchrow(
            conn,
            """SELECT (SELECT count(*) FROM users) AS users,
               (SELECT count(*) FROM conversations) AS conversations,
               (SELECT count(*) FROM conversations WHERE grp <> %s)
                   AS held_back_conversations,
               (SELECT count(*) FROM turns) AS turns,
               (SELECT count(*) FROM turns
                WHERE created_at > now() - interval '24 hours') AS turns_24h,
               (SELECT count(*) FROM turns WHERE reply_mode = %s)
                   AS widget_turns,
               (SELECT count(DISTINCT r.turn_id) FROM replies r
                JOIN turns t ON t.id = r.turn_id WHERE t.reply_mode = %s)
                   AS widget_turns_with_reply,
               (SELECT count(DISTINCT turn_id) FROM buttons_shown)
                   AS turns_with_buttons,
               (SELECT count(DISTINCT turn_id) FROM clicks) AS turns_with_click,
               (SELECT count(*) FROM turn_inference) AS finalized_turns,
               (SELECT count(*) FROM gold_labels) AS gold_labels,
               (SELECT count(*) FROM topic_clusters) AS topic_clusters,
               (SELECT max(updated_at) FROM hyperparams) AS last_nightly""",
            vocab.Arm.TREATMENT,
            vocab.ReplyMode.WIDGET,
            vocab.ReplyMode.WIDGET,
        )
        hosts = await db.fetch(
            conn,
            "SELECT host, protocol_version, ui_supported, "
            "count(*) AS conversations, max(created_at) AS last_seen "
            "FROM conversations GROUP BY 1, 2, 3 ORDER BY last_seen DESC",
        )
        tools = await db.fetch(
            conn,
            "SELECT tool, count(*) AS calls_24h FROM tool_calls "
            "WHERE created_at > now() - interval '24 hours' "
            "GROUP BY tool ORDER BY tool",
        )
        clusters = await db.fetch(
            conn,
            "SELECT id, medoid_topic, size FROM topic_clusters ORDER BY id",
        )
    return {
        "counts": counts,
        "hosts": hosts,
        "tool_calls": tools,
        "topic_clusters": clusters,
        "settings": {
            "policy": settings.policy,
            "control_unit": settings.control_unit,
            "arm_shares": settings.arm_shares,
            "q": settings.q,
            "cache_ttl_ms": settings.cache_ttl_ms,
            "shadow_policies": settings.shadow_policies,
        },
    }


@api.get("/turns")
async def turns(request: fastapi.Request, limit: int = 100) -> list[dict]:
    where, args = [], []
    for key, clause in FILTERS.items():
        if value := request.query_params.get(key):
            where.append(clause)
            args.append(value)
    condition = f"WHERE {" AND ".join(where)}" if where else ""
    sql = f"SELECT {TURN_COLUMNS} {condition} ORDER BY t.id DESC LIMIT %s"
    async with db.pool.connection() as conn:
        rows = await db.fetch(
            conn, sql, vocab.Source.ASSISTANT, *args, min(limit, MAX_TURNS)
        )
    return [_inferred(r) for r in rows]


@api.get("/turns/{turn_id}")
async def turn_detail(turn_id: int) -> dict:
    async with db.pool.connection() as conn:
        turn = await db.fetchrow(
            conn, "SELECT * FROM turns WHERE id = %s", turn_id
        )
        if not turn:
            raise fastapi.HTTPException(status_code=404)

        async def events(table: str, order: str = "created_at") -> list[dict]:
            return await db.fetch(
                conn,
                f"SELECT * FROM {table} WHERE turn_id = %s ORDER BY {order}",
                turn_id,
            )

        conversation_id = turn["conversation_id"]
        return {
            "turn": turn,
            "conversation": await db.fetchrow(
                conn,
                "SELECT * FROM conversations WHERE id = %s",
                conversation_id,
            ),
            "conversation_state": await db.fetchrow(
                conn,
                "SELECT * FROM conversation_state WHERE conversation_id = %s",
                conversation_id,
            ),
            "next_turn": await db.fetchrow(
                conn,
                "SELECT id, created_at FROM turns "
                "WHERE conversation_id = %s AND turn_index = %s",
                conversation_id,
                turn["turn_index"] + 1,
            ),
            "replies": await events("replies"),
            "outcomes": await events("outcomes"),
            "buttons_shown": await events("buttons_shown", "position"),
            "clicks": await events("clicks"),
            "context": await events("turn_context"),
            "gold_labels": await events("gold_labels"),
            "shadow_decisions": await events(
                "shadow_decisions", "model_version"
            ),
            "inference": await db.fetchrow(
                conn, "SELECT * FROM turn_inference WHERE turn_id = %s", turn_id
            ),
        }


@api.get("/users")
async def users() -> list[dict]:
    async with db.pool.connection() as conn:
        rows = await db.fetch(
            conn,
            """SELECT u.id, u.created_at, count(DISTINCT c.id) AS conversations,
               count(t.id) AS turns, max(t.created_at) AS last_seen,
               a.grid AS alpha, ur.rho_mean
               FROM users u LEFT JOIN conversations c ON c.user_id = u.id
               LEFT JOIN turns t ON t.conversation_id = c.id
               LEFT JOIN alpha_posteriors a ON a.user_id = u.id
               LEFT JOIN user_reading ur ON ur.user_id = u.id
               GROUP BY u.id, a.grid, ur.rho_mean
               ORDER BY last_seen DESC NULLS LAST""",
        )
    for r in rows:
        grid, rho = r.pop("alpha"), r.pop("rho_mean")
        r["alpha_ratio"] = (
            math.exp(stats.grid_mean_var(grid)[0]) if grid else None
        )
        r["reading_wps"] = math.exp(rho) if rho is not None else None
    return rows


@api.get("/users/{user_id}")
async def user_detail(user_id: int) -> dict:
    async with db.pool.connection() as conn:
        thetas = await db.fetch(
            conn,
            "SELECT request_type AS cell, grid FROM theta_posteriors "
            "WHERE user_id = %s ORDER BY request_type",
            user_id,
        )
        scopes = [
            r["scope"]
            for r in await db.fetch(
                conn,
                "SELECT DISTINCT scope FROM pref_counts WHERE user_id = %s "
                "ORDER BY scope",
                user_id,
            )
        ]
        dimensions = await db.fetch(
            conn,
            "SELECT DISTINCT scope, dimension FROM pref_counts "
            "WHERE user_id = %s ORDER BY scope, dimension",
            user_id,
        )
        counts = await engine.pref_counts(conn, user_id, scopes)
        reading = await engine.user_reading(conn, user_id)
    preferences = []
    for row in dimensions:
        belief = prefs.posterior(counts[row["scope"]], row["dimension"])
        total = sum(belief.values())
        preferences.append(
            {
                **row,
                "means": {v: a / total for v, a in belief.items()},
                "evidence": total - prefs.STRENGTH,
            }
        )
    lengths = []
    for row in thetas:
        mean, var = stats.grid_mean_var(row["grid"])
        lengths.append(
            {"cell": row["cell"], "ratio": math.exp(mean), "sd": math.sqrt(var)}
        )
    return {
        "lengths": lengths,
        "preferences": preferences,
        "reading": {
            "wps": math.exp(reading["rho_mean"]),
            "rho_sd": math.sqrt(reading["rho_var"]),
            "mixture": dict(
                zip(reading_lib.COMPONENTS, stats.normalize(reading["mixture"]))
            ),
            "reaction_prior": dict(
                zip(
                    reaction.OUTCOMES,
                    stats.normalize(reading["reaction_prior"]),
                )
            ),
        },
    }


@api.get("/model")
async def model() -> dict:
    async with db.pool.connection() as conn:
        p = await params_lib.load_params(conn)
        learned = {
            t: bool(await db.fetchrow(conn, f"SELECT 1 FROM {t} LIMIT 1"))
            for t in LEARNED_TABLES
        }
    h = p.hyper
    return {
        "learned": learned,
        "hyper": {
            "mu0_ratio": math.exp(h.mu0),
            "sigma_alpha": h.sigma_alpha,
            "sigma_gamma": h.sigma_gamma,
            "sigma_eta": h.sigma_eta,
            "tau": h.tau,
            "overhead_s": p.overhead,
            "gamma": {t: math.exp(g) for t, g in h.gamma.items()},
        },
        "reading": vars(p.reading),
        "family": {
            f: {**v, "speed_wps": math.exp(v["speed_mu"])}
            for f, v in p.family.items()
        },
        "noise": p.noise,
        "confusions": p.confusions,
        "reliability": p.reliability,
        "host_prior": p.host_prior,
    }


@api.post("/nightly")
async def run_nightly() -> dict:
    async with db.pool.connection() as conn, conn.transaction():
        await db.execute(
            conn, f"LOCK TABLE {", ".join(nightly.DERIVED)} IN EXCLUSIVE MODE"
        )
        return {"replayed_turns": await nightly.run(conn)}
