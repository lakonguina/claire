"""Admin dashboard: the page at /admin and its read-only JSON API (plus a manual nightly run)."""

import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from app import nightly
from app.config import settings
from app.db import execute, fetch, fetchrow, pool
from app.engine import user_reading
from app.model.reaction import OUTCOMES
from app.model.stats import grid_mean_var, normalize
from app.params import load_params

PAGE = (Path(__file__).resolve().parent / "dashboard.html").read_text()
TURN_COLUMNS = """
    t.id, t.created_at, t.user_id, t.conversation_id, t.turn_index, c.host, c.grp, t.reply_mode, t.request_type,
    t.topic, t.language, t.needed_words, t.target_words, t.requested_words, t.model_version, t.transport_session,
    r.reply_words_exact, extract(epoch FROM r.created_at - t.created_at)::float8 AS writing_seconds,
    o.observed AS verdict, o.user_quote, k.button AS clicked, i.reaction_probs,
    (SELECT array_agg(b.button ORDER BY b.position) FROM buttons_shown b WHERE b.turn_id = t.id) AS buttons
    FROM turns t JOIN conversations c ON c.id = t.conversation_id
    LEFT JOIN LATERAL (SELECT * FROM replies WHERE turn_id = t.id ORDER BY created_at LIMIT 1) r ON true
    LEFT JOIN LATERAL (SELECT * FROM outcomes WHERE turn_id = t.id AND source = 'assistant'
                       ORDER BY created_at LIMIT 1) o ON true
    LEFT JOIN LATERAL (SELECT * FROM clicks WHERE turn_id = t.id ORDER BY created_at LIMIT 1) k ON true
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


def admin(request: Request) -> None:
    supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
    if not settings.admin_token or supplied != settings.admin_token:
        raise HTTPException(status_code=401)


page = APIRouter()
api = APIRouter(prefix="/admin/api", dependencies=[Depends(admin)])


@page.get("/admin", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page():
    return PAGE  # holds no data: the API calls below need the admin token


def _inferred(row: dict) -> dict:
    probs = row.pop("reaction_probs", None)
    row["inferred"] = max(probs, key=probs.get) if probs else None
    row["inferred_p"] = max(probs.values()) if probs else None
    return row


@api.get("/overview")
async def overview():
    async with pool.connection() as conn:
        counts = await fetchrow(conn, """
            SELECT (SELECT count(*) FROM users) AS users,
                   (SELECT count(*) FROM conversations) AS conversations,
                   (SELECT count(*) FROM conversations WHERE grp = 'control') AS control_conversations,
                   (SELECT count(*) FROM turns) AS turns,
                   (SELECT count(*) FROM turns WHERE created_at > now() - interval '24 hours') AS turns_24h,
                   (SELECT count(*) FROM turns WHERE reply_mode = 'widget') AS widget_turns,
                   (SELECT count(DISTINCT r.turn_id) FROM replies r JOIN turns t ON t.id = r.turn_id
                    WHERE t.reply_mode = 'widget') AS widget_turns_with_reply,
                   (SELECT count(DISTINCT turn_id) FROM buttons_shown) AS turns_with_buttons,
                   (SELECT count(DISTINCT turn_id) FROM clicks) AS turns_with_click,
                   (SELECT count(*) FROM turn_inference) AS finalized_turns,
                   (SELECT count(*) FROM gold_labels) AS gold_labels,
                   (SELECT max(updated_at) FROM hyperparams) AS last_nightly""")
        hosts = await fetch(conn, """
            SELECT host, protocol_version, ui_supported, count(*) AS conversations, max(created_at) AS last_seen
            FROM conversations GROUP BY 1, 2, 3 ORDER BY last_seen DESC""")
        tools = await fetch(conn, """
            SELECT tool, count(*) AS calls_24h FROM tool_calls WHERE created_at > now() - interval '24 hours'
            GROUP BY tool ORDER BY tool""")
        sessions = await fetchrow(conn, """
            SELECT count(DISTINCT transport_session) AS sessions,
                   (SELECT max(n) FROM (SELECT count(DISTINCT conversation_id) AS n FROM turns
                    WHERE transport_session IS NOT NULL GROUP BY transport_session) s) AS max_conversations_per_session,
                   (SELECT max(n) FROM (SELECT count(DISTINCT transport_session) AS n FROM turns
                    WHERE transport_session IS NOT NULL GROUP BY conversation_id) c) AS max_sessions_per_conversation
            FROM turns WHERE transport_session IS NOT NULL""")
    return {"counts": counts, "hosts": hosts, "tool_calls": tools, "sessions": sessions,
            "settings": {"policy": settings.policy, "control_unit": settings.control_unit,
                         "control_share": settings.control_share, "q": settings.q,
                         "cache_ttl_ms": settings.cache_ttl_ms, "shadow_policies": settings.shadow_policies}}


@api.get("/turns")
async def turns(request: Request, limit: int = 100):
    where, args = [], []
    for key, clause in FILTERS.items():
        if value := request.query_params.get(key):
            where.append(clause)
            args.append(value)
    sql = f"SELECT {TURN_COLUMNS} {'WHERE ' + ' AND '.join(where) if where else ''} ORDER BY t.id DESC LIMIT %s"
    async with pool.connection() as conn:
        return [_inferred(r) for r in await fetch(conn, sql, *args, min(limit, 500))]


@api.get("/turns/{turn_id}")
async def turn_detail(turn_id: int):
    async with pool.connection() as conn:
        turn = await fetchrow(conn, "SELECT * FROM turns WHERE id = %s", turn_id)
        if not turn:
            raise HTTPException(status_code=404)
        one = lambda table, order="created_at": fetch(conn, f"SELECT * FROM {table} WHERE turn_id = %s ORDER BY {order}", turn_id)
        return {
            "turn": turn,
            "conversation": await fetchrow(conn, "SELECT * FROM conversations WHERE id = %s", turn["conversation_id"]),
            "conversation_state": await fetchrow(conn, "SELECT * FROM conversation_state WHERE conversation_id = %s",
                                                 turn["conversation_id"]),
            "next_turn": await fetchrow(conn, """SELECT id, created_at FROM turns WHERE conversation_id = %s
                                                 AND turn_index = %s""", turn["conversation_id"], turn["turn_index"] + 1),
            "replies": await one("replies"),
            "outcomes": await one("outcomes"),
            "buttons_shown": await one("buttons_shown", "position"),
            "clicks": await one("clicks"),
            "gold_labels": await one("gold_labels"),
            "shadow_decisions": await one("shadow_decisions", "model_version"),
            "inference": await fetchrow(conn, "SELECT * FROM turn_inference WHERE turn_id = %s", turn_id),
        }


@api.get("/users")
async def users():
    async with pool.connection() as conn:
        rows = await fetch(conn, """
            SELECT u.id, u.created_at, count(DISTINCT c.id) AS conversations, count(t.id) AS turns,
                   max(t.created_at) AS last_seen, a.grid AS alpha, ur.rho_mean
            FROM users u LEFT JOIN conversations c ON c.user_id = u.id LEFT JOIN turns t ON t.conversation_id = c.id
            LEFT JOIN alpha_posteriors a ON a.user_id = u.id LEFT JOIN user_reading ur ON ur.user_id = u.id
            GROUP BY u.id, a.grid, ur.rho_mean ORDER BY last_seen DESC NULLS LAST""")
    for r in rows:
        grid = r.pop("alpha")
        r["alpha_ratio"] = math.exp(grid_mean_var(grid)[0]) if grid else None
        rho = r.pop("rho_mean")
        r["reading_wps"] = math.exp(rho) if rho is not None else None
    return rows


@api.get("/users/{user_id}")
async def user_detail(user_id: int):
    async with pool.connection() as conn:
        thetas = await fetch(conn, "SELECT request_type, grid FROM theta_posteriors WHERE user_id = %s", user_id)
        counts = await fetch(conn, """SELECT request_type, key, alpha, beta FROM beta_counts WHERE user_id = %s
                                      ORDER BY request_type, key""", user_id)
        reading = await user_reading(conn, user_id)
    by_type = []
    for row in thetas:
        mean, var = grid_mean_var(row["grid"])
        by_type.append({"request_type": row["request_type"], "ratio": math.exp(mean), "sd": math.sqrt(var),
                        "grid": row["grid"]})
    return {
        "theta": sorted(by_type, key=lambda r: r["request_type"]),
        "beta_counts": [{**c, "mean": c["alpha"] / (c["alpha"] + c["beta"]), "signals": c["alpha"] + c["beta"] - 2}
                        for c in counts],
        "reading": {"wps": math.exp(reading["rho_mean"]), "rho_sd": math.sqrt(reading["rho_var"]),
                    "mixture": dict(zip(["read", "skim", "away"], normalize(reading["mixture"]))),
                    "reaction_prior": dict(zip(OUTCOMES, normalize(reading["reaction_prior"])))},
    }


@api.get("/model")
async def model():
    async with pool.connection() as conn:
        p = await load_params(conn)
        learned = {t: bool(await fetchrow(conn, f"SELECT 1 FROM {t} LIMIT 1")) for t in (
            "hyperparams", "family_params", "sensor_noise", "annotator_confusion", "self_report_reliability",
            "host_family_prior")}
    h = p.hyper
    return {
        "learned": learned,
        "hyper": {"mu0_ratio": math.exp(h.mu0), "sigma_alpha": h.sigma_alpha, "sigma_gamma": h.sigma_gamma,
                  "sigma_eta": h.sigma_eta, "tau": h.tau, "overhead_s": p.overhead,
                  "gamma": {t: math.exp(g) for t, g in h.gamma.items()}},
        "reading": vars(p.reading),
        "family": {f: {**v, "speed_wps": math.exp(v["speed_mu"])} for f, v in p.family.items()},
        "noise": p.noise,
        "confusions": p.confusions,
        "reliability": p.reliability,
        "host_prior": p.host_prior,
    }


@api.post("/nightly")
async def run_nightly():
    async with pool.connection() as conn, conn.transaction():
        await execute(conn, f"LOCK TABLE {', '.join(nightly.DERIVED)} IN EXCLUSIVE MODE")
        return {"replayed_turns": await nightly.run(conn)}
