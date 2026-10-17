"""Admin dashboard: the page at /admin and its JSON API (SPEC 11)."""

import datetime
import math
import pathlib

import fastapi
from fastapi import responses
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import db
from app import engine
from app import models
from app import nightly
from app import params as params_lib
from app import vocab
from app.model import prefs
from app.model import reaction
from app.model import reading as reading_lib
from app.model import stats

PAGE = (pathlib.Path(__file__).resolve().parent / "dashboard.html").read_text()
MAX_TURNS = 500
DAY = datetime.timedelta(hours=24)
BEARER = "Bearer "
FILTERS = {
    "user": lambda v: models.Turn.user_id == int(v),
    "conversation": lambda v: models.Turn.conversation_id == int(v),
    "host": lambda v: models.Conversation.host == v,
    "mode": lambda v: models.Turn.reply_mode == v,
    "group": lambda v: models.Conversation.grp == v,
    "q": models.Turn.topic.icontains,
    "before": lambda v: models.Turn.id < int(v),
}
Session = sa_asyncio.AsyncSession


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


def _row(obj) -> dict:
    """Returns a model instance as a dict of its columns."""
    return {c.key: getattr(obj, c.key) for c in sa.inspect(obj).mapper.columns}


async def _count(session: Session, statement) -> int:
    return await session.scalar(
        sa.select(sa.func.count()).select_from(statement.subquery())
    )


@api.get("/overview")
async def overview() -> dict:
    settings = config.settings
    t, c = models.Turn, models.Conversation
    async with db.session() as session:
        now = await session.scalar(sa.select(sa.func.now()))
        widget = sa.select(t.id).where(t.reply_mode == vocab.ReplyMode.WIDGET)
        counts = {
            "users": await _count(session, sa.select(models.User.id)),
            "conversations": await _count(session, sa.select(c.id)),
            "held_back_conversations": await _count(
                session, sa.select(c.id).where(c.grp != vocab.Arm.TREATMENT)
            ),
            "turns": await _count(session, sa.select(t.id)),
            "turns_24h": await _count(
                session, sa.select(t.id).where(t.created_at > now - DAY)
            ),
            "widget_turns": await _count(session, widget),
            "widget_turns_with_reply": await _count(
                session,
                sa.select(models.Reply.turn_id)
                .distinct()
                .where(models.Reply.turn_id.in_(widget)),
            ),
            "turns_with_buttons": await _count(
                session, sa.select(models.ButtonShown.turn_id).distinct()
            ),
            "turns_with_click": await _count(
                session, sa.select(models.Click.turn_id).distinct()
            ),
            "finalized_turns": await _count(
                session, sa.select(models.TurnInference.turn_id)
            ),
            "gold_labels": await _count(
                session, sa.select(models.GoldLabel.id)
            ),
            "topic_clusters": await _count(
                session, sa.select(models.TopicCluster.id)
            ),
            "last_nightly": await session.scalar(
                sa.select(sa.func.max(models.Hyperparam.updated_at))
            ),
        }
        hosts = (
            await session.execute(
                sa.select(
                    c.host,
                    c.protocol_version,
                    c.ui_supported,
                    sa.func.count().label("conversations"),
                    sa.func.max(c.created_at).label("last_seen"),
                )
                .group_by(c.host, c.protocol_version, c.ui_supported)
                .order_by(sa.desc("last_seen"))
            )
        ).mappings()
        tools = (
            await session.execute(
                sa.select(
                    models.ToolCall.tool, sa.func.count().label("calls_24h")
                )
                .where(models.ToolCall.created_at > now - DAY)
                .group_by(models.ToolCall.tool)
                .order_by(models.ToolCall.tool)
            )
        ).mappings()
        clusters = await session.scalars(
            sa.select(models.TopicCluster).order_by(models.TopicCluster.id)
        )
        return {
            "counts": counts,
            "hosts": [dict(r) for r in hosts],
            "tool_calls": [dict(r) for r in tools],
            "topic_clusters": [
                {"id": k.id, "medoid_topic": k.medoid_topic, "size": k.size}
                for k in clusters
            ],
            "settings": {
                "policy": settings.policy,
                "control_unit": settings.control_unit,
                "arm_shares": settings.arm_shares,
                "q": settings.q,
                "cache_ttl_ms": settings.cache_ttl_ms,
                "shadow_policies": settings.shadow_policies,
            },
        }


async def _first(session: Session, model, turn_ids: list[int], *where) -> dict:
    out = {}
    for event in await session.scalars(
        sa.select(model)
        .where(model.turn_id.in_(turn_ids), *where)
        .order_by(model.created_at)
    ):
        out.setdefault(event.turn_id, event)
    return out


@api.get("/turns")
async def turns(request: fastapi.Request, limit: int = 100) -> list[dict]:
    where = [
        build(value)
        for key, build in FILTERS.items()
        if (value := request.query_params.get(key))
    ]
    async with db.session() as session:
        rows = (
            await session.execute(
                sa.select(models.Turn, models.Conversation)
                .join(
                    models.Conversation,
                    models.Conversation.id == models.Turn.conversation_id,
                )
                .where(*where)
                .order_by(models.Turn.id.desc())
                .limit(min(limit, MAX_TURNS))
            )
        ).all()
        ids = [turn.id for turn, _ in rows]
        replies = await _first(session, models.Reply, ids)
        verdicts = await _first(
            session,
            models.Outcome,
            ids,
            models.Outcome.source == vocab.Source.ASSISTANT,
        )
        clicks = await _first(session, models.Click, ids)
        inferences = {
            i.turn_id: i
            for i in await session.scalars(
                sa.select(models.TurnInference).where(
                    models.TurnInference.turn_id.in_(ids)
                )
            )
        }
        buttons = {}
        for b in await session.scalars(
            sa.select(models.ButtonShown)
            .where(models.ButtonShown.turn_id.in_(ids))
            .order_by(models.ButtonShown.position)
        ):
            buttons.setdefault(b.turn_id, []).append(b.button)
    out = []
    for turn, conv in rows:
        reply, verdict = replies.get(turn.id), verdicts.get(turn.id)
        click, inference = clicks.get(turn.id), inferences.get(turn.id)
        probs = inference.reaction_probs if inference else {}
        out.append(
            {
                **_row(turn),
                "host": conv.host,
                "grp": conv.grp,
                "reply_words_exact": reply.reply_words_exact if reply else None,
                "writing_seconds": (
                    (reply.created_at - turn.created_at).total_seconds()
                    if reply
                    else None
                ),
                "verdict": verdict.observed if verdict else None,
                "user_quote": verdict.user_quote if verdict else None,
                "format_issue": verdict.format_issue if verdict else None,
                "clicked": click.button if click else None,
                "buttons": buttons.get(turn.id),
                "inferred": max(probs, key=probs.get, default=None),
                "inferred_p": max(probs.values(), default=None),
            }
        )
    return out


@api.get("/turns/{turn_id}")
async def turn_detail(turn_id: int) -> dict:
    async with db.session() as session:
        turn = await session.get(models.Turn, turn_id)
        if not turn:
            raise fastapi.HTTPException(status_code=404)

        async def events(model, order=None) -> list[dict]:
            rows = await session.scalars(
                sa.select(model)
                .where(model.turn_id == turn_id)
                .order_by(order if order is not None else model.id)
            )
            return [_row(r) for r in rows]

        state = await session.get(
            models.ConversationState, turn.conversation_id
        )
        inference = await session.get(models.TurnInference, turn_id)
        next_turn = await session.scalar(
            sa.select(models.Turn).where(
                models.Turn.conversation_id == turn.conversation_id,
                models.Turn.turn_index == turn.turn_index + 1,
            )
        )
        return {
            "turn": _row(turn),
            "conversation": _row(
                await session.get(models.Conversation, turn.conversation_id)
            ),
            "conversation_state": _row(state) if state else None,
            "next_turn": (
                {"id": next_turn.id, "created_at": next_turn.created_at}
                if next_turn
                else None
            ),
            "replies": await events(models.Reply),
            "outcomes": await events(models.Outcome),
            "buttons_shown": await events(
                models.ButtonShown, models.ButtonShown.position
            ),
            "clicks": await events(models.Click),
            "context": await events(models.TurnContext),
            "gold_labels": await events(models.GoldLabel),
            "shadow_decisions": await events(models.ShadowDecision),
            "inference": _row(inference) if inference else None,
        }


@api.get("/users")
async def users() -> list[dict]:
    u, c, t = models.User, models.Conversation, models.Turn
    async with db.session() as session:
        rows = (
            await session.execute(
                sa.select(
                    u.id,
                    u.created_at,
                    sa.func.count(sa.distinct(c.id)).label("conversations"),
                    sa.func.count(t.id).label("turns"),
                    sa.func.max(t.created_at).label("last_seen"),
                )
                .outerjoin(c, c.user_id == u.id)
                .outerjoin(t, t.conversation_id == c.id)
                .group_by(u.id)
                .order_by(sa.desc("last_seen").nulls_last())
            )
        ).mappings()
        alphas = {
            a.user_id: a.grid
            for a in await session.scalars(sa.select(models.AlphaPosterior))
        }
        speeds = {
            r.user_id: r.rho_mean
            for r in await session.scalars(sa.select(models.UserReading))
        }
    return [
        {
            **r,
            "alpha_ratio": (
                math.exp(stats.grid_mean_var(alphas[r["id"]])[0])
                if r["id"] in alphas
                else None
            ),
            "reading_wps": (
                math.exp(speeds[r["id"]]) if r["id"] in speeds else None
            ),
        }
        for r in rows
    ]


@api.get("/users/{user_id}")
async def user_detail(user_id: int) -> dict:
    pc = models.PrefCount
    async with db.session() as session:
        thetas = list(
            await session.scalars(
                sa.select(models.ThetaPosterior)
                .where(models.ThetaPosterior.user_id == user_id)
                .order_by(models.ThetaPosterior.request_type)
            )
        )
        dimensions = (
            await session.execute(
                sa.select(pc.scope, pc.dimension)
                .distinct()
                .where(pc.user_id == user_id)
                .order_by(pc.scope, pc.dimension)
            )
        ).all()
        counts = await engine.pref_counts(
            session, user_id, {scope for scope, _ in dimensions}
        )
        reading = await engine.user_reading(session, user_id)
    preferences = []
    for scope, dimension in dimensions:
        belief = prefs.posterior(counts[scope], dimension)
        total = sum(belief.values())
        preferences.append(
            {
                "scope": scope,
                "dimension": dimension,
                "means": {v: a / total for v, a in belief.items()},
                "evidence": total - prefs.STRENGTH,
            }
        )
    lengths = []
    for cell in thetas:
        mean, var = stats.grid_mean_var(cell.grid)
        lengths.append(
            {
                "cell": cell.request_type,
                "ratio": math.exp(mean),
                "sd": math.sqrt(var),
            }
        )
    return {
        "lengths": lengths,
        "preferences": preferences,
        "reading": {
            "wps": math.exp(reading.rho_mean),
            "rho_sd": math.sqrt(reading.rho_var),
            "mixture": dict(
                zip(reading_lib.COMPONENTS, stats.normalize(reading.mixture))
            ),
            "reaction_prior": dict(
                zip(reaction.OUTCOMES, stats.normalize(reading.reaction_prior))
            ),
        },
    }


@api.get("/model")
async def learned_model() -> dict:
    async with db.session() as session:
        p = await params_lib.load_params(session)
        learned = {
            m.__tablename__: await _count(session, sa.select(m)) > 0
            for m in (*models.LEARNED, models.TopicCluster)
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
    async with db.session() as session, session.begin():
        await nightly.lock_derived(session)
        return {"replayed_turns": await nightly.run(session)}
