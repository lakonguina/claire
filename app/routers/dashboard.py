"""Admin dashboard: the page at /admin and its JSON API (SPEC 11)."""

import datetime
import math

import fastapi
from fastapi import responses
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import config
from app.core import db
from app.core import i18n
from app.core import vocab
from app.jobs import nightly
from app.model import prefs
from app.model import reaction
from app.model import reading as reading_lib
from app.model import stats
from app.model import theta
from app import models
from app.services import engine
from app.services import params as params_lib
from app.services import questions
from app.services import topics

PAGE = i18n.TEMPLATES.get_template("dashboard.html")
# The catalog sections the page reads.
PAGE_TEXTS = ("dashboard", "buttons", "dimensions", "lengths")
MAX_TURNS = 500
GENERAL = theta.features({}, mobile=False)
MAX_QUESTIONS = 10
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
async def dashboard_page(request: fastapi.Request) -> str:
    """Returns the page in the browser's language; it holds no data."""
    language = i18n.preferred(request.headers.get("accept-language"))
    texts = i18n.catalog(language)
    return PAGE.render(
        language=language, texts={k: texts[k] for k in PAGE_TEXTS}
    )


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
        distinct_turns = {
            name: sa.select(model.turn_id).distinct()
            for name, model in (
                ("turns_with_buttons", models.ButtonShown),
                ("turns_with_click", models.Click),
                ("finalized_turns", models.TurnInference),
            )
        }
        statements = {
            "users": sa.select(models.User.id),
            "conversations": sa.select(c.id),
            "held_back_conversations": sa.select(c.id).where(
                c.grp != vocab.Arm.TREATMENT
            ),
            "turns": sa.select(t.id),
            "turns_24h": sa.select(t.id).where(t.created_at > now - DAY),
            "widget_turns": widget,
            "widget_turns_with_reply": sa.select(models.Reply.turn_id)
            .distinct()
            .where(models.Reply.turn_id.in_(widget)),
            **distinct_turns,
            "gold_labels": sa.select(models.GoldLabel.id),
            "topic_clusters": sa.select(models.TopicCluster.id),
        }
        counts = {
            name: await _count(session, statement)
            for name, statement in statements.items()
        }
        counts["last_nightly"] = await session.scalar(
            sa.select(sa.func.max(models.Hyperparam.updated_at))
        )
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
        q, f = models.Question, models.QuestionFollow
        frequent = (
            await session.execute(
                sa.select(
                    q.text,
                    q.language,
                    q.users,
                    sa.func.coalesce(sa.func.sum(f.users), 0).label("follows"),
                )
                .outerjoin(f, f.question_id == q.id)
                .group_by(q.id)
                .order_by(q.users.desc(), q.id)
                .limit(MAX_QUESTIONS)
            )
        ).mappings()
        return {
            "counts": counts,
            "hosts": [dict(r) for r in hosts],
            "tool_calls": [dict(r) for r in tools],
            "topic_clusters": [
                {"id": k.id, "medoid_topic": k.medoid_topic, "size": k.size}
                for k in clusters
            ],
            "questions": [dict(r) for r in frequent],
            "min_users": questions.MIN_USERS,
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
    where = [
        build(value)
        for key, build in FILTERS.items()
        if (value := request.query_params.get(key))
    ]
    async with db.session() as session:
        rows = (
            await session.execute(
                engine.with_events(
                    sa.select(
                        models.Turn, models.Conversation, models.TurnInference
                    )
                    .join(models.Conversation)
                    .outerjoin(
                        models.TurnInference,
                        models.TurnInference.turn_id == models.Turn.id,
                    )
                    .where(*where)
                    .order_by(models.Turn.id.desc())
                    .limit(min(limit, MAX_TURNS))
                )
            )
        ).all()
    out = []
    for turn, conv, inference in rows:
        reply = engine.first(turn.replies)
        verdict = engine.first(turn.outcomes, source=vocab.Source.ASSISTANT)
        click = engine.last(turn.clicks)
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
                "clicked": click.position if click else None,
                "buttons": [
                    {"id": b.button, "label": b.label, "position": b.position}
                    for b in turn.buttons_shown
                ],
                "inferred": max(probs, key=probs.get, default=None),
                "inferred_p": max(probs.values(), default=None),
            }
        )
    return out


@api.get("/turns/{turn_id}")
async def turn_detail(turn_id: int) -> dict:
    async with db.session() as session:
        turn = await session.scalar(
            engine.with_events(
                sa.select(models.Turn).where(models.Turn.id == turn_id)
            )
        )
        if not turn:
            raise fastapi.HTTPException(status_code=404)
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
        conversation = await session.get(
            models.Conversation, turn.conversation_id
        )
    events = {
        r.key: [_row(e) for e in getattr(turn, r.key)]
        for r in engine.TURN_EVENTS
        if r is not models.Turn.embeddings  # 384 floats, not for reading.
    }
    return {
        "turn": _row(turn),
        "conversation": _row(conversation),
        "conversation_state": _row(state) if state else None,
        "next_turn": (
            {"id": next_turn.id, "created_at": next_turn.created_at}
            if next_turn
            else None
        ),
        **events,
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
        beliefs = {
            b.user_id: engine.belief_of(b)
            for b in await session.scalars(sa.select(models.LengthPosterior))
        }
        p = await params_lib.cache.get(session)
        speeds = {
            r.user_id: r.rho_mean
            for r in await session.scalars(sa.select(models.UserReading))
        }
    return [
        {
            **r,
            "length_ratio": (
                math.exp(beliefs[r["id"]].predict(GENERAL, p.prior)[0])
                if r["id"] in beliefs
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
        belief = await engine.length_belief(session, user_id)
        p = await params_lib.cache.get(session)
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
        pseudo_counts = prefs.posterior(counts[scope], dimension)
        total = sum(pseudo_counts.values())
        preferences.append(
            {
                "scope": scope,
                "dimension": dimension,
                "means": {v: a / total for v, a in pseudo_counts.items()},
                "evidence": total - prefs.STRENGTH,
            }
        )
    lengths = []
    for scope in [prefs.USER_SCOPE, *theta.informed(belief, p.prior)]:
        x = theta.edit_features(scope, topics.DOMAIN_PREFIX)
        mean, var = belief.predict(x, p.prior)
        lengths.append(
            {"scope": scope, "ratio": math.exp(mean), "sd": math.sqrt(var)}
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
        clusters = list(await session.scalars(sa.select(models.TopicCluster)))
    prior = p.prior
    return {
        "learned": learned,
        "hyper": {
            "tau": prior.tau,
            "overhead_s": p.overhead,
            "default_sd": math.sqrt(prior.default_var),
            "prior": [
                {
                    "feature": f,
                    "ratio": math.exp(prior.mean.get(f, 0.0)),
                    "sd": math.sqrt(prior.var.get(f, prior.default_var)),
                }
                for f in dict.fromkeys([*prior.mean, *prior.var])
            ],
        },
        "domains": {
            topics.domain_scope(k.id): k.medoid_topic for k in clusters
        },
        "reading": vars(p.reading),
        "family": {
            f: {**v, "speed_wps": math.exp(v["speed_mu"])}
            for f, v in p.families.params.items()
        },
        "noise": p.noise,
        "confusions": p.confusions,
        "host_prior": p.families.host_prior,
    }


@api.post("/nightly")
async def run_nightly() -> dict:
    async with db.session() as session, session.begin():
        await nightly.lock_derived(session)
        return {"replayed_turns": await nightly.run(session)}
