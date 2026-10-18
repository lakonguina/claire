"""The profile page: "What I learned about you" (SPEC 4.6)."""

import math
import urllib.parse

import fastapi
from fastapi import responses
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import db
from app import engine
from app import i18n
from app import models
from app import params as params_lib
from app import topics
from app import vocab
from app.model import prefs
from app.model import theta

router = fastapi.APIRouter()


Session = sa_asyncio.AsyncSession


async def _user(session: Session, key: str) -> tuple[models.User, str | None]:
    """Returns the user behind a profile key and their latest language."""
    user = await session.scalar(
        sa.select(models.User).where(models.User.profile_key == key)
    )
    if not user:
        raise fastapi.HTTPException(status_code=404)
    language = await session.scalar(
        sa.select(models.Turn.language)
        .where(models.Turn.user_id == user.id)
        .order_by(models.Turn.id.desc())
        .limit(1)
    )
    return user, language


async def _scopes(
    session: Session, user_id: int, belief: theta.Belief, prior: theta.Prior
) -> dict[str, list[str]]:
    """Returns the scopes shown per dimension: in general, then domains.

    A domain is shown for a dimension once something was learned in it.
    """
    pc = models.PrefCount
    learned = (
        await session.execute(
            sa.select(pc.dimension, pc.scope)
            .distinct()
            .where(
                pc.user_id == user_id,
                pc.scope.startswith(topics.DOMAIN_PREFIX),
                pc.scope != topics.ANY_DOMAIN,
            )
            .order_by(pc.scope)
        )
    ).all()
    out = {d: [prefs.USER_SCOPE] for d in vocab.Dimension}
    for dimension, scope in learned:
        if dimension in out:
            out[dimension].append(scope)
    out[vocab.LENGTH] = [prefs.USER_SCOPE, *theta.informed(belief, prior)]
    return out


async def _scope_name(session: Session, scope: str, texts: dict) -> str:
    cluster = None
    if scope.startswith(topics.DOMAIN_PREFIX) and scope != topics.ANY_DOMAIN:
        cluster = await session.get(
            models.TopicCluster, int(scope.removeprefix(topics.DOMAIN_PREFIX))
        )
    if cluster:
        return texts["profile"]["topics_like"].format(
            topic=cluster.medoid_topic
        )
    general = texts["profile"]["in_general"]
    if scope == theta.MOBILE:
        return f"{general} · {texts["profile"]["on_mobile"]}"
    return general


def _length_choice(belief: theta.Belief, scope: str, p) -> str:
    x = theta.edit_features(scope, topics.DOMAIN_PREFIX)
    mean, _ = belief.predict(x, p.prior)
    return min(
        theta.EDIT_RATIOS,
        key=lambda c: abs(math.log(theta.EDIT_RATIOS[c]) - mean),
    )


@router.get("/me/{key}", response_class=responses.HTMLResponse)
async def page(key: str) -> str:
    """Renders what the server believes about the user, line by line."""
    async with db.session() as session:
        user, language = await _user(session, key)
        texts = i18n.catalog(language)
        p = await params_lib.cache.get(session)
        belief = await engine.length_belief(session, user.id)
        rows = []
        for dimension, scopes in (
            await _scopes(session, user.id, belief, p.prior)
        ).items():
            counts = await engine.pref_counts(session, user.id, scopes)
            for scope in scopes:
                if dimension == vocab.LENGTH:
                    name, names = texts["profile"]["length"], texts["lengths"]
                    current = _length_choice(belief, scope, p)
                else:
                    pseudo_counts = prefs.posterior(counts[scope], dimension)
                    entry = texts["dimensions"][dimension]
                    name, names = entry["name"], entry["values"]
                    current = max(pseudo_counts, key=pseudo_counts.get)
                rows.append(
                    {
                        "name": name,
                        "scope": await _scope_name(session, scope, texts),
                        "scope_id": scope,
                        "dimension": dimension,
                        "options": names,
                        "current": current,
                    }
                )
    return i18n.TEMPLATES.get_template("profile.html").render(
        language=language or i18n.FALLBACK, texts=texts["profile"], rows=rows
    )


@router.post("/me/{key}")
async def correct(key: str, request: fastapi.Request) -> responses.Response:
    """Stores a correction and applies it at once (SPEC 4.6)."""
    form = dict(urllib.parse.parse_qsl((await request.body()).decode()))
    dimension, value, scope = (
        form.get("dimension"),
        form.get("value"),
        form.get("scope"),
    )
    async with db.session() as session, session.begin():
        user, _ = await _user(session, key)
        valid = (
            value in vocab.LengthChoice
            if dimension == vocab.LENGTH
            else value in vocab.DIMENSION_VALUES.get(dimension, ())
        )
        belief = await engine.length_belief(session, user.id)
        p = await params_lib.cache.get(session)
        scopes = (await _scopes(session, user.id, belief, p.prior)).get(
            dimension, []
        )
        if not valid or scope not in scopes:
            raise fastapi.HTTPException(status_code=422)
        edit = models.ProfileEdit(
            user_id=user.id, scope=scope, dimension=dimension, value=value
        )
        session.add(edit)
        await session.flush()
        await engine.apply_edit(session, edit, p)
    return responses.RedirectResponse(f"/me/{key}", status_code=303)
