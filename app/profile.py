"""The profile page: "What I learned about you" (SPEC 4.6)."""

import html
import math
import pathlib
import string
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
from app.model import stats
from app.model import theta

HERE = pathlib.Path(__file__).resolve().parent
TEMPLATE = string.Template((HERE / "profile.html").read_text())
ROW = string.Template((HERE / "profile_row.html").read_text())

router = fastapi.APIRouter()


Session = sa_asyncio.AsyncSession


async def _user(session: Session, key: str) -> tuple[models.User, str | None]:
    """Returns the user behind a profile key and their latest language."""
    user = await session.scalar(
        sa.select(models.User).where(models.User.token_hash == key)
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


async def _scopes(session: Session, user_id: int) -> dict[str, list[str]]:
    """Returns the scopes shown per dimension: general, domains, cells."""
    pc, tp = models.PrefCount, models.ThetaPosterior
    domains = await session.scalars(
        sa.select(pc.scope)
        .distinct()
        .where(
            pc.user_id == user_id,
            pc.dimension == vocab.Dimension.LEVEL,
            pc.scope.startswith(topics.DOMAIN_PREFIX),
        )
        .order_by(pc.scope)
    )
    cells = await session.scalars(
        sa.select(tp.request_type)
        .where(tp.user_id == user_id)
        .order_by(tp.request_type)
    )
    out = {d: [prefs.USER_SCOPE] for d in vocab.Dimension}
    out[vocab.Dimension.LEVEL] += list(domains)
    out[vocab.LENGTH] = list(cells)
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
    if scope == prefs.USER_SCOPE or scope.startswith(topics.DOMAIN_PREFIX):
        return texts["profile"]["in_general"]
    name = texts["request_types"][theta.request_type_of(scope)]
    mobile = scope != theta.request_type_of(scope)
    return f"{name} · {texts["profile"]["on_mobile"]}" if mobile else name


def _length_choice(grid: list[float]) -> str:
    mean, _ = stats.grid_mean_var(grid)
    return min(
        theta.EDIT_RATIOS,
        key=lambda c: abs(math.log(theta.EDIT_RATIOS[c]) - mean),
    )


def _options(names: dict[str, str], current: str) -> str:
    return "".join(
        f'<option value="{v}"{" selected" if v == current else ""}>'
        f"{html.escape(label)}</option>"
        for v, label in names.items()
    )


@router.get("/me/{key}", response_class=responses.HTMLResponse)
async def page(key: str) -> str:
    """Renders what the server believes about the user, line by line."""
    async with db.session() as session:
        user, language = await _user(session, key)
        texts = i18n.catalog(language)
        p = await params_lib.cache.get(session)
        rows = []
        for dimension, scopes in (await _scopes(session, user.id)).items():
            counts = await engine.pref_counts(session, user.id, scopes)
            for scope in scopes:
                if dimension == vocab.LENGTH:
                    cell, _ = await engine.grids(session, user.id, scope, p)
                    name, names = texts["profile"]["length"], texts["lengths"]
                    current = _length_choice(cell.grid)
                else:
                    belief = prefs.posterior(counts[scope], dimension)
                    entry = texts["dimensions"][dimension]
                    name, names = entry["name"], entry["values"]
                    current = max(belief, key=belief.get)
                rows.append(
                    ROW.substitute(
                        name=html.escape(name),
                        scope=html.escape(
                            await _scope_name(session, scope, texts)
                        ),
                        scope_id=html.escape(scope),
                        dimension=dimension,
                        options=_options(names, current),
                        save=html.escape(texts["profile"]["save"]),
                    )
                )
    return TEMPLATE.substitute(
        language=html.escape(language or i18n.FALLBACK),
        title=html.escape(texts["profile"]["title"]),
        intro=html.escape(texts["profile"]["intro"]),
        rows="".join(rows),
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
        scopes = (await _scopes(session, user.id)).get(dimension, [])
        if not valid or scope not in scopes:
            raise fastapi.HTTPException(status_code=422)
        edit = models.ProfileEdit(
            user_id=user.id, scope=scope, dimension=dimension, value=value
        )
        session.add(edit)
        await session.flush()
        await engine.apply_edit(
            session, edit, await params_lib.cache.get(session)
        )
    return responses.RedirectResponse(f"/me/{key}", status_code=303)
