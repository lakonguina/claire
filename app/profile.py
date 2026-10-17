"""The profile page: "What I learned about you" (SPEC 4.6)."""

import html
import math
import pathlib
import string
import urllib.parse

import fastapi
from fastapi import responses

from app import db
from app import engine
from app import i18n
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


async def _user(conn: db.Connection, key: str) -> dict:
    user = await db.fetchrow(
        conn,
        "SELECT u.id, (SELECT language FROM turns t WHERE t.user_id = u.id "
        "ORDER BY t.id DESC LIMIT 1) AS language "
        "FROM users u WHERE u.token_hash = %s",
        key,
    )
    if not user:
        raise fastapi.HTTPException(status_code=404)
    return user


async def _scopes(conn: db.Connection, user_id: int) -> dict[str, list[str]]:
    """Returns the scopes shown per dimension: general, domains, cells."""
    domains = await db.fetch(
        conn,
        "SELECT DISTINCT scope FROM pref_counts "
        "WHERE user_id = %s AND dimension = %s AND scope LIKE %s "
        "ORDER BY scope",
        user_id,
        vocab.Dimension.LEVEL,
        topics.DOMAIN_PREFIX + "%",
    )
    cells = await db.fetch(
        conn,
        "SELECT request_type FROM theta_posteriors WHERE user_id = %s "
        "ORDER BY request_type",
        user_id,
    )
    out = {d: [prefs.USER_SCOPE] for d in vocab.Dimension}
    out[vocab.Dimension.LEVEL] += [r["scope"] for r in domains]
    out[vocab.LENGTH] = [r["request_type"] for r in cells]
    return out


async def _scope_name(conn: db.Connection, scope: str, texts: dict) -> str:
    if scope in (prefs.USER_SCOPE, topics.ANY_DOMAIN):
        return texts["profile"]["in_general"]
    if scope.startswith(topics.DOMAIN_PREFIX):
        row = await db.fetchrow(
            conn,
            "SELECT medoid_topic FROM topic_clusters WHERE id = %s",
            int(scope.removeprefix(topics.DOMAIN_PREFIX)),
        )
        topic = row["medoid_topic"] if row else "…"
        return texts["profile"]["topics_like"].format(topic=topic)
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
    async with db.pool.connection() as conn:
        user = await _user(conn, key)
        texts = i18n.catalog(user["language"])
        p = await params_lib.cache.get(conn)
        rows = []
        for dimension, scopes in (await _scopes(conn, user["id"])).items():
            counts = await engine.pref_counts(conn, user["id"], scopes)
            for scope in scopes:
                if dimension == vocab.LENGTH:
                    grid, _ = await engine.grids(conn, user["id"], scope, p)
                    name, names = texts["profile"]["length"], texts["lengths"]
                    current = _length_choice(grid)
                else:
                    belief = prefs.posterior(counts[scope], dimension)
                    entry = texts["dimensions"][dimension]
                    name, names = entry["name"], entry["values"]
                    current = max(belief, key=belief.get)
                rows.append(
                    ROW.substitute(
                        name=html.escape(name),
                        scope=html.escape(
                            await _scope_name(conn, scope, texts)
                        ),
                        scope_id=html.escape(scope),
                        dimension=dimension,
                        options=_options(names, current),
                        save=html.escape(texts["profile"]["save"]),
                    )
                )
    return TEMPLATE.substitute(
        language=html.escape(user["language"] or i18n.FALLBACK),
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
    async with db.pool.connection() as conn, conn.transaction():
        user = await _user(conn, key)
        valid = (
            value in vocab.LengthChoice
            if dimension == vocab.LENGTH
            else value in vocab.DIMENSION_VALUES.get(dimension, ())
        )
        scopes = (await _scopes(conn, user["id"])).get(dimension, [])
        if not valid or scope not in scopes:
            raise fastapi.HTTPException(status_code=422)
        edit = await db.fetchrow(
            conn,
            "INSERT INTO profile_edits (user_id, scope, dimension, value) "
            "VALUES (%s, %s, %s, %s) RETURNING *",
            user["id"],
            scope,
            dimension,
            value,
        )
        await engine.apply_edit(conn, edit, await params_lib.cache.get(conn))
    return responses.RedirectResponse(f"/me/{key}", status_code=303)
