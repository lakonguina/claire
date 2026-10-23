"""Admin pages: users, conversations, readable turns + JSON.

Gated by ``ADMIN_TOKEN``. Empty token → every route 404s.
"""

from __future__ import annotations

from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app import models
from app.admin import views
from app.core import config, db
from app.jobs import nightly

router = APIRouter(prefix="/admin", tags=["admin"])
_TEMPLATES = Jinja2Templates(
    directory=Path(__file__).resolve().parent.parent / "templates"
)
_COOKIE = "claire_admin"


def _required_token() -> str:
    token = (config.settings.admin_token or "").strip()
    if not token:
        raise HTTPException(status_code=404)
    return token


def _gate(request: Request) -> Response | None:
    """Return a login redirect when the admin cookie is missing/wrong."""
    expected = _required_token()
    if request.cookies.get(_COOKIE) == expected:
        return None
    return RedirectResponse(url="/admin/login", status_code=303)


async def _conversation_events(
    session, *, person_id: int, conversation_id: str
) -> list[models.Event]:
    """All events for a chat, including replies linked only by turn_id."""
    base = list(
        await session.scalars(
            sa.select(models.Event)
            .where(
                models.Event.person_id == person_id,
                models.Event.conversation_id == conversation_id,
            )
            .order_by(models.Event.id.asc())
        )
    )
    turn_ids = {event.turn_id for event in base if event.turn_id}
    if not turn_ids:
        return base
    extra = list(
        await session.scalars(
            sa.select(models.Event)
            .where(
                models.Event.person_id == person_id,
                models.Event.turn_id.in_(turn_ids),
                sa.or_(
                    models.Event.conversation_id.is_(None),
                    models.Event.conversation_id != conversation_id,
                ),
            )
            .order_by(models.Event.id.asc())
        )
    )
    by_id = {event.id: event for event in base}
    for event in extra:
        by_id[event.id] = event
    return sorted(by_id.values(), key=lambda event: event.id)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    _required_token()
    return _TEMPLATES.TemplateResponse(
        request=request,
        name="admin/login.html",
        context={"error": None},
    )


@router.post("/login", response_model=None)
async def login_submit(request: Request, token: str = Form(...)) -> Response:
    expected = _required_token()
    if token.strip() != expected:
        return _TEMPLATES.TemplateResponse(
            request=request,
            name="admin/login.html",
            context={"error": "Wrong token."},
            status_code=401,
        )
    response = RedirectResponse(url="/admin", status_code=303)
    response.set_cookie(
        _COOKIE,
        expected,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 14,
    )
    return response


@router.post("/logout")
async def logout() -> RedirectResponse:
    _required_token()
    response = RedirectResponse(url="/admin/login", status_code=303)
    response.delete_cookie(_COOKIE)
    return response


@router.post("/train")
async def train(request: Request) -> Response:
    """Replay the event log into person bandit caches (same as nightly)."""
    blocked = _gate(request)
    if blocked is not None:
        return blocked

    async with db.Session() as session, session.begin():
        report = await nightly.run(session)

    return RedirectResponse(
        url=(
            "/admin"
            f"?trained=1"
            f"&people={report['people']}"
            f"&matched={report['matched']}"
            f"&rewritten={report['rewritten']}"
            f"&closed={report['closed_turns']}"
        ),
        status_code=303,
    )


@router.get("", response_class=HTMLResponse)
async def people_list(request: Request) -> Response:
    blocked = _gate(request)
    if blocked is not None:
        return blocked

    train_report = None
    if request.query_params.get("trained") == "1":
        train_report = {
            "people": request.query_params.get("people", "0"),
            "matched": request.query_params.get("matched", "0"),
            "rewritten": request.query_params.get("rewritten", "0"),
            "closed": request.query_params.get("closed", "0"),
        }

    async with db.Session() as session:
        event_count = (
            sa.select(
                models.Event.person_id,
                sa.func.count(sa.distinct(models.Event.conversation_id)).label(
                    "conversations"
                ),
                sa.func.max(models.Event.created_at).label("last_event_at"),
            )
            .where(models.Event.conversation_id.is_not(None))
            .group_by(models.Event.person_id)
            .subquery()
        )
        rows = (
            await session.execute(
                sa.select(
                    models.Person,
                    event_count.c.conversations,
                    event_count.c.last_event_at,
                )
                .outerjoin(
                    event_count, models.Person.id == event_count.c.person_id
                )
                .order_by(
                    sa.desc(
                        sa.func.coalesce(
                            event_count.c.last_event_at,
                            models.Person.created_at,
                        )
                    )
                )
            )
        ).all()
        people = [
            {
                "person": person,
                "conversations": int(conversations or 0),
                "last_event_at": last_event_at,
            }
            for person, conversations, last_event_at in rows
        ]

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="admin/people.html",
        context={"people": people, "train_report": train_report},
    )


@router.get("/people/{person_id}", response_class=HTMLResponse)
async def person_detail(request: Request, person_id: int) -> Response:
    blocked = _gate(request)
    if blocked is not None:
        return blocked

    async with db.Session() as session:
        person = await session.get(models.Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person not found")

        conv_ids = (
            await session.execute(
                sa.select(
                    models.Event.conversation_id,
                    sa.func.max(models.Event.created_at).label("last_at"),
                )
                .where(
                    models.Event.person_id == person_id,
                    models.Event.conversation_id.is_not(None),
                )
                .group_by(models.Event.conversation_id)
                .order_by(sa.desc("last_at"))
            )
        ).all()

        conversations = []
        for conversation_id, last_at in conv_ids:
            events = await _conversation_events(
                session,
                person_id=person_id,
                conversation_id=conversation_id,
            )
            preview, exchanges = views.conversation_preview(events)
            conversations.append(
                {
                    "id": conversation_id,
                    "preview": preview,
                    "exchanges": exchanges,
                    "last_at": last_at,
                }
            )

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="admin/person.html",
        context={"person": person, "conversations": conversations},
    )


@router.get(
    "/people/{person_id}/c/{conversation_id}",
    response_class=HTMLResponse,
)
async def conversation_detail(
    request: Request, person_id: int, conversation_id: str
) -> Response:
    blocked = _gate(request)
    if blocked is not None:
        return blocked

    async with db.Session() as session:
        person = await session.get(models.Person, person_id)
        if person is None:
            raise HTTPException(status_code=404, detail="person not found")

        events = await _conversation_events(
            session, person_id=person_id, conversation_id=conversation_id
        )
        if not events:
            raise HTTPException(
                status_code=404, detail="conversation not found"
            )
        turns = views.build_turns(events)

    return _TEMPLATES.TemplateResponse(
        request=request,
        name="admin/conversation.html",
        context={
            "person": person,
            "conversation_id": conversation_id,
            "turns": turns,
        },
    )
