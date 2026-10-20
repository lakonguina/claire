"""Page views of the landing and consent pages: the acquisition funnel."""

from app.core import db
from app import models

UTM = ("utm_source", "utm_medium", "utm_campaign", "utm_content", "utm_term")
MAX_CHARS = 200


def _clip(value: str | None) -> str | None:
    return value[:MAX_CHARS] if value else None


async def record(page: str, referrer: str | None, query: dict) -> None:
    """Records a visit: the page, where it came from, its UTM parameters.

    No address, cookie or identifier: a visit cannot be tied to anyone.
    """
    utm = {k: _clip(query[k]) for k in UTM if query.get(k)}
    async with db.session() as session, session.begin():
        session.add(models.Visit(page=page, referrer=_clip(referrer), utm=utm))
