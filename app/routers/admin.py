"""Admin API: gold labels and metrics, behind the admin token."""

import fastapi
import pydantic

from app import models
from app.core import db
from app.core import vocab
from app.routers import dashboard
from app.services import metrics
from app.services import replies

router = fastapi.APIRouter(dependencies=[fastapi.Depends(dashboard.admin)])


class GoldLabel(pydantic.BaseModel):
    turn_id: str = pydantic.Field(pattern=rf"^{replies.TURN_PREFIX}\d+$")
    outcome: vocab.Outcome


@router.post("/gold_label")
async def gold_label(label: GoldLabel) -> dict:
    async with db.session() as session, session.begin():
        session.add(
            models.GoldLabel(
                turn_id=int(label.turn_id.removeprefix(replies.TURN_PREFIX)),
                outcome=label.outcome,
            )
        )
    return {"ok": True}


@router.get("/metrics")
async def get_metrics() -> dict:
    async with db.session() as session:
        return await metrics.compute(session)
