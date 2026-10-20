"""Data requests: erasing everything known about a person.

A person asks by email, with the code their widget showed; the owner
processes it from the dashboard. What remains is the bare user row (an
internal number) and the record of the request, which the law requires.
"""

import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app.core import crypto
from app import models

Session = sa_asyncio.AsyncSession
# Rows about one turn, deleted before their turns.
TURN_ROWS = (
    models.Reply,
    models.Message,
    models.Goal,
    models.Outcome,
    models.WidgetEvent,
    models.GoldLabel,
    models.TurnContext,
    models.TopicEmbedding,
    models.ShadowDecision,
    models.TurnInference,
)


class UnknownCodeError(Exception):
    """The code is forged, or names no one."""


async def erase(session: Session, code: str) -> int:
    """Erases the person a privacy code names; returns their user id.

    Their connector stops working: its grant is revoked and its credentials
    deleted. The nightly replay then rebuilds the population without them.

    Raises:
        UnknownCodeError: When the code is forged or names no one.
    """
    user_id = crypto.user_of_code(code)
    if user_id is None or not await session.get(models.User, user_id):
        raise UnknownCodeError
    request = models.DeletionRequest(user_id=user_id)
    session.add(request)
    turns = sa.select(models.Turn.id).where(models.Turn.user_id == user_id)
    for model in TURN_ROWS:
        await session.execute(sa.delete(model).where(model.turn_id.in_(turns)))
    conversations = sa.select(models.Conversation.id).where(
        models.Conversation.user_id == user_id
    )
    await session.execute(
        sa.delete(models.ConversationState).where(
            models.ConversationState.conversation_id.in_(conversations)
        )
    )
    for model in (
        models.Turn,
        models.Conversation,
        models.ToolCall,
        models.LengthPosterior,
        models.UserReading,
        models.PrefCount,
    ):
        await session.execute(sa.delete(model).where(model.user_id == user_id))
    grants = sa.select(models.OAuthGrant.id).where(
        models.OAuthGrant.user_id == user_id
    )
    await session.execute(
        sa.delete(models.OAuthCredential).where(
            models.OAuthCredential.grant_id.in_(grants)
        )
    )
    await session.execute(
        sa.update(models.OAuthGrant)
        .where(models.OAuthGrant.user_id == user_id)
        .values(revoked_at=sa.func.now())
    )
    await session.flush()
    request.done_at = await session.scalar(sa.select(sa.func.now()))
    return user_id
