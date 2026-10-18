"""Frequent follow-up questions (SPEC 4.8): what people asked next.

Nightly, the topics of each language are grouped into questions: topics
closer than SAME_QUESTION in cosine distance ask the same thing (average
linkage). Consecutive turns of a conversation give the follows between
questions, counted in distinct users. Online, the question nearest to a topic
offers the follow-up with the best score: how many people asked it next,
times a Thompson draw of its click rate. A question is shown only once
MIN_USERS different people asked it, in its typical wording, so nothing one
person wrote reaches another.
"""

import random

import numpy as np
import pandas as pd
from sklearn import cluster
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import models
from app import topics
from app.model import matching

SAME_QUESTION = 0.3  # Cosine distance under which two topics ask the same.
MIN_USERS = 5  # Distinct people who asked a question before it is shown.
Session = sa_asyncio.AsyncSession


def _group(vectors: np.ndarray) -> np.ndarray:
    """Returns the question label of each topic of one language."""
    if len(vectors) == 1:
        return np.zeros(1, dtype=int)
    return cluster.AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=SAME_QUESTION,
        metric="cosine",
        linkage="average",
    ).fit_predict(vectors)


async def _turns(session: Session) -> pd.DataFrame:
    t, e = models.Turn, models.TopicEmbedding
    rows = (
        await session.execute(
            sa.select(
                t.id,
                t.user_id,
                t.conversation_id,
                t.turn_index,
                t.language,
                t.topic,
                e.embedding,
            ).join(e, e.turn_id == t.id)
        )
    ).all()
    return pd.DataFrame(
        rows,
        columns=[
            "id",
            "user_id",
            "conversation_id",
            "turn_index",
            "language",
            "topic",
            "embedding",
        ],
    )


async def _offers(session: Session) -> pd.DataFrame:
    """Returns every frequent question offered: its turn and if clicked."""
    b, c = models.ButtonShown, models.Click
    clicked = sa.exists().where(
        c.turn_id == b.turn_id, c.position == b.position
    )
    rows = (
        await session.execute(
            sa.select(b.turn_id, b.question_id, clicked.label("clicked")).where(
                b.question_id.is_not(None)
            )
        )
    ).all()
    return pd.DataFrame(rows, columns=["id", "next_id", "clicked"])


async def rebuild(session: Session) -> int:
    """Regroups every topic into questions and counts the follows.

    Args:
        session: A session inside the nightly transaction.

    Returns:
        The number of questions.
    """
    frame = await _turns(session)
    old = list(await session.scalars(sa.select(models.Question)))
    await session.execute(sa.delete(models.QuestionFollow))
    await session.execute(sa.delete(models.Question))
    if frame.empty:
        return 0
    vectors = topics.unit(np.stack(frame.embedding.map(np.asarray)))
    groups = []  # (language, member positions)
    for language, members in frame.groupby("language").indices.items():
        labels = _group(vectors[members])
        groups += [(language, members[labels == g]) for g in np.unique(labels)]
    centroids = topics.unit(np.stack([vectors[m].mean(0) for _, m in groups]))
    similarity = np.full((len(groups), len(old)), -np.inf)
    for j, q in enumerate(old):
        same = [
            i
            for i, (language, _) in enumerate(groups)
            if language == q.language
        ]
        similarity[same, j] = centroids[same] @ np.asarray(q.centroid)
    ids = matching.stable_ids(
        similarity, [q.id for q in old], minimum=1 - SAME_QUESTION
    )
    question = np.empty(len(frame), dtype=int)
    for (language, members), question_id, centroid in zip(
        groups, ids, centroids
    ):
        question[members] = question_id
        medoid = members[np.argmax(vectors[members] @ centroid)]
        session.add(
            models.Question(
                id=question_id,
                language=language,
                text=frame.topic.iat[medoid],
                users=frame.user_id.iloc[members].nunique(),
                centroid=centroid,
            )
        )
    frame["question_id"] = question
    after = frame[["conversation_id", "turn_index", "question_id"]].assign(
        turn_index=frame.turn_index - 1
    )
    pairs = frame.merge(
        after.rename(columns={"question_id": "next_id"}),
        on=["conversation_id", "turn_index"],
    )
    pairs = pairs[pairs.question_id != pairs.next_id]
    keys = ["question_id", "next_id"]
    follows = pairs.groupby(keys).user_id.nunique().rename("users").to_frame()
    offers = (
        (await _offers(session))
        .merge(frame[["id", "question_id"]], on="id")
        .groupby(keys)
        .clicked.agg(shown="size", clicked="sum")
    )
    follows = follows.join(offers).fillna(0).astype(int).reset_index()
    session.add_all(
        models.QuestionFollow(**row) for row in follows.to_dict("records")
    )
    await session.flush()
    return len(groups)


async def follow_up(
    session: Session, embedding: np.ndarray, language: str, rng: random.Random
) -> dict | None:
    """Returns the frequent question to offer after a topic, if any.

    Args:
        session: A database session.
        embedding: The topic's unit embedding.
        language: The language of the conversation turn.
        rng: The random source of the Thompson draws.

    Returns:
        {"id", "text"} of the best follow-up, or None.
    """
    q, f = models.Question, models.QuestionFollow
    distance = q.centroid.cosine_distance(embedding)
    source = await session.scalar(
        sa.select(q.id)
        .where(q.language == language, distance <= SAME_QUESTION)
        .order_by(distance)
        .limit(1)
    )
    if source is None:
        return None
    rows = (
        await session.execute(
            sa.select(f, q)
            .join(q, q.id == f.next_id)
            .where(
                f.question_id == source,
                q.language == language,
                q.users >= MIN_USERS,
            )
        )
    ).all()
    if not rows:
        return None
    _, best = max(
        rows,
        key=lambda r: r[0].users
        * rng.betavariate(1 + r[0].clicked, 1 + r[0].shown - r[0].clicked),
    )
    return {"id": best.id, "text": best.text}
