"""Topic domains learned from embeddings (SPEC 7.1).

Topics are embedded locally with a multilingual sentence model, clustered
nightly by spherical k-means, and each topic belongs to its nearest centroid.
"""

import asyncio
import functools
import os

import fastembed
import numpy as np
from scipy import optimize
from sklearn import cluster
from sklearn import metrics
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import models

MIN_TOPICS = 20  # Below this, every topic is in one domain.
MAX_CLUSTERS = 30
DOMAIN_PREFIX = "topic:"
ANY_DOMAIN = DOMAIN_PREFIX + "any"


@functools.cache
def _model() -> fastembed.TextEmbedding:
    return fastembed.TextEmbedding(
        config.settings.embedding_model,
        cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
    )


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


async def embed(topic: str) -> np.ndarray:
    """Returns the unit-norm embedding of a topic."""
    vectors = await asyncio.to_thread(lambda: list(_model().embed([topic])))
    return _unit(np.asarray(vectors[0], dtype=np.float32))


def domain_scope(cluster_id: int | None) -> str:
    """Returns the preference scope of a topic cluster."""
    return ANY_DOMAIN if cluster_id is None else f"{DOMAIN_PREFIX}{cluster_id}"


async def domain_of(
    session: sa_asyncio.AsyncSession, embedding: np.ndarray
) -> str:
    """Returns the scope of the cluster nearest to an embedding."""
    cluster_id = await session.scalar(
        sa.select(models.TopicCluster.id)
        .order_by(models.TopicCluster.centroid.cosine_distance(embedding))
        .limit(1)
    )
    return domain_scope(cluster_id)


def _best_k_means(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns unit centroids and labels, k chosen by the cosine silhouette."""
    best_score, best = -1.0, None
    for k in range(2, min(MAX_CLUSTERS, int(len(vectors) ** 0.5)) + 1):
        model = cluster.KMeans(k, n_init=10, random_state=0).fit(vectors)
        score = metrics.silhouette_score(
            vectors, model.labels_, metric="cosine"
        )
        if score > best_score:
            best_score, best = score, model
    return _unit(best.cluster_centers_), best.labels_


def stable_ids(new: np.ndarray, old: dict[int, np.ndarray]) -> list[int]:
    """Matches new centroids to old ids by cosine similarity (Hungarian).

    Args:
        new: The new unit centroids, one per row.
        old: The previous centroids by id.

    Returns:
        An id for each new centroid: the matched old id, or a fresh one.
    """
    ids = [-1] * len(new)
    if old:
        old_ids = list(old)
        similarity = new @ np.stack([old[i] for i in old_ids]).T
        rows, cols = optimize.linear_sum_assignment(-similarity)
        for r, c in zip(rows, cols):
            ids[r] = old_ids[c]
    fresh = max(old, default=0) + 1
    for i, assigned in enumerate(ids):
        if assigned < 0:
            ids[i], fresh = fresh, fresh + 1
    return ids


async def recluster(session: sa_asyncio.AsyncSession) -> int:
    """Refits the topic clusters on every embedding; returns their number."""
    rows = (
        await session.execute(
            sa.select(models.TopicEmbedding.embedding, models.Turn.topic).join(
                models.Turn, models.Turn.id == models.TopicEmbedding.turn_id
            )
        )
    ).all()
    old = {
        c.id: np.asarray(c.centroid)
        for c in await session.scalars(sa.select(models.TopicCluster))
    }
    await session.execute(sa.delete(models.TopicCluster))
    if len(rows) < MIN_TOPICS:
        return 0
    vectors = _unit(np.stack([np.asarray(r.embedding) for r in rows]))
    centroids, labels = _best_k_means(vectors)
    for index, cluster_id in enumerate(stable_ids(centroids, old)):
        members = np.flatnonzero(labels == index)
        medoid = members[np.argmax(vectors[members] @ centroids[index])]
        session.add(
            models.TopicCluster(
                id=cluster_id,
                centroid=centroids[index],
                medoid_topic=rows[medoid].topic,
                size=len(members),
            )
        )
    await session.flush()
    return len(centroids)
