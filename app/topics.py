"""Topic domains learned from embeddings (SPEC 7.1).

Topics are embedded locally with a multilingual sentence model, clustered
nightly by spherical k-means, and each topic belongs to its nearest centroid.
"""

import asyncio
import functools
import os

import fastembed
import numpy as np
from sklearn import cluster
from sklearn import metrics
import sqlalchemy as sa
from sqlalchemy.ext import asyncio as sa_asyncio

from app import config
from app import models
from app.model import matching

MIN_TOPICS = 20  # Below this, every topic is in one domain.
MAX_CLUSTERS = 30
DOMAIN_PREFIX = "topic:"
ANY_DOMAIN = DOMAIN_PREFIX + "any"
SOFTNESS = 0.05  # Softmax temperature of domain memberships.
MIN_MEMBERSHIP = 0.01  # Smaller memberships are dropped.


@functools.cache
def _model() -> fastembed.TextEmbedding:
    return fastembed.TextEmbedding(
        config.settings.embedding_model,
        cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
    )


def unit(rows: np.ndarray) -> np.ndarray:
    """Scales vectors to unit norm (cosine geometry)."""
    return rows / np.linalg.norm(rows, axis=-1, keepdims=True)


async def embed(topic: str) -> np.ndarray:
    """Returns the unit-norm embedding of a topic."""
    vectors = await asyncio.to_thread(lambda: list(_model().embed([topic])))
    return unit(np.asarray(vectors[0], dtype=np.float32))


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


async def memberships(
    session: sa_asyncio.AsyncSession, embedding: np.ndarray
) -> dict[str, float]:
    """Returns a topic's soft membership in every domain (sums to 1).

    A softmax of the cosine similarities: a topic between two domains
    belongs to both, so nothing learned has a hard frontier (SPEC 6.5).
    """
    clusters = (
        await session.execute(
            sa.select(models.TopicCluster.id, models.TopicCluster.centroid)
        )
    ).all()
    if not clusters:
        return {}
    centroids = np.stack([np.asarray(c.centroid) for c in clusters])
    logits = centroids @ embedding / SOFTNESS
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return {
        domain_scope(c.id): float(w)
        for c, w in zip(clusters, weights)
        if w >= MIN_MEMBERSHIP
    }


def _best_k_means(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns unit centroids and labels, k chosen by the cosine silhouette.

    k never exceeds the number of distinct topics: the same topic asked many
    times is one point.
    """
    distinct = len(np.unique(vectors, axis=0))
    k_max = min(MAX_CLUSTERS, int(len(vectors) ** 0.5), distinct)
    if k_max < 2:
        return unit(vectors.mean(0, keepdims=True)), np.zeros(len(vectors), int)
    best_score, best = -1.0, None
    for k in range(2, k_max + 1):
        model = cluster.KMeans(k, n_init=10, random_state=0).fit(vectors)
        score = metrics.silhouette_score(
            vectors, model.labels_, metric="cosine"
        )
        if score > best_score:
            best_score, best = score, model
    return unit(best.cluster_centers_), best.labels_


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
    vectors = unit(np.stack([np.asarray(r.embedding) for r in rows]))
    centroids, labels = _best_k_means(vectors)
    old_ids = list(old)
    similarity = (
        centroids @ np.stack([old[i] for i in old_ids]).T
        if old_ids
        else np.zeros((len(centroids), 0))
    )
    ids = matching.stable_ids(similarity, old_ids)
    for index, cluster_id in enumerate(ids):
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
