"""Stable identities for clusters refit nightly (topic domains, families)."""

import numpy as np
from scipy import optimize


def stable_ids(similarity: np.ndarray, old_ids: list[int]) -> list[int]:
    """Matches new clusters to old ids (Hungarian), fresh ids for the rest.

    Args:
        similarity: new x old similarity matrix (higher is closer).
        old_ids: The id of each old cluster, one per column.

    Returns:
        An id for each new cluster: its matched old id, or a fresh one.
    """
    ids = [-1] * similarity.shape[0]
    if old_ids:
        rows, cols = optimize.linear_sum_assignment(-similarity)
        for r, c in zip(rows, cols):
            ids[r] = old_ids[c]
    fresh = max(old_ids, default=0) + 1
    for i, assigned in enumerate(ids):
        if assigned < 0:
            ids[i], fresh = fresh, fresh + 1
    return ids
