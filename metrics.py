"""Shared binary-relevance ranking metrics for API and command-line evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from collections.abc import Set as AbstractSet
from math import log2


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int
) -> float:
    """Return the fraction of all known relevant IDs retrieved within K."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def dcg_at_k(
    ranked_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int
) -> float:
    """Compute binary-gain DCG using the spec's log2(rank + 1) discount."""
    if k < 1:
        raise ValueError("k must be at least 1")
    relevant = set(relevant_ids)
    return sum(
        (1.0 if item_id in relevant else 0.0) / log2(rank + 1)
        for rank, item_id in enumerate(ranked_ids[:k], start=1)
    )


def ndcg_at_k(
    ranked_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int
) -> float:
    """Return normalized DCG, or zero when the query has no relevant labels."""
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant_ids:
        return 0.0
    ideal_hits = min(len(relevant_ids), k)
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg_at_k(ranked_ids, relevant_ids, k) / ideal_dcg if ideal_dcg else 0.0


def ranking_metrics(
    ranked_ids: Sequence[str], relevant_ids: AbstractSet[str], k: int
) -> dict[str, float]:
    """Calculate Recall@K and NDCG@K together."""
    return {
        f"recall@{k}": recall_at_k(ranked_ids, relevant_ids, k),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, relevant_ids, k),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return the sample mean and percentile bootstrap interval for a mean.

    Resampling is deterministic for a fixed seed and samples queries with
    replacement, keeping each query as one paired observation.
    """
    if not values:
        raise ValueError("values must contain at least one observation")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    observations = [float(value) for value in values]
    mean = sum(observations) / len(observations)
    generator = random.Random(seed)
    samples = sorted(
        sum(generator.choices(observations, k=len(observations))) / len(observations)
        for _ in range(n_resamples)
    )
    tail = (1.0 - confidence) / 2.0

    def percentile(fraction: float) -> float:
        position = fraction * (len(samples) - 1)
        lower = int(position)
        upper = min(lower + 1, len(samples) - 1)
        weight = position - lower
        return samples[lower] * (1.0 - weight) + samples[upper] * weight

    return mean, percentile(tail), percentile(1.0 - tail)
