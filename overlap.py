"""Set agreement statistics for two ranked retrieval result lists."""

from __future__ import annotations

from collections.abc import Sequence


def compute_overlap(
    hybrid_ids: Sequence[int], bm25_ids: Sequence[int]
) -> dict[str, int | float]:
    """Return overlap statistics; overlap is agreement, not retrieval accuracy."""
    hybrid = set(hybrid_ids)
    bm25 = set(bm25_ids)
    shared_count = len(hybrid & bm25)
    denominator = max(len(hybrid_ids), len(bm25_ids), 1)
    return {
        "k": denominator,
        "shared_count": shared_count,
        "overlap_pct": round(100.0 * shared_count / denominator, 1),
        "hybrid_only_count": len(hybrid - bm25),
        "bm25_only_count": len(bm25 - hybrid),
    }
