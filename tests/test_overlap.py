from overlap import compute_overlap


def test_identical_lists_have_full_overlap() -> None:
    assert compute_overlap([1, 2, 3], [1, 2, 3]) == {
        "k": 3,
        "shared_count": 3,
        "overlap_pct": 100.0,
        "hybrid_only_count": 0,
        "bm25_only_count": 0,
    }


def test_disjoint_lists_have_zero_overlap() -> None:
    result = compute_overlap([1, 2], [3, 4])
    assert result["overlap_pct"] == 0.0
    assert result["hybrid_only_count"] == 2
    assert result["bm25_only_count"] == 2


def test_empty_lists_are_safe() -> None:
    assert compute_overlap([], []) == {
        "k": 1,
        "shared_count": 0,
        "overlap_pct": 0.0,
        "hybrid_only_count": 0,
        "bm25_only_count": 0,
    }


def test_partial_overlap_uses_longer_result_length() -> None:
    result = compute_overlap([1, 2, 3], [3, 4])
    assert result == {
        "k": 3,
        "shared_count": 1,
        "overlap_pct": 33.3,
        "hybrid_only_count": 2,
        "bm25_only_count": 1,
    }
