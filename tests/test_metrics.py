import pytest

from metrics import bootstrap_mean_ci, ndcg_at_k, recall_at_k


def test_perfect_ranking_has_ndcg_one() -> None:
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 5) == pytest.approx(1.0)
    assert recall_at_k(["a", "b"], {"a", "b", "c"}, 5) == pytest.approx(2 / 3)


def test_relevant_item_at_rank_three() -> None:
    assert ndcg_at_k(["x", "y", "relevant", "z"], {"relevant"}, 5) == pytest.approx(0.5)


def test_no_relevant_items_returns_zero() -> None:
    assert recall_at_k(["a", "b"], set(), 5) == 0.0
    assert ndcg_at_k(["a", "b"], set(), 5) == 0.0


def test_fewer_results_than_cutoff() -> None:
    assert recall_at_k(["a"], {"a", "b"}, 10) == pytest.approx(0.5)
    assert ndcg_at_k(["a"], {"a", "b"}, 10) == pytest.approx(
        1 / (1 + 1 / 1.584962500721156)
    )


def test_bootstrap_interval_is_exact_for_constant_paired_differences() -> None:
    assert bootstrap_mean_ci([0.25, 0.25, 0.25], seed=7) == (0.25, 0.25, 0.25)


def test_bootstrap_interval_is_reproducible() -> None:
    values = [0.3, -0.1, 0.0, 0.2]
    assert bootstrap_mean_ci(values, seed=17) == bootstrap_mean_ci(values, seed=17)
