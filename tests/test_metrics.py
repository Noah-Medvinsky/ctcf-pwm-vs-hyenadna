"""Metric tests: compare against small hand-calculated examples (design spec, Testing Plan)."""

import pytest

from ctcf.metrics import bootstrap_ci, choose_threshold, evaluate

# 4 positives, 4 negatives. Ranking by score: P P N P N P N N
Y = [1, 1, 0, 1, 0, 1, 0, 0]
S = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]


def test_auroc_by_hand():
    # Count (positive, negative) pairs ranked correctly: 4 + 4 + 3 + 2 = 13 of 16.
    assert evaluate(Y, S, 0.5)["auroc"] == pytest.approx(13 / 16)


def test_threshold_metrics_by_hand():
    # Cutoff 0.55 predicts the top 4 as positive: P P N P -> TP=3, FP=1, FN=1.
    m = evaluate(Y, S, 0.55)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)


def test_pr_auc_by_hand():
    # Average precision = mean precision at each positive: (1/1 + 2/2 + 3/4 + 4/6) / 4.
    assert evaluate(Y, S, 0.5)["pr_auc"] == pytest.approx((1 + 1 + 3 / 4 + 4 / 6) / 4)


def test_choose_threshold_maximizes_f1():
    # Best F1 here is at cutoff 0.4 (all 4 positives caught with 2 false positives: F1 = 0.8).
    assert choose_threshold(Y, S) == pytest.approx(0.4)


def test_perfect_and_random_scores():
    assert evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)["auroc"] == 1.0
    lo, hi = bootstrap_ci(Y, S, n_boot=200)
    assert lo <= 13 / 16 <= hi
