"""Evaluation metrics shared by every method.

Precision, recall and F1 need a cutoff. To keep the test set untouched, the cutoff is
chosen on the validation set (the score that maximizes F1 there) and then applied to
the test set.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def choose_threshold(y_valid, score_valid):
    """Score cutoff that maximizes F1 on the validation set."""
    precision, recall, thresholds = precision_recall_curve(y_valid, score_valid)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def evaluate(y, score, threshold):
    """AUROC, PR-AUC, and precision/recall/F1 at the given cutoff."""
    pred = (np.asarray(score) >= threshold).astype(int)
    return {
        "auroc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "threshold": threshold,
    }


def bootstrap_ci(y, score, metric=roc_auc_score, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval: resample test sequences with replacement."""
    y, score = np.asarray(y), np.asarray(score)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        stats.append(metric(y[idx], score[idx]))
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
