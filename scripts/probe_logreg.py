"""Logistic-regression probe on mean HyenaDNA embeddings (this project's extension).

Fits L2-regularized logistic regression on the 256-number mean embedding of each
sequence. The regularization strength C and the F1 cutoff are both chosen on the
validation set; the test set is only used once, for the final scores.

Writes results/predictions/{TF}_hyena_logreg.csv and a row in
results/tables/logreg_summary.csv.

Usage:
    python scripts/probe_logreg.py --tf CTCF
"""

import argparse

import h5py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ctcf.data import ROOT, SPLITS
from ctcf.metrics import bootstrap_ci, choose_threshold, evaluate

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tf", default="CTCF")
    args = parser.parse_args()

    with h5py.File(ROOT / "data" / "processed" / f"{args.tf}_hyena.h5", "r") as f:
        data = {s: (f[f"x_mean_{s}"][()], f[f"y_{s}"][()].ravel().astype(int)) for s in SPLITS}

    best = None
    for c in C_GRID:
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
        model.fit(*data["train"])
        auroc = roc_auc_score(data["valid"][1], model.predict_proba(data["valid"][0])[:, 1])
        print(f"  C={c:<6} validation AUROC {auroc:.4f}")
        if best is None or auroc > best[0]:
            best = (auroc, c, model)
    _, c, model = best

    valid_score = model.predict_proba(data["valid"][0])[:, 1]
    test_x, test_y = data["test"]
    test_score = model.predict_proba(test_x)[:, 1]
    row = {"tf": args.tf, "C": c, **evaluate(test_y, test_score, choose_threshold(data["valid"][1], valid_score))}
    row["auroc_ci_low"], row["auroc_ci_high"] = bootstrap_ci(test_y, test_score)
    print(row)

    pd.DataFrame({"index": range(len(test_y)), "label": test_y, "prob": test_score}).to_csv(
        ROOT / "results" / "predictions" / f"{args.tf}_hyena_logreg.csv", index=False)
    summary = ROOT / "results" / "tables" / "logreg_summary.csv"
    rows = [row]
    if summary.exists():
        old = pd.read_csv(summary)
        rows = old[old["tf"] != args.tf].to_dict("records") + rows
    pd.DataFrame(rows).sort_values("tf").to_csv(summary, index=False)


if __name__ == "__main__":
    main()
