"""Score the PWM baseline on each TF's test set.

For each TF this writes per-sequence test results (score, best window position and
strand, number of motif hits) to results/predictions/{TF}_pwm.csv, and one summary
row per TF to results/tables/pwm_summary.csv. The F1 cutoff is chosen on the
validation set. AUROC gets a 95% bootstrap confidence interval.

Usage:
    python scripts/score_pwm.py              # all 10 TFs
    python scripts/score_pwm.py --tf CTCF GABPA
"""

import argparse
import time

import pandas as pd

from ctcf.data import ROOT, load_onehot
from ctcf.metrics import bootstrap_ci, choose_threshold, evaluate
from ctcf.pwm import MOTIF_IDS, load_motif, pssm_matrix, score_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tf", nargs="+", default=sorted(MOTIF_IDS))
    args = parser.parse_args()

    pred_dir = ROOT / "results" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for tf in args.tf:
        path = ROOT / "data" / "raw" / f"{tf}_200.h5"
        motif = load_motif(MOTIF_IDS[tf])
        pssm = pssm_matrix(motif)

        x_valid, y_valid = load_onehot("valid", path)
        x_test, y_test = load_onehot("test", path)
        start = time.perf_counter()
        test = score_sequences(x_test, pssm)
        seconds = time.perf_counter() - start
        threshold = choose_threshold(y_valid, score_sequences(x_valid, pssm)["score"])

        pd.DataFrame({"index": range(len(y_test)), "label": y_test, **test}).to_csv(
            pred_dir / f"{tf}_pwm.csv", index=False)
        lo, hi = bootstrap_ci(y_test, test["score"])
        row = {"tf": tf, "motif": MOTIF_IDS[tf], "motif_length": motif.length, "n_test": len(y_test),
               **evaluate(y_test, test["score"], threshold), "auroc_ci_low": lo, "auroc_ci_high": hi,
               "scoring_seconds": round(seconds, 3)}
        rows.append(row)
        print(f"{tf:6s} {MOTIF_IDS[tf]} ({motif.length:2d} bp)  AUROC {row['auroc']:.3f} [{lo:.3f}, {hi:.3f}]  "
              f"PR-AUC {row['pr_auc']:.3f}  F1 {row['f1']:.3f}  ({seconds:.2f} s)")

    summary = ROOT / "results" / "tables" / "pwm_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    if summary.exists():  # keep rows for TFs not rescored in this run
        old = pd.read_csv(summary)
        rows = old[~old["tf"].isin(args.tf)].to_dict("records") + rows
    pd.DataFrame(rows).sort_values("tf").to_csv(summary, index=False)


if __name__ == "__main__":
    main()
