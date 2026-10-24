"""Train the Tang et al. (2025) baseline CNN on one-hot sequences or HyenaDNA embeddings.

Training settings follow chip-clip-seq/representation_perf.py: Adam (lr 1e-3), binary
cross-entropy, batch size 256, up to 100 epochs, early stopping on validation loss
(patience 10, best weights restored), and learning-rate decay x0.2 after 5 epochs
without improvement (min 1e-6).

For each seed this writes per-sequence test predictions and appends one summary row.
AUROC/PR-AUC are reported twice: with the Keras AUC metric the authors used
(200-threshold approximation) and with scikit-learn's exact computation.

Usage:
    python scripts/train_cnn.py --input onehot --seeds 0 1 2 3 4
    python scripts/train_cnn.py --input onehot --seeds 0 --subset 2000 --epochs 2   # smoke test
    python scripts/train_cnn.py --input hyena --h5 data/processed/CTCF_hyena.h5
"""

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras as keras
from sklearn.metrics import average_precision_score, roc_auc_score

from ctcf.data import ROOT, SPLITS, load_onehot
from ctcf.models import chip_cnn

BATCH_SIZE = 256


class Split:
    """One data split: labels in memory, inputs either in memory (one-hot) or streamed
    from the embedding file in chunks (HyenaDNA embeddings are too large for RAM)."""

    def __init__(self, args, split):
        if args.input == "onehot":
            x, y = load_onehot(split, ROOT / "data" / "raw" / f"{args.tf}_200.h5")
            self.x = np.transpose(x, (0, 2, 1)).astype("float32")  # (N, 4, 200) -> (N, 200, 4) for Conv1D
            self.h5 = None
        else:
            self.x, self.h5, self.key = None, args.h5, f"x_{split}"
            with h5py.File(args.h5, "r") as f:
                y = f[f"y_{split}"][()].ravel().astype(int)
                self.shape = f[self.key].shape[1:]
        self.y = y[:args.subset] if args.subset else y
        if self.x is not None:
            self.x = self.x[:len(self.y)]
            self.shape = self.x.shape[1:]

    def dataset(self, shuffle):
        if self.x is not None:
            ds = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        else:
            n, h5, key, labels = len(self.y), self.h5, self.key, self.y

            def chunks():
                with h5py.File(h5, "r") as f:
                    for i in range(0, n, 1024):
                        stop = min(i + 1024, n)
                        yield f[key][i:stop].astype("float32"), labels[i:stop]

            ds = tf.data.Dataset.from_generator(chunks, output_signature=(
                tf.TensorSpec((None, *self.shape), tf.float32), tf.TensorSpec((None,), tf.int64))).unbatch()
        if shuffle:  # same buffer as the authors: shuffle(256 * 4)
            ds = ds.shuffle(BATCH_SIZE * 4)
        return ds.batch(BATCH_SIZE).prefetch(2)


def load_inputs(args):
    return {split: Split(args, split) for split in SPLITS}


def train_one(data, seed, epochs):
    keras.utils.set_random_seed(seed)
    model = chip_cnn(data["train"].shape, 1)
    model.compile(
        loss=keras.losses.BinaryCrossentropy(from_logits=False),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy", keras.metrics.AUC(curve="ROC", name="auroc"),
                 keras.metrics.AUC(curve="PR", name="aupr")],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        data["train"].dataset(shuffle=True),
        validation_data=data["valid"].dataset(shuffle=False),
        epochs=epochs, verbose=2, callbacks=callbacks,
    )
    return model, len(history.history["loss"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tf", default="CTCF", help="transcription factor dataset name")
    parser.add_argument("--input", choices=["onehot", "hyena"], default="onehot")
    parser.add_argument("--h5", type=Path, help="embedding file for --input hyena")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--subset", type=int, help="use only the first N sequences per split (testing)")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.input == "hyena" and args.h5 is None:
        parser.error("--input hyena requires --h5")

    data = load_inputs(args)
    name = f"{args.tf}_{args.input}_cnn" + (f"_subset{args.subset}" if args.subset else "")
    pred_dir = args.out / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "tables" / f"{name}_runs.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    y_test = data["test"].y
    for seed in args.seeds:
        print(f"##### {name} seed {seed} #####")
        start = time.time()
        model, n_epochs = train_one(data, seed, args.epochs)
        _, acc, keras_auroc, keras_aupr = model.evaluate(data["test"].dataset(shuffle=False), verbose=0)
        prob = model.predict(data["test"].dataset(shuffle=False), verbose=0).ravel()
        pd.DataFrame({"index": np.arange(len(y_test)), "label": y_test, "prob": prob}).to_csv(
            pred_dir / f"{name}_seed{seed}.csv", index=False)
        row = {
            "tf": args.tf, "input": args.input, "seed": seed, "epochs_run": n_epochs,
            "accuracy": acc, "auroc_keras": keras_auroc, "aupr_keras": keras_aupr,
            "auroc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob),
            "train_seconds": round(time.time() - start, 1),
        }
        print(row)
        pd.DataFrame([row]).to_csv(summary_path, mode="a", header=not summary_path.exists(), index=False)

    runs = pd.read_csv(summary_path)
    print(runs[["auroc", "auroc_keras", "pr_auc"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
