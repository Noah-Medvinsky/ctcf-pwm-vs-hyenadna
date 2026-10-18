"""Load the CTCF ChIP-seq dataset from Tang et al. (2025).

The h5 file stores one-hot sequences as x_{split} with shape (N, 4, 200)
(channel order A, C, G, T) and labels as y_{split} with shape (N, 1),
using the authors' fixed train/valid/test split.
"""

from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CTCF_H5 = ROOT / "data" / "raw" / "CTCF_200.h5"
SPLITS = ("train", "valid", "test")
ALPHABET = "ACGT"

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def onehot_to_seq(onehot):
    """Convert one-hot arrays of shape (N, 4, L) to strings; all-zero columns become N."""
    onehot = np.asarray(onehot)
    idx = onehot.argmax(axis=1)
    idx[onehot.sum(axis=1) == 0] = 4
    lookup = np.array(list(ALPHABET + "N"))
    return ["".join(row) for row in lookup[idx]]


def reverse_complement(seq):
    return seq.translate(_COMPLEMENT)[::-1]


def load_onehot(split, path=CTCF_H5):
    """Return (x, y): one-hot array (N, 4, 200) and int labels (N,)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    with h5py.File(path, "r") as f:
        x = f[f"x_{split}"][()]
        y = f[f"y_{split}"][()].ravel().astype(int)
    return x, y


def load_split(split, path=CTCF_H5):
    """Return (sequences, labels) for one split, with sequences as strings."""
    x, y = load_onehot(split, path)
    return onehot_to_seq(x), y
