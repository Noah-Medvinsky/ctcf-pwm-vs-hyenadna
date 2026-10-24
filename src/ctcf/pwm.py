"""Position weight matrix (PWM) baseline built from JASPAR motifs.

Biopython turns the JASPAR count matrix into a log-odds position-specific scoring
matrix (PSSM), using JASPAR's standard pseudocounts and a uniform background. Each
sequence is then scored at every window on both strands, and its PWM score is the
best window score.

Scoring works directly on one-hot arrays (N, 4, L) with channel order A, C, G, T, so
the reverse strand is the PSSM with both axes flipped.
"""

import numpy as np
from Bio import motifs
from Bio.motifs import jaspar

from ctcf.data import ALPHABET, ROOT

MOTIF_IDS = {
    "CTCF": "MA0139.2", "ATF2": "MA1632.2", "Arid3": "MA0151.1", "BACH1": "MA1633.2",
    "ELK1": "MA0028.3", "GABPA": "MA0062.4", "MAX": "MA0058.4", "REST": "MA0138.3",
    "SRF": "MA0083.3", "ZNF24": "MA1124.1",
}


def load_motif(motif_id):
    with open(ROOT / "data" / "raw" / f"{motif_id}.jaspar") as handle:
        return motifs.read(handle, "jaspar")


def pssm_matrix(motif):
    """Log-odds PSSM as an array of shape (4, motif length), rows in A, C, G, T order."""
    motif.pseudocounts = jaspar.calculate_pseudocounts(motif)
    pssm = motif.pssm
    return np.array([pssm[base] for base in ALPHABET])


def window_scores(onehot, pssm):
    """Log-odds score of every window on both strands: shape (N, 2, L - k + 1)."""
    onehot = np.asarray(onehot, dtype=np.float32)
    if onehot.ndim != 3 or onehot.shape[1] != 4:
        raise ValueError(f"expected one-hot array of shape (N, 4, L), got {onehot.shape}")
    k = pssm.shape[1]
    if onehot.shape[2] < k:
        raise ValueError(f"sequences ({onehot.shape[2]} nt) are shorter than the motif ({k} nt)")
    if not np.all(onehot.sum(axis=1) == 1):
        raise ValueError("every position must be exactly one of A, C, G, T")
    windows = np.lib.stride_tricks.sliding_window_view(onehot, k, axis=2)  # (N, 4, positions, k)
    forward = np.einsum("nbpk,bk->np", windows, pssm)
    reverse = np.einsum("nbpk,bk->np", windows, pssm[::-1, ::-1])
    return np.stack([forward, reverse], axis=1)


def score_sequences(onehot, pssm, hit_fraction=0.8):
    """Per-sequence PWM results.

    Returns a dict of arrays: best score, best window position, strand of the best
    window (0 forward, 1 reverse), and the number of motif hits, counting windows
    whose relative score is at least `hit_fraction` of the way from the PSSM's
    minimum to its maximum possible score.
    """
    scores = window_scores(onehot, pssm)
    n = scores.shape[0]
    flat = scores.reshape(n, -1)
    best = flat.argmax(axis=1)
    lo, hi = pssm.min(axis=0).sum(), pssm.max(axis=0).sum()
    threshold = lo + hit_fraction * (hi - lo)
    return {
        "score": flat[np.arange(n), best],
        "best_pos": best % scores.shape[2],
        "best_strand": best // scores.shape[2],
        "n_hits": (scores >= threshold).sum(axis=(1, 2)),
    }
