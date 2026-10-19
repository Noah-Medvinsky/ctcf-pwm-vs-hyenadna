"""Data integrity tests for the CTCF dataset (design spec, Testing Plan)."""

import numpy as np
import pytest

from ctcf.data import CTCF_H5, SPLITS, load_onehot, load_split, onehot_to_seq, reverse_complement

# Split sizes from the authors' CTCF_200.h5 (Zenodo 8279715).
EXPECTED_SIZES = {"train": 61_083, "valid": 8_726, "test": 17_453}

needs_data = pytest.mark.skipif(not CTCF_H5.exists(), reason="run scripts/download_data.py first")


@pytest.fixture(scope="module")
def data():
    return {split: load_split(split) for split in SPLITS}


def test_onehot_to_seq_roundtrip():
    x = np.zeros((1, 4, 5))
    for pos, base in enumerate([0, 1, 2, 3]):
        x[0, base, pos] = 1
    assert onehot_to_seq(x) == ["ACGTN"]


def test_reverse_complement():
    assert reverse_complement("AACGTN") == "NACGTT"


@needs_data
def test_split_sizes(data):
    assert {s: len(data[s][0]) for s in SPLITS} == EXPECTED_SIZES


@needs_data
@pytest.mark.parametrize("split", SPLITS)
def test_balanced_binary_labels(data, split):
    y = data[split][1]
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 0.5) < 0.01


@needs_data
@pytest.mark.parametrize("split", SPLITS)
def test_onehot_is_valid(split):
    x, _ = load_onehot(split)
    assert x.shape[1:] == (4, 200)
    assert set(np.unique(x)) <= {0.0, 1.0}
    assert (x.sum(axis=1) == 1).all(), "every position should be exactly one of A, C, G, T"


@needs_data
def test_no_exact_overlap_between_splits(data):
    sets = {s: set(data[s][0]) for s in SPLITS}
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["valid"]
    assert not sets["valid"] & sets["test"]


ALL_TFS = ["ATF2", "Arid3", "BACH1", "CTCF", "ELK1", "GABPA", "MAX", "REST", "SRF", "ZNF24"]


# The authors' splits for some TFs contain a few exact duplicate sequences across
# splits (all positives; at most 0.51% of a test set, GABPA). The split is kept as
# published so results stay comparable; this cap catches anything larger.
MAX_OVERLAP_FRAC = 0.01


@pytest.mark.parametrize("tf", ALL_TFS)
def test_every_tf_dataset(tf):
    """Same integrity checks for each transcription factor's file."""
    path = CTCF_H5.parent / f"{tf}_200.h5"
    if not path.exists():
        pytest.skip("run scripts/download_data.py first")
    seqs = {}
    for split in SPLITS:
        x, y = load_onehot(split, path)
        assert x.shape[1:] == (4, 200)
        assert (x.sum(axis=1) == 1).all()
        assert set(np.unique(y)) == {0, 1} and abs(y.mean() - 0.5) < 0.03
        seqs[split] = set(onehot_to_seq(x))
    seen = seqs["train"] | seqs["valid"] | {reverse_complement(s) for s in seqs["train"]}
    assert len(seqs["test"] & seen) / len(seqs["test"]) < MAX_OVERLAP_FRAC


@needs_data
def test_no_reverse_complement_overlap_with_test(data):
    """A training sequence's reverse complement in the test set would also leak."""
    test = set(data["test"][0])
    train_rc = {reverse_complement(s) for s in data["train"][0]}
    assert not train_rc & test
