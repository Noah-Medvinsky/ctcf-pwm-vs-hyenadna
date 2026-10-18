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


@needs_data
def test_no_reverse_complement_overlap_with_test(data):
    """A training sequence's reverse complement in the test set would also leak."""
    test = set(data["test"][0])
    train_rc = {reverse_complement(s) for s in data["train"][0]}
    assert not train_rc & test
