"""HyenaDNA embedding tests (design spec, Testing Plan: embedding tests)."""

import numpy as np
import pytest

from ctcf.data import ALPHABET
from ctcf.hyena import CLS_ID, MODEL_DIR, SEP_ID, embed, load_model, tokenize

needs_weights = pytest.mark.skipif(not (MODEL_DIR / "weights.ckpt").exists(),
                                   reason="run scripts/embed_hyena.py once to download the weights")


def onehot(seqs):
    idx = np.array([[ALPHABET.index(c) for c in s] for s in seqs])
    return np.transpose(np.eye(4, dtype=np.float32)[idx], (0, 2, 1))


def random_onehot(n, length=200, seed=0):
    rng = np.random.default_rng(seed)
    return onehot(["".join(rng.choice(list(ALPHABET), length)) for _ in range(n)])


def test_tokenize_matches_character_tokenizer_vocabulary():
    # HyenaDNA's CharacterTokenizer: [CLS]=0, [SEP]=1, then A, C, G, T, N = 7..11.
    ids = tokenize(onehot(["ACGT"]))
    assert ids.tolist() == [[CLS_ID, 7, 8, 9, 10, SEP_ID]]


def test_tokenize_rejects_bad_input():
    with pytest.raises(ValueError):
        tokenize(np.zeros((2, 200)))
    x = random_onehot(1)
    x[0, :, 10] = 0  # an N: no base set at this position
    with pytest.raises(ValueError):
        tokenize(x)


@pytest.fixture(scope="module")
def model():
    return load_model()


@needs_weights
def test_embedding_shape(model):
    e = embed(model, random_onehot(3))
    assert e.shape == (3, 202, 256) and e.dtype == np.float32 and np.isfinite(e).all()


@needs_weights
def test_same_sequence_same_embedding(model):
    x = random_onehot(4, seed=1)
    np.testing.assert_allclose(embed(model, x), embed(model, x), atol=1e-6)


@needs_weights
def test_batched_matches_single(model):
    x = random_onehot(6, seed=2)
    batched = embed(model, x, batch_size=6)
    single = np.concatenate([embed(model, x[i:i + 1]) for i in range(6)])
    np.testing.assert_allclose(batched, single, atol=1e-5)


@needs_weights
def test_different_sequences_differ(model):
    e = embed(model, random_onehot(2, seed=3))
    assert not np.allclose(e[0], e[1])
