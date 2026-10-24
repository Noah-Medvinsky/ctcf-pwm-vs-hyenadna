"""PWM baseline tests (design spec, Testing Plan: PWM tests and invalid-input tests)."""

import numpy as np
import pytest
from Bio.Seq import Seq

from ctcf.data import ALPHABET, ROOT, reverse_complement
from ctcf.pwm import MOTIF_IDS, load_motif, pssm_matrix, score_sequences, window_scores

CTCF = MOTIF_IDS["CTCF"]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "raw" / f"{CTCF}.jaspar").exists(),
                                reason="run scripts/download_data.py first")


def onehot(seqs):
    idx = np.array([[ALPHABET.index(c) for c in s] for s in seqs])
    return np.transpose(np.eye(4, dtype=np.float32)[idx], (0, 2, 1))


@pytest.fixture(scope="module")
def ctcf():
    motif = load_motif(CTCF)
    return motif, pssm_matrix(motif)


def random_seqs(n, length=200, seed=0):
    rng = np.random.default_rng(seed)
    return ["".join(rng.choice(list(ALPHABET), length)) for _ in range(n)]


def test_pssm_matches_jaspar_counts(ctcf):
    motif, pssm = ctcf
    assert pssm.shape == (4, motif.length)
    # The base with the highest count at each position has the highest log-odds score.
    counts = np.array([motif.counts[b] for b in ALPHABET])
    assert (counts.argmax(axis=0) == pssm.argmax(axis=0)).all()


def test_matches_biopython_scanner(ctcf):
    motif, pssm = ctcf
    seq = random_seqs(1, seed=1)[0]
    ours = window_scores(onehot([seq]), pssm)[0]
    theirs = motif.pssm.calculate(Seq(seq))
    np.testing.assert_allclose(ours[0], theirs, rtol=1e-4, atol=1e-4)


def test_planted_motif_scores_top(ctcf):
    motif, pssm = ctcf
    consensus = str(motif.consensus)
    background = random_seqs(50, seed=2)
    planted = [s[:90] + consensus + s[90 + len(consensus):] for s in background]
    res_bg = score_sequences(onehot(background), pssm)
    res_pl = score_sequences(onehot(planted), pssm)
    best_possible = pssm.max(axis=0).sum()
    np.testing.assert_allclose(res_pl["score"], best_possible, rtol=1e-5)
    assert (res_pl["best_pos"] == 90).all() and (res_pl["best_strand"] == 0).all()
    assert res_pl["score"].min() > res_bg["score"].max()


def test_reverse_complement_scores_the_same(ctcf):
    _, pssm = ctcf
    seqs = random_seqs(20, seed=3)
    fwd = score_sequences(onehot(seqs), pssm)["score"]
    rev = score_sequences(onehot([reverse_complement(s) for s in seqs]), pssm)["score"]
    np.testing.assert_allclose(fwd, rev, rtol=1e-5)


def test_random_sequences_score_low(ctcf):
    _, pssm = ctcf
    scores = score_sequences(onehot(random_seqs(200, seed=4)), pssm)
    assert np.median(scores["score"]) < 0.6 * pssm.max(axis=0).sum()
    # At the 80% relative-score cutoff, chance hits in random DNA are uncommon but not zero.
    assert np.mean(scores["n_hits"] > 0) < 0.25


def test_planted_motif_counts_as_hit(ctcf):
    motif, pssm = ctcf
    consensus = str(motif.consensus)
    planted = [s[:50] + consensus + s[50 + len(consensus):] for s in random_seqs(20, seed=6)]
    assert (score_sequences(onehot(planted), pssm)["n_hits"] >= 1).all()


@pytest.mark.parametrize("bad", [np.zeros((2, 200)), np.zeros((2, 5, 200)), np.zeros((2, 4, 10))])
def test_invalid_shapes_raise(ctcf, bad):
    with pytest.raises(ValueError):
        window_scores(bad, ctcf[1])


def test_unknown_base_raises(ctcf):
    x = onehot(random_seqs(1, seed=5))
    x[0, :, 50] = 0  # an N: no base set at this position
    with pytest.raises(ValueError):
        window_scores(x, ctcf[1])


@pytest.mark.parametrize("tf", sorted(MOTIF_IDS))
def test_every_tf_motif_loads(tf):
    path = ROOT / "data" / "raw" / f"{MOTIF_IDS[tf]}.jaspar"
    if not path.exists():
        pytest.skip("run scripts/download_data.py first")
    pssm = pssm_matrix(load_motif(MOTIF_IDS[tf]))
    assert pssm.shape[0] == 4 and np.isfinite(pssm).all()
