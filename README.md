# PWM vs. HyenaDNA for CTCF binding-site prediction

CSS 497 capstone (Noah Medvinsky). This project reproduces the CTCF ChIP-seq task (Task 2) from
Tang et al. (2025), *Evaluating the representational power of pre-trained DNA language models for
regulatory genomics*, Genome Biology, [doi:10.1186/s13059-025-03674-8](https://doi.org/10.1186/s13059-025-03674-8).
It then compares the language model against a classical position weight matrix (PWM) built from the
JASPAR CTCF motif (MA0139).

- Original code: <https://github.com/amberT15/LLM_eval>
- Original data: Zenodo [10.5281/zenodo.8279715](https://doi.org/10.5281/zenodo.8279715)

## Dataset

The dataset is the authors' `CTCF_200.h5`, which they took from Majdandzic et al. (2023):

| Split | Sequences | Positive |
|-------|-----------|----------|
| train | 61,083    | 50.0%    |
| valid | 8,726     | 50.2%    |
| test  | 17,453    | 49.9%    |

Each example is a 200-nt sequence from GM12878 cells. Positives are CTCF ChIP-seq peaks. Negatives
are DNase I hypersensitive sites that don't overlap any ChIP-seq peak, down-sampled to a 1:1 balance.
The split is the authors' random 70/10/20 split, stored in the file.

The same archive has 9 more GM12878 transcription factors in the same format: ATF2, Arid3, BACH1,
ELK1, GABPA, MAX, REST, SRF and ZNF24. `download_data.py` keeps all 10. Five of their author-provided
splits contain a few exact duplicate sequences across splits: all positives, and at most 0.51% of a
test set (GABPA). CTCF has none. The splits are kept as published so results stay comparable to the
paper.

## Setup (local)

Requires Python 3.10+. From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
pip install -e .
python scripts/download_data.py
pytest
```

`download_data.py` downloads the authors' 486 MB data archive and checks its MD5 checksum. It keeps
the 10 ChIP-seq files (`data/raw/{TF}_200.h5`) and saves each TF's JASPAR motif into `data/raw/`.
Zenodo often drops long downloads; the script resumes them automatically, so rerun it if it stops.

## PWM baseline

```bash
python scripts/score_pwm.py
```

This scores every TF's test set with its JASPAR motif in a few seconds. It writes per-sequence
results to `results/predictions/{TF}_pwm.csv` and a summary to `results/tables/pwm_summary.csv`.

## HyenaDNA embeddings

```bash
python scripts/embed_hyena.py --tf CTCF
python scripts/train_cnn.py --tf CTCF --input hyena --h5 data/processed/CTCF_hyena.h5
```

HyenaDNA-tiny (`LongSafari/hyenadna-tiny-1k-seqlen-d256`, 1.6 million parameters) runs on a laptop
CPU at about 20 seconds per 1,000 sequences. The weights download automatically on first use. Full
CTCF embeddings take about 9 GB in `data/processed/`. The CNN reads them from disk in chunks, so
they don't need to fit in RAM.

## Reproducing the paper's CNN

```bash
python scripts/train_cnn.py --input onehot --seeds 0 1 2 3 4
```

This trains the authors' baseline CNN five times and writes per-sequence test predictions to
`results/predictions/` and one summary row per run to `results/tables/`. The published CTCF values,
read off Fig. 3, are in `results/tables/published_ctcf_auroc.csv`.

Differences from the original setup, all of which leave the method unchanged:

- **Software versions:** TensorFlow 2.21 with `tf-keras` (the Keras 2 API) instead of TensorFlow 2.12 on
  Python 3.9, which doesn't install on Python 3.12.
- **One-hot input orientation:** the h5 file stores sequences as (N, 4, 200). Keras `Conv1D` needs
  (N, 200, 4), so they're transposed. Without this, the original network can't run.
- **Dead branch removed:** the original `chip_cnn` builds a BatchNorm + 1×1 conv branch whose output is
  never used. It's omitted; `tests/test_models.py` checks that the network matches the original exactly.
- **Fixed random seeds:** seeds 0–4 are set for reproducibility. The authors didn't set seeds.
- **Exact AUROC:** AUROC is reported both from the Keras metric the authors used (a 200-threshold
  approximation) and from scikit-learn's exact computation.

## Layout

```
src/ctcf/      package code (data loading, PWM scoring, evaluation)
scripts/       command-line entry points (data download, experiment runs)
notebooks/     analysis notebooks, run in order
tests/         data-integrity, PWM, embedding and metric tests
data/raw/      downloaded inputs (gitignored)
data/processed cached embeddings and intermediate files (gitignored)
results/       per-sequence predictions, tables and figures for the paper
```
