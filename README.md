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
only `data/raw/CTCF_200.h5` and also saves the JASPAR motifs `MA0139.1` and `MA0139.2` into
`data/raw/`. Zenodo often drops long downloads; the script resumes them automatically, so rerun it
if it stops.

## Setup (Colab)

HyenaDNA embedding extraction runs on a Colab GPU. See `requirements-colab.txt`.

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
